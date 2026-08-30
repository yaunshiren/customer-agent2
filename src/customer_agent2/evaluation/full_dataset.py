"""Strict loader for the versioned 150-case M5-C evaluation snapshot."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

EXPECTED_FULL_CASES = 150
EXPECTED_SMOKE_CASES = 20
EXPECTED_RAG_CASES = 132
EXPECTED_NO_RAG_CASES = 18
EXPECTED_DOCUMENTS = 116

EvaluationCategory = Literal["01_product", "02_manual", "03_policy", "04_faq"]
_CATEGORIES: tuple[EvaluationCategory, ...] = (
    "01_product",
    "02_manual",
    "03_policy",
    "04_faq",
)
_DOC_ID_PATTERN = re.compile(r"^doc_id:\s*([A-Z][A-Z0-9_]+)\s*$", re.MULTILINE)


class FullEvaluationSample(BaseModel):
    """One immutable input and gold-label record from the original dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=10_000)
    intent_l1: Literal["SUPPORT", "FEEDBACK", "CHAT"]
    intent_l2: str = Field(min_length=1, max_length=100)
    difficulty: Literal["easy", "medium", "hard"]
    requires_rag: bool
    requires_tool: bool = False
    expected_answer_type: str = Field(min_length=1, max_length=100)
    expected_doc_ids: tuple[str, ...]
    expected_doc_ids_nice: tuple[str, ...] = ()
    trap_type: str = Field(min_length=1, max_length=100)
    ground_truth: str = Field(min_length=1, max_length=20_000)
    eval_metrics: tuple[str, ...]

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        required = self.expected_doc_ids
        nice = self.expected_doc_ids_nice
        if len(set(required)) != len(required) or len(set(nice)) != len(nice):
            raise ValueError("评测文档标签不能重复")
        if set(required).intersection(nice):
            raise ValueError("required 和 nice 文档标签不能重叠")
        if self.requires_rag != bool(required):
            raise ValueError("requires_rag 必须与 required 文档标签一致")
        if not self.eval_metrics or any(not metric.strip() for metric in self.eval_metrics):
            raise ValueError("eval_metrics 必须非空且不能包含空值")
        if self.requires_tool and "tool_routing" not in self.eval_metrics:
            raise ValueError("requires_tool 样本必须声明 tool_routing 指标")
        return self


class FullEvaluationDataset(BaseModel):
    """Validated full dataset plus the IDs of its legacy 20-case quick set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = "ragenteval-v1-all"
    samples: tuple[FullEvaluationSample, ...]
    smoke_query_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if len(self.samples) != EXPECTED_FULL_CASES:
            raise ValueError(f"完整评测集必须恰好包含 {EXPECTED_FULL_CASES} 条")
        query_ids = tuple(sample.query_id for sample in self.samples)
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("完整评测集 query_id 不能重复")
        if len(self.smoke_query_ids) != EXPECTED_SMOKE_CASES:
            raise ValueError(f"快速子集必须恰好包含 {EXPECTED_SMOKE_CASES} 条")
        if len(set(self.smoke_query_ids)) != len(self.smoke_query_ids):
            raise ValueError("快速子集 query_id 不能重复")
        if not set(self.smoke_query_ids).issubset(query_ids):
            raise ValueError("快速子集必须属于完整评测集")
        rag_count = sum(sample.requires_rag for sample in self.samples)
        if rag_count != EXPECTED_RAG_CASES:
            raise ValueError(f"完整评测集必须包含 {EXPECTED_RAG_CASES} 条 RAG 样本")
        if len(self.samples) - rag_count != EXPECTED_NO_RAG_CASES:
            raise ValueError(f"完整评测集必须包含 {EXPECTED_NO_RAG_CASES} 条 no-rag 样本")
        return self

    @property
    def rag_samples(self) -> tuple[FullEvaluationSample, ...]:
        """Return only samples that belong in the retrieval denominator."""
        return tuple(sample for sample in self.samples if sample.requires_rag)


@dataclass(frozen=True, slots=True)
class EvaluationDocument:
    """One source document with a stable business ID and snapshot path."""

    document_id: str
    category: EvaluationCategory
    path: Path
    relative_path: str
    content_sha256: str

    def __post_init__(self) -> None:
        if not self.document_id.strip() or not self.relative_path.strip():
            raise ValueError("评测文档 ID 和路径不能为空")
        if len(self.content_sha256) != 64:
            raise ValueError("评测文档 SHA-256 无效")


@dataclass(frozen=True, slots=True)
class FullEvaluationAssets:
    """Validated dataset and corpus ready for import or scoring."""

    dataset: FullEvaluationDataset
    documents: tuple[EvaluationDocument, ...]

    def __post_init__(self) -> None:
        if len(self.documents) != EXPECTED_DOCUMENTS:
            raise ValueError(f"评测语料必须恰好包含 {EXPECTED_DOCUMENTS} 篇文档")
        document_ids = tuple(document.document_id for document in self.documents)
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("评测语料文档 ID 不能重复")
        available = set(document_ids)
        referenced = {
            document_id
            for sample in self.dataset.samples
            for document_id in (*sample.expected_doc_ids, *sample.expected_doc_ids_nice)
        }
        missing = sorted(referenced - available)
        if missing:
            raise ValueError(f"评测集引用了不存在的文档: {','.join(missing)}")


def load_full_evaluation_assets(snapshot_root: Path) -> FullEvaluationAssets:
    """Load and cross-check the immutable JSONL datasets and Markdown corpus."""
    full_samples = _load_jsonl(snapshot_root / "eval_set_v1_all.jsonl")
    smoke_samples = _load_jsonl(snapshot_root / "eval_set_v1.jsonl")
    dataset = FullEvaluationDataset(
        samples=full_samples,
        smoke_query_ids=tuple(sample.query_id for sample in smoke_samples),
    )
    return FullEvaluationAssets(
        dataset=dataset,
        documents=_load_documents(snapshot_root / "knowledge_base"),
    )


def _load_jsonl(path: Path) -> tuple[FullEvaluationSample, ...]:
    if not path.is_file():
        raise ValueError(f"评测数据文件不存在: {path.name}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"评测数据文件为空或包含空行: {path.name}")
    return tuple(FullEvaluationSample.model_validate_json(line) for line in lines)


def _load_documents(knowledge_root: Path) -> tuple[EvaluationDocument, ...]:
    documents: list[EvaluationDocument] = []
    for category in _CATEGORIES:
        category_root = knowledge_root / category
        if not category_root.is_dir():
            raise ValueError(f"缺少评测知识目录: {category}")
        for path in sorted(category_root.rglob("*.md")):
            content = path.read_bytes()
            documents.append(
                EvaluationDocument(
                    document_id=_front_matter_document_id(content, path),
                    category=category,
                    path=path,
                    relative_path=path.relative_to(knowledge_root).as_posix(),
                    content_sha256=hashlib.sha256(content).hexdigest(),
                )
            )

    mapping_path = knowledge_root / "_meta" / "product_mapping.md"
    if not mapping_path.is_file():
        raise ValueError("缺少 PRODUCT_MAPPING 特殊文档")
    mapping_content = mapping_path.read_bytes()
    documents.append(
        EvaluationDocument(
            document_id="PRODUCT_MAPPING",
            category="01_product",
            path=mapping_path,
            relative_path=mapping_path.relative_to(knowledge_root).as_posix(),
            content_sha256=hashlib.sha256(mapping_content).hexdigest(),
        )
    )
    return tuple(documents)


def _front_matter_document_id(content: bytes, path: Path) -> str:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"评测 Markdown 不是 UTF-8: {path.name}") from None
    matches = _DOC_ID_PATTERN.findall(text)
    if len(matches) != 1:
        raise ValueError(f"评测 Markdown 必须包含唯一 doc_id: {path.name}")
    return matches[0]
