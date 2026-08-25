"""Pinned Hugging Face tokenizer adapter for exact BGE token budgets."""

from collections.abc import Callable, Sequence
from threading import Lock
from typing import Protocol, cast

from customer_agent2.config import Settings
from customer_agent2.domain.models import ChunkingError, ChunkingErrorCode


class TransformersTokenizerBackend(Protocol):
    """Narrow tokenizer surface needed by the document chunker."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


TokenizerBackendFactory = Callable[[str, str], TransformersTokenizerBackend]


class TransformersTextTokenCodec:
    """Lazily load one pinned tokenizer and expose a provider-neutral codec."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        backend_factory: TokenizerBackendFactory | None = None,
    ) -> None:
        normalized_model_id = model_id.strip()
        normalized_revision = revision.strip()
        if not normalized_model_id or not normalized_revision:
            raise ValueError("分词器模型 ID 和 revision 不能为空")
        self._model_id = normalized_model_id
        self._revision = normalized_revision
        self._backend_factory = backend_factory or _load_tokenizer
        self._backend: TransformersTokenizerBackend | None = None
        self._load_lock = Lock()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        backend_factory: TokenizerBackendFactory | None = None,
    ) -> "TransformersTextTokenCodec":
        """Use the same pinned identity as the configured embedding model."""
        return cls(
            model_id=settings.local_embedding_model,
            revision=settings.local_embedding_revision,
            backend_factory=backend_factory,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode text without model-added special tokens."""
        try:
            encoded = self._get_backend().encode(text, add_special_tokens=False)
        except ChunkingError:
            raise
        except Exception:
            raise ChunkingError(
                ChunkingErrorCode.TOKENIZER_PROTOCOL,
                "本地分词器编码失败",
            ) from None
        return tuple(encoded)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode an exact token window without adding or cleaning special tokens."""
        if not token_ids:
            raise ValueError("token_ids 不能为空")
        try:
            return self._get_backend().decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except ChunkingError:
            raise
        except Exception:
            raise ChunkingError(
                ChunkingErrorCode.TOKENIZER_PROTOCOL,
                "本地分词器解码失败",
            ) from None

    def _get_backend(self) -> TransformersTokenizerBackend:
        if self._backend is not None:
            return self._backend
        with self._load_lock:
            if self._backend is not None:
                return self._backend
            try:
                backend = self._backend_factory(self._model_id, self._revision)
            except Exception:
                raise ChunkingError(
                    ChunkingErrorCode.TOKENIZER_UNAVAILABLE,
                    "本地分词器不可用",
                ) from None
            self._backend = backend
            return backend


def _load_tokenizer(model_id: str, revision: str) -> TransformersTokenizerBackend:
    from transformers import AutoTokenizer

    loader = cast(
        Callable[..., object],
        AutoTokenizer.from_pretrained,  # pyright: ignore[reportUnknownMemberType]
    )
    tokenizer = loader(
        model_id,
        revision=revision,
        trust_remote_code=False,
        use_fast=True,
    )
    return cast(TransformersTokenizerBackend, tokenizer)
