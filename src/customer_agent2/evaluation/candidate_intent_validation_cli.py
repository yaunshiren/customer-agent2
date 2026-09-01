"""Run the M5-D v4 development and frozen challenge stages with exact cost limits."""

import argparse
import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from customer_agent2.config import Settings
from customer_agent2.domain.models import ModelError, ModelErrorCode
from customer_agent2.evaluation.candidate_intent_validation import (
    EXPECTED_CHALLENGE_CASES,
    CandidateIntentValidationCache,
    CandidateIntentValidationManifest,
    CandidateIntentValidationReport,
    CandidateIntentValidationStage,
    build_candidate_intent_validation_report,
    candidate_intent_validation_fingerprint,
    candidate_stage_samples,
    ensure_candidate_validation_stage_unlocked,
    load_candidate_intent_validation_manifest,
    pending_candidate_stage_samples,
    validate_candidate_intent_validation_cache,
    validate_candidate_intent_validation_manifest,
)
from customer_agent2.evaluation.full_dataset import (
    FullEvaluationDataset,
    load_full_evaluation_assets,
)
from customer_agent2.evaluation.full_intent import (
    FullIntentCaseResult,
    FullIntentEvaluationConfiguration,
    FullIntentFailedAttempt,
    FullIntentRunError,
    run_intent_case_evaluation,
)
from customer_agent2.evaluation.full_intent_cli import (
    build_live_intent_classifier,
    load_intent_tree_file,
)
from customer_agent2.evaluation.staged_intent import IntentStageGate
from customer_agent2.evaluation.staged_intent_cli import (
    staged_intent_cache_lock,
    validate_paid_call_acknowledgement,
)
from customer_agent2.infrastructure.intents import intent_tree_fingerprint

_DEFAULT_SNAPSHOT = Path("evaluation/datasets/ragenteval-v1")
_DEFAULT_MANIFEST = Path("evaluation/config/m5d-intent-v4-validation.json")
_DEFAULT_TREE = Path("evaluation/config/m5d-intent-tree-v4.json")
_DEFAULT_CACHE = Path("evaluation/reports/m5d-intent-v4.cache.json")


def new_candidate_intent_validation_cache(
    settings: Settings,
    manifest: CandidateIntentValidationManifest,
    dataset: FullEvaluationDataset,
    *,
    intent_tree_version: str,
    intent_tree_sha256: str,
    timeout_seconds: float,
) -> CandidateIntentValidationCache:
    """Create one immutable v4 cache identity."""
    return CandidateIntentValidationCache(
        dataset_id=manifest.version,
        manifest_version=manifest.version,
        validation_sha256=candidate_intent_validation_fingerprint(manifest, dataset),
        configuration=FullIntentEvaluationConfiguration(
            model_id=settings.chat_model_fast,
            intent_tree_version=intent_tree_version,
            intent_tree_sha256=intent_tree_sha256,
            high_confidence_threshold=settings.intent_high_confidence_threshold,
            ambiguity_margin=settings.intent_ambiguity_margin,
            timeout_seconds=timeout_seconds,
            max_output_tokens=settings.intent_max_output_tokens,
            temperature=0,
            reasoning_enabled=False,
        ),
    )


def load_candidate_intent_validation_cache(
    path: Path,
    expected: CandidateIntentValidationCache,
) -> CandidateIntentValidationCache:
    """Refuse cache reuse after any candidate or validation input changes."""
    if not path.exists():
        return expected
    cache = CandidateIntentValidationCache.model_validate_json(path.read_text(encoding="utf-8"))
    identity = cache.model_copy(update={"completed_cases": (), "failed_attempts": ()})
    if identity != expected:
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            "v4 验证缓存与当前清单、模型、候选树或参数不一致",
            retryable=False,
        )
    return cache


def write_candidate_intent_validation_cache(
    path: Path,
    cache: CandidateIntentValidationCache,
) -> None:
    """Atomically save every paid v4 result before another request starts."""
    _write_model(path, cache)


async def _run(arguments: argparse.Namespace) -> CandidateIntentValidationReport:
    settings = Settings()
    timeout_seconds = (
        settings.intent_timeout_seconds
        if arguments.intent_timeout_seconds is None
        else arguments.intent_timeout_seconds
    )
    if timeout_seconds <= 0 or timeout_seconds >= settings.llm_timeout_seconds:
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            "Intent 评测超时必须大于 0 且小于 Chat 客户端总超时",
            retryable=False,
        )
    output = arguments.output or _default_output(arguments.stage)
    validate_candidate_intent_validation_paths(
        snapshot=arguments.snapshot,
        manifest=arguments.manifest,
        intent_tree=arguments.intent_tree,
        cache=arguments.cache,
        output=output,
    )
    with staged_intent_cache_lock(arguments.cache):
        return await _run_locked(arguments, settings, timeout_seconds, output)


async def _run_locked(
    arguments: argparse.Namespace,
    settings: Settings,
    timeout_seconds: float,
    output: Path,
) -> CandidateIntentValidationReport:
    assets = load_full_evaluation_assets(arguments.snapshot)
    manifest = load_candidate_intent_validation_manifest(
        arguments.manifest.read_text(encoding="utf-8")
    )
    validate_candidate_intent_validation_manifest(manifest, assets.dataset)
    intent_tree = load_intent_tree_file(arguments.intent_tree)
    if intent_tree.version != "m5-d-v4-candidate":
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            "v4 聚焦验证只能使用 m5-d-v4-candidate 意图树",
            retryable=False,
        )
    expected_cache = new_candidate_intent_validation_cache(
        settings,
        manifest,
        assets.dataset,
        intent_tree_version=intent_tree.version,
        intent_tree_sha256=intent_tree_fingerprint(intent_tree),
        timeout_seconds=timeout_seconds,
    )
    cache = load_candidate_intent_validation_cache(arguments.cache, expected_cache)
    validate_candidate_intent_validation_cache(cache, manifest, assets.dataset)
    ensure_candidate_validation_stage_unlocked(
        arguments.stage,
        manifest,
        assets.dataset,
        cache,
    )
    targets = candidate_stage_samples(arguments.stage, manifest, assets.dataset)
    target_ids = {sample.query_id for sample in targets}
    cached_target_before_count = sum(case.query_id in target_ids for case in cache.completed_cases)
    pending = pending_candidate_stage_samples(
        arguments.stage,
        manifest,
        assets.dataset,
        cache,
    )
    validate_paid_call_acknowledgement(arguments.accept_paid_calls, len(pending))

    if pending:
        classifier, model = build_live_intent_classifier(
            settings,
            intent_tree,
            timeout_seconds=timeout_seconds,
        )

        def record(case: FullIntentCaseResult) -> None:
            nonlocal cache
            if case.degradation_reason is None:
                cache = CandidateIntentValidationCache(
                    **cache.model_dump(exclude={"completed_cases", "failed_attempts"}),
                    completed_cases=(*cache.completed_cases, case),
                    failed_attempts=cache.failed_attempts,
                )
            else:
                error_code = case.model_error_code
                cache = CandidateIntentValidationCache(
                    **cache.model_dump(exclude={"completed_cases", "failed_attempts"}),
                    completed_cases=cache.completed_cases,
                    failed_attempts=(
                        *cache.failed_attempts,
                        FullIntentFailedAttempt(
                            query_id=case.query_id,
                            error_code=(
                                error_code.value
                                if error_code is not None
                                else case.degradation_reason.value
                            ),
                        ),
                    ),
                )
            write_candidate_intent_validation_cache(arguments.cache, cache)

        try:
            await run_intent_case_evaluation(pending, classifier, on_case=record)
        finally:
            await model.aclose()

    report = build_candidate_intent_validation_report(
        arguments.stage,
        manifest,
        assets.dataset,
        cache,
        cached_target_before_count=cached_target_before_count,
        new_calls=len(pending),
    )
    _write_model(output, report)
    return report


def validate_candidate_intent_validation_paths(
    *,
    snapshot: Path,
    manifest: Path,
    intent_tree: Path,
    cache: Path,
    output: Path,
) -> None:
    """Keep local cache and reports away from immutable inputs."""
    protected = {manifest.resolve(), intent_tree.resolve()}
    outputs = {cache.resolve(), output.resolve()}
    if (
        protected.intersection(outputs)
        or len(outputs) != 2
        or any(path.is_relative_to(snapshot.resolve()) for path in outputs)
    ):
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            "v4 缓存和报告必须使用互不相同且不覆盖输入的独立路径",
            retryable=False,
        )


def _write_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def _default_output(stage: CandidateIntentValidationStage) -> Path:
    return Path(f"evaluation/reports/m5d-intent-v4-{stage.value}.json")


def _gate_summary(gate: IntentStageGate) -> dict[str, object]:
    return {
        "passed": gate.passed,
        "checks": {check.name: check.passed for check in gate.checks},
    }


def main() -> None:
    """Require exact approval for only 4 development or 20 challenge calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-intent", action="store_true")
    parser.add_argument("--stage", type=CandidateIntentValidationStage, required=True)
    parser.add_argument("--accept-paid-calls", type=int, default=0)
    parser.add_argument("--snapshot", type=Path, default=_DEFAULT_SNAPSHOT)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--intent-tree", type=Path, default=_DEFAULT_TREE)
    parser.add_argument("--cache", type=Path, default=_DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--intent-timeout-seconds", type=float, default=None)
    arguments = parser.parse_args()
    if (
        not arguments.live_intent
        or not 0 <= arguments.accept_paid_calls <= EXPECTED_CHALLENGE_CASES
    ):
        parser.error(
            "v4 聚焦验证必须显式传入 --live-intent, 并精确确认当前阶段缺少的最多 20 次调用"
        )
    try:
        report = asyncio.run(_run(arguments))
    except (FullIntentRunError, ModelError, OSError, TypeError, ValueError) as error:
        parser.exit(status=2, message=f"v4 Intent 验证已安全终止: {error}\n")
    print(
        json.dumps(
            {
                "dataset_id": report.dataset_id,
                "stage": report.stage.value,
                "target_sample_count": report.target_sample_count,
                "cached_target_before_count": report.cached_target_before_count,
                "new_calls": report.new_calls,
                "total_cached_count": report.total_cached_count,
                "failed_attempt_count": report.failed_attempt_count,
                "total_attempt_count": report.total_attempt_count,
                "remaining_candidate_calls": report.remaining_candidate_calls,
                "correct_count": report.overall.correct_count,
                "gate": _gate_summary(report.gate),
                "output": str(arguments.output or _default_output(arguments.stage)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
