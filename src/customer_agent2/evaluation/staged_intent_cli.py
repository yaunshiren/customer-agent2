"""Run the budget-aware M5-D Intent evaluation one authorized stage at a time."""

import argparse
import asyncio
import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel

from customer_agent2.config import Settings
from customer_agent2.domain.models import ModelError, ModelErrorCode
from customer_agent2.evaluation.full_dataset import (
    EXPECTED_FULL_CASES,
    load_full_evaluation_assets,
)
from customer_agent2.evaluation.full_intent import (
    FullIntentCaseResult,
    FullIntentEvaluationConfiguration,
    FullIntentFailedAttempt,
    FullIntentReport,
    FullIntentRunError,
    build_full_intent_report,
    run_intent_case_evaluation,
)
from customer_agent2.evaluation.full_intent_cli import (
    build_live_intent_classifier,
    load_intent_tree_file,
)
from customer_agent2.evaluation.staged_intent import (
    IntentEvaluationStage,
    M5DIntentStageManifest,
    StagedIntentCache,
    StagedIntentReport,
    baseline_report_fingerprint,
    build_staged_intent_report,
    ensure_stage_unlocked,
    load_m5d_stage_manifest,
    ordered_full_cache_cases,
    pending_stage_samples,
    stage_query_ids,
    staged_manifest_fingerprint,
    validate_m5d_stage_manifest,
    validate_staged_intent_cache,
)
from customer_agent2.infrastructure.intents import intent_tree_fingerprint

_DEFAULT_SNAPSHOT = Path("evaluation/datasets/ragenteval-v1")
_DEFAULT_BASELINE = Path("evaluation/reports/m5c-full-intent.json")
_DEFAULT_MANIFEST = Path("evaluation/config/m5d-intent-stages.json")
_DEFAULT_TREE = Path("evaluation/config/m5d-intent-tree-v3.json")
_DEFAULT_CACHE = Path("evaluation/reports/m5d-intent-v3.cache.json")
_DEFAULT_FULL_OUTPUT = Path("evaluation/reports/m5d-full-intent-v3.json")


def new_staged_intent_cache(
    settings: Settings,
    dataset_id: str,
    manifest: M5DIntentStageManifest,
    baseline: FullIntentReport,
    *,
    intent_tree_version: str,
    intent_tree_sha256: str,
    timeout_seconds: float,
) -> StagedIntentCache:
    """Create the exact immutable identity for one candidate experiment."""
    return StagedIntentCache(
        dataset_id=dataset_id,
        manifest_version=manifest.version,
        manifest_sha256=staged_manifest_fingerprint(manifest),
        baseline_report_sha256=baseline_report_fingerprint(baseline),
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


def load_staged_intent_cache(path: Path, expected: StagedIntentCache) -> StagedIntentCache:
    """Load a cache only when every cost-affecting input matches exactly."""
    if not path.exists():
        return expected
    cache = StagedIntentCache.model_validate_json(path.read_text(encoding="utf-8"))
    identity = cache.model_copy(update={"completed_cases": (), "failed_attempts": ()})
    if identity != expected:
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            "Intent 阶段缓存与当前数据、模型、候选树或参数不一致",
            retryable=False,
        )
    return cache


def write_staged_intent_cache(path: Path, cache: StagedIntentCache) -> None:
    """Atomically persist paid successes before another request can start."""
    _write_model(path, cache)


@contextmanager
def staged_intent_cache_lock(path: Path) -> Generator[None, None, None]:
    """Prevent two processes from paying for the same uncached Query IDs."""
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.touch(exist_ok=False)
    except FileExistsError as error:
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            "Intent 阶段缓存正在被另一个进程使用或存在未清理的锁文件",
            retryable=False,
        ) from error
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


async def _run(arguments: argparse.Namespace) -> tuple[StagedIntentReport, FullIntentReport | None]:
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
    output = arguments.output or _default_stage_output(arguments.stage)
    validate_staged_intent_paths(
        snapshot=arguments.snapshot,
        baseline_report=arguments.baseline_report,
        stage_manifest=arguments.stage_manifest,
        intent_tree=arguments.intent_tree,
        cache=arguments.cache,
        output=output,
        full_output=arguments.full_output,
    )
    with staged_intent_cache_lock(arguments.cache):
        return await _run_locked(arguments, settings, timeout_seconds, output)


async def _run_locked(
    arguments: argparse.Namespace,
    settings: Settings,
    timeout_seconds: float,
    output: Path,
) -> tuple[StagedIntentReport, FullIntentReport | None]:
    """Execute one stage while holding the candidate cache's exclusive lock."""
    assets = load_full_evaluation_assets(arguments.snapshot)
    baseline = FullIntentReport.model_validate_json(
        arguments.baseline_report.read_text(encoding="utf-8")
    )
    manifest = load_m5d_stage_manifest(arguments.stage_manifest.read_text(encoding="utf-8"))
    validate_m5d_stage_manifest(manifest, assets.dataset, baseline)
    intent_tree = load_intent_tree_file(arguments.intent_tree)
    expected_cache = new_staged_intent_cache(
        settings,
        assets.dataset.dataset_id,
        manifest,
        baseline,
        intent_tree_version=intent_tree.version,
        intent_tree_sha256=intent_tree_fingerprint(intent_tree),
        timeout_seconds=timeout_seconds,
    )
    cache = load_staged_intent_cache(arguments.cache, expected_cache)
    validate_staged_intent_cache(cache, assets.dataset)
    ensure_stage_unlocked(arguments.stage, manifest, assets.dataset, cache)

    target_ids = set(stage_query_ids(arguments.stage, manifest, assets.dataset))
    cached_target_before_count = sum(case.query_id in target_ids for case in cache.completed_cases)
    pending = pending_stage_samples(arguments.stage, manifest, assets.dataset, cache)
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
                cache = StagedIntentCache(
                    **cache.model_dump(exclude={"completed_cases", "failed_attempts"}),
                    completed_cases=(*cache.completed_cases, case),
                    failed_attempts=cache.failed_attempts,
                )
            else:
                error_code = case.model_error_code
                cache = StagedIntentCache(
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
            write_staged_intent_cache(arguments.cache, cache)

        try:
            await run_intent_case_evaluation(pending, classifier, on_case=record)
        finally:
            await model.aclose()

    report = build_staged_intent_report(
        arguments.stage,
        manifest,
        assets.dataset,
        cache,
        cached_target_before_count=cached_target_before_count,
        new_calls=len(pending),
    )
    _write_model(output, report)

    full_report: FullIntentReport | None = None
    if arguments.stage is IntentEvaluationStage.FULL:
        full_report = build_full_intent_report(
            assets.dataset,
            ordered_full_cache_cases(assets.dataset, cache),
            configuration=cache.configuration,
        )
        _write_model(arguments.full_output, full_report)
    return report, full_report


def _write_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def _default_stage_output(stage: IntentEvaluationStage) -> Path:
    return Path(f"evaluation/reports/m5d-intent-v3-{stage.value}.json")


def validate_paid_call_acknowledgement(accepted: int, required: int) -> None:
    """Reject an approval that does not equal the exact uncached request count."""
    if accepted != required:
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            f"当前阶段和缓存需要精确确认 {required} 次 Intent 调用",
            retryable=False,
        )


def validate_staged_intent_paths(
    *,
    snapshot: Path,
    baseline_report: Path,
    stage_manifest: Path,
    intent_tree: Path,
    cache: Path,
    output: Path,
    full_output: Path,
) -> None:
    """Prevent reports and caches from overwriting immutable experiment inputs."""
    protected = {
        baseline_report.resolve(),
        stage_manifest.resolve(),
        intent_tree.resolve(),
    }
    outputs = {
        cache.resolve(),
        output.resolve(),
        full_output.resolve(),
    }
    snapshot_root = snapshot.resolve()
    if (
        protected.intersection(outputs)
        or any(path.is_relative_to(snapshot_root) for path in outputs)
        or len(outputs) != 3
    ):
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            "M5-D 缓存、阶段报告和完整报告必须使用互不相同的独立路径",
            retryable=False,
        )


def main() -> None:
    """Require exact approval for only the currently missing stage cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-intent", action="store_true")
    parser.add_argument("--stage", type=IntentEvaluationStage, required=True)
    parser.add_argument("--accept-paid-calls", type=int, default=0)
    parser.add_argument("--snapshot", type=Path, default=_DEFAULT_SNAPSHOT)
    parser.add_argument("--baseline-report", type=Path, default=_DEFAULT_BASELINE)
    parser.add_argument("--stage-manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--intent-tree", type=Path, default=_DEFAULT_TREE)
    parser.add_argument("--cache", type=Path, default=_DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--full-output", type=Path, default=_DEFAULT_FULL_OUTPUT)
    parser.add_argument(
        "--intent-timeout-seconds",
        type=float,
        default=None,
        help="仅覆盖本次分阶段评测的单条 Intent 超时, 不修改线上默认值",
    )
    arguments = parser.parse_args()
    if not arguments.live_intent or not 0 <= arguments.accept_paid_calls <= EXPECTED_FULL_CASES:
        parser.error(
            "分阶段真实 Intent 必须显式传入 --live-intent, 并用 "
            "--accept-paid-calls 精确确认当前缓存缺少的调用数"
        )
    try:
        report, full_report = asyncio.run(_run(arguments))
    except (FullIntentRunError, ModelError, OSError, TypeError, ValueError) as error:
        parser.exit(status=2, message=f"M5-D Intent 已安全终止: {error}\n")
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
                "remaining_full_calls": report.remaining_full_calls,
                "correct_count": report.overall.correct_count,
                "gate_passed": report.gate.passed,
                "full_report_written": full_report is not None,
                "output": str(arguments.output or _default_stage_output(arguments.stage)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
