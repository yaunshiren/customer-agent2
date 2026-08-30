"""Run the explicitly authorized 150-call M5-C live intent evaluation."""

import argparse
import asyncio
import json
from pathlib import Path

from customer_agent2.application import FastModelIntentClassifier
from customer_agent2.config import Settings
from customer_agent2.domain.models import ModelError, ModelErrorCode
from customer_agent2.evaluation.full_dataset import (
    EXPECTED_FULL_CASES,
    load_full_evaluation_assets,
)
from customer_agent2.evaluation.full_intent import (
    FullIntentCaseResult,
    FullIntentCheckpoint,
    FullIntentEvaluationConfiguration,
    FullIntentFailedAttempt,
    FullIntentReport,
    FullIntentRunError,
    run_full_intent_evaluation,
)
from customer_agent2.infrastructure.intents import load_default_intent_tree
from customer_agent2.infrastructure.models import OpenAICompatibleChatModel


def _live_classifier(
    settings: Settings,
    *,
    timeout_seconds: float,
) -> tuple[FastModelIntentClassifier, OpenAICompatibleChatModel]:
    if not settings.dashscope_api_key.get_secret_value().strip():
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            "完整 Intent 评测需要本地 API Key",
            retryable=False,
        )
    model = OpenAICompatibleChatModel(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_chat_api_base_url,
        model_id=settings.chat_model_fast,
        timeout_seconds=settings.llm_timeout_seconds,
        first_packet_timeout_seconds=settings.llm_first_packet_timeout_seconds,
    )
    classifier = FastModelIntentClassifier(
        model,
        load_default_intent_tree(),
        high_confidence_threshold=settings.intent_high_confidence_threshold,
        ambiguity_margin=settings.intent_ambiguity_margin,
        timeout_seconds=timeout_seconds,
        max_output_tokens=settings.intent_max_output_tokens,
    )
    return classifier, model


def new_intent_checkpoint(
    settings: Settings,
    dataset_id: str,
    *,
    timeout_seconds: float,
) -> FullIntentCheckpoint:
    return FullIntentCheckpoint(
        dataset_id=dataset_id,
        model_id=settings.chat_model_fast,
        high_confidence_threshold=settings.intent_high_confidence_threshold,
        ambiguity_margin=settings.intent_ambiguity_margin,
        timeout_seconds=timeout_seconds,
        max_output_tokens=settings.intent_max_output_tokens,
        reasoning_enabled=False,
    )


def load_intent_checkpoint(
    path: Path,
    settings: Settings,
    dataset_id: str,
    *,
    timeout_seconds: float,
) -> FullIntentCheckpoint:
    expected = new_intent_checkpoint(settings, dataset_id, timeout_seconds=timeout_seconds)
    if not path.exists():
        return expected
    checkpoint = FullIntentCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
    if checkpoint.model_copy(update={"completed_cases": (), "failed_attempts": ()}) != expected:
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            "Intent checkpoint 与当前数据集或模型配置不一致",
            retryable=False,
        )
    return checkpoint


def write_intent_checkpoint(path: Path, checkpoint: FullIntentCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


async def _run(
    snapshot: Path,
    output: Path,
    checkpoint_path: Path,
    accepted_paid_calls: int,
    intent_timeout_seconds: float | None,
) -> FullIntentReport:
    settings = Settings()
    timeout_seconds = (
        settings.intent_timeout_seconds
        if intent_timeout_seconds is None
        else intent_timeout_seconds
    )
    if timeout_seconds <= 0 or timeout_seconds >= settings.llm_timeout_seconds:
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            "Intent 评测超时必须大于 0 且小于 Chat 客户端总超时",
            retryable=False,
        )
    assets = load_full_evaluation_assets(snapshot)
    checkpoint = load_intent_checkpoint(
        checkpoint_path,
        settings,
        assets.dataset.dataset_id,
        timeout_seconds=timeout_seconds,
    )
    remaining_calls = EXPECTED_FULL_CASES - len(checkpoint.completed_cases)
    if accepted_paid_calls != remaining_calls:
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            f"当前 checkpoint 需要精确确认 {remaining_calls} 次 Intent 调用",
            retryable=False,
        )
    classifier, model = _live_classifier(settings, timeout_seconds=timeout_seconds)

    def record(case: FullIntentCaseResult) -> None:
        nonlocal checkpoint
        if case.degradation_reason is None:
            checkpoint = FullIntentCheckpoint(
                **checkpoint.model_dump(exclude={"completed_cases", "failed_attempts"}),
                completed_cases=(*checkpoint.completed_cases, case),
                failed_attempts=checkpoint.failed_attempts,
            )
        else:
            error_code = case.model_error_code
            checkpoint = FullIntentCheckpoint(
                **checkpoint.model_dump(exclude={"completed_cases", "failed_attempts"}),
                completed_cases=checkpoint.completed_cases,
                failed_attempts=(
                    *checkpoint.failed_attempts,
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
        write_intent_checkpoint(checkpoint_path, checkpoint)

    try:
        report = await run_full_intent_evaluation(
            assets.dataset,
            classifier,
            configuration=FullIntentEvaluationConfiguration(
                model_id=settings.chat_model_fast,
                high_confidence_threshold=settings.intent_high_confidence_threshold,
                ambiguity_margin=settings.intent_ambiguity_margin,
                timeout_seconds=timeout_seconds,
                max_output_tokens=settings.intent_max_output_tokens,
                temperature=0,
                reasoning_enabled=False,
            ),
            initial_cases=checkpoint.completed_cases,
            on_case=record,
        )
    finally:
        await model.aclose()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    checkpoint_path.unlink(missing_ok=True)
    return report


def main() -> None:
    """Require an exact paid-call acknowledgement before any live request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-intent", action="store_true")
    parser.add_argument("--accept-paid-calls", type=int, default=0)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("evaluation/datasets/ragenteval-v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/reports/m5c-full-intent.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("evaluation/reports/m5c-full-intent.checkpoint.json"),
    )
    parser.add_argument(
        "--intent-timeout-seconds",
        type=float,
        default=None,
        help="仅覆盖本次评测的单条 Intent 超时, 不修改线上默认值",
    )
    arguments = parser.parse_args()
    if not arguments.live_intent or not 0 <= arguments.accept_paid_calls <= EXPECTED_FULL_CASES:
        parser.error(
            f"真实 Intent 单次运行最多产生 {EXPECTED_FULL_CASES} 次付费调用; "
            "必须显式传入 --live-intent 和与 checkpoint 剩余样本数完全一致的 "
            "--accept-paid-calls"
        )
    try:
        report = asyncio.run(
            _run(
                arguments.snapshot,
                arguments.output,
                arguments.checkpoint,
                arguments.accept_paid_calls,
                arguments.intent_timeout_seconds,
            )
        )
    except (FullIntentRunError, ModelError) as error:
        parser.exit(status=2, message=f"M5-C Intent 已安全终止: {error}\n")
    print(
        json.dumps(
            {
                "dataset_id": report.dataset_id,
                "sample_count": report.sample_count,
                "successful_calls": report.successful_calls,
                "failed_calls": report.failed_calls,
                "input_tokens": report.input_tokens,
                "output_tokens": report.output_tokens,
                "overall": report.overall.model_dump(),
                "rag": report.rag.model_dump(),
                "no_rag": report.no_rag.model_dump(),
                "output": str(arguments.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
