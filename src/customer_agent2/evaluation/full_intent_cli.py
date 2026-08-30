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
    FullIntentReport,
    FullIntentRunError,
    run_full_intent_evaluation,
)
from customer_agent2.infrastructure.intents import load_default_intent_tree
from customer_agent2.infrastructure.models import OpenAICompatibleChatModel


def _live_classifier(
    settings: Settings,
) -> tuple[FastModelIntentClassifier, OpenAICompatibleChatModel]:
    if not settings.dashscope_api_key.get_secret_value().strip():
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            "完整 Intent 评测需要本地 API Key",
            retryable=False,
        )
    model = OpenAICompatibleChatModel(
        api_key=settings.dashscope_api_key,
        base_url=str(settings.dashscope_base_url).rstrip("/"),
        model_id=settings.chat_model_fast,
        timeout_seconds=settings.llm_timeout_seconds,
        first_packet_timeout_seconds=settings.llm_first_packet_timeout_seconds,
    )
    classifier = FastModelIntentClassifier(
        model,
        load_default_intent_tree(),
        high_confidence_threshold=settings.intent_high_confidence_threshold,
        ambiguity_margin=settings.intent_ambiguity_margin,
        timeout_seconds=settings.intent_timeout_seconds,
        max_output_tokens=settings.intent_max_output_tokens,
    )
    return classifier, model


async def _run(snapshot: Path, output: Path) -> FullIntentReport:
    settings = Settings()
    assets = load_full_evaluation_assets(snapshot)
    classifier, model = _live_classifier(settings)
    try:
        report = await run_full_intent_evaluation(assets.dataset, classifier)
    finally:
        await model.aclose()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
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
    arguments = parser.parse_args()
    if not arguments.live_intent or arguments.accept_paid_calls != EXPECTED_FULL_CASES:
        parser.error(
            f"真实 Intent 评测最多产生 {EXPECTED_FULL_CASES} 次付费调用; "
            f"必须显式传入 --live-intent --accept-paid-calls {EXPECTED_FULL_CASES}"
        )
    try:
        report = asyncio.run(_run(arguments.snapshot, arguments.output))
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
