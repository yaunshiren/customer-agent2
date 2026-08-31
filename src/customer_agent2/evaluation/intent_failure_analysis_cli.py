"""Generate the offline M5-D Intent failure-slice report."""

import argparse
import json
from pathlib import Path

from customer_agent2.evaluation.full_dataset import load_full_evaluation_assets
from customer_agent2.evaluation.full_intent import FullIntentReport
from customer_agent2.evaluation.intent_failure_analysis import analyze_full_intent_failures


def main() -> None:
    """Validate the source report and write one content-free deterministic analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("evaluation/datasets/ragenteval-v1"),
    )
    parser.add_argument(
        "--intent-report",
        type=Path,
        default=Path("evaluation/reports/m5c-full-intent.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/reports/m5d-intent-failure-analysis.json"),
    )
    arguments = parser.parse_args()

    assets = load_full_evaluation_assets(arguments.snapshot)
    report = FullIntentReport.model_validate_json(
        arguments.intent_report.read_text(encoding="utf-8")
    )
    analysis = analyze_full_intent_failures(assets.dataset, report)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset_id": analysis.dataset_id,
                "incorrect_count": analysis.incorrect_count,
                "over_retrieval_count": analysis.over_retrieval_count,
                "under_retrieval_count": analysis.under_retrieval_count,
                "incorrect_clarification_count": analysis.incorrect_clarification_count,
                "output": str(arguments.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
