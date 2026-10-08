"""Keep readable results in results/ and detailed computations in the local workspace."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
KEEP = {
    "analysis": [
        "inventory_daily_diagnostics.csv",
        "transaction_daily_quality.csv",
        "transaction_event_quality.csv",
        "transaction_category_structure.csv",
        "transaction_amount_distribution.csv",
        "session_device_standardization.csv",
        "session_segment_metrics.csv",
        "checkout_basket_sensitivity.csv",
        "checkout_basket_segments.csv",
        "checkout_native_price_audit.csv",
        "event_semantics_summary.csv",
        "checkout_unit_economics_scenarios.csv",
        "measurement_regimes.png",
        "transaction_quality_regimes.png",
        "checkout_basket_sensitivity.png",
        "checkout_concentrated_basket_block.png",
        "checkout_device_standardization.png",
        "event_item_semantics.png",
        "checkout_payment_opportunity.png",
    ],
    "validation": [
        "baseline_comparison.csv",
        "baseline_weekly_traffic.csv",
        "review_sensitivity_baselines.csv",
        "review_sensitivity_power.csv",
        "historical_aa_replay_summary.csv",
        "semi_synthetic_rct_summary.csv",
        "semi_synthetic_causal_summary.csv",
        "synthetic_sample_size_validation.csv",
        "experiment_sample_size_scenarios.csv",
        "business_threshold.json",
        "validation_config.json",
        "semi_synthetic_population_truth.csv",
        "semi_synthetic_causal_bias_coverage.png",
        "semi_synthetic_rct_detection.png",
    ],
    "overview": [
        "session_funnel.csv",
        "device_source_metrics.csv",
        "daily_metrics.csv",
        "daily_reconciliation.csv",
        "retention_cohorts.csv",
        "retention_summary.csv",
        "funnel_retention.png",
        "daily_trends.png",
    ],
}


def main():
    for folder, names in KEEP.items():
        dest = ROOT / "data/processed/details" / folder
        dest.mkdir(parents=True, exist_ok=True)
        for path in (ROOT / "results" / folder).iterdir():
            if path.is_file() and path.name not in names:
                shutil.move(str(path), str(dest / path.name))


if __name__ == "__main__":
    main()
