                      

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from xgb_protocol_ids_utils import (
    choose_default_protocol,
    discover_latest_hpo_run,
    global_feature_importance,
    load_feature_columns,
    load_thresholds_by_protocol,
    load_xgb_models_by_protocol,
    local_feature_contributions,
    mean_abs_contrib_by_feature,
    prepare_feature_matrix,
    routed_predict,
)


OUTCOME_ORDER = ("TP", "TN", "FP", "FN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to full_gpu_hpo_models_* run directory. Defaults to latest under reports/.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/merged/metadata_test.csv",
        help="CSV used to build local explanation cases and UI reference rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <run-dir>/xgb_explainability.",
    )
    parser.add_argument("--chunk-size", type=int, default=250000, help="CSV chunk size for streaming.")
    parser.add_argument("--cases-per-outcome", type=int, default=4, help="Number of TP/TN/FP/FN cases per protocol.")
    parser.add_argument(
        "--global-sample-per-protocol",
        type=int,
        default=4096,
        help="Rows per protocol used for mean absolute contribution estimates.",
    )
    parser.add_argument(
        "--reference-rows-per-protocol",
        type=int,
        default=200,
        help="Rows per protocol saved to reference_rows.csv for UI quick testing.",
    )
    parser.add_argument("--top-features", type=int, default=15, help="Top contributing features to store per local case.")
    return parser.parse_args()


def classify_outcome(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def to_int_label(value: object) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def normalize_scalar(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def enrich_row_for_output(row: Dict[str, object]) -> Dict[str, object]:
    return {k: normalize_scalar(v) for k, v in row.items()}


def read_csv_columns(csv_path: Path) -> List[str]:
    return pd.read_csv(csv_path, nrows=0).columns.astype(str).tolist()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else discover_latest_hpo_run("reports")
    if run_dir is None:
        raise RuntimeError("Could not find a run directory. Pass --run-dir explicitly.")
    if not run_dir.exists():
        raise RuntimeError(f"Run directory does not exist: {run_dir}")

    test_csv = Path(args.test_csv)
    if not test_csv.exists():
        raise RuntimeError(f"Test CSV does not exist: {test_csv}")

    output_dir = Path(args.output_dir) if args.output_dir else (run_dir / "xgb_explainability")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_columns(run_dir)
    models_by_protocol = load_xgb_models_by_protocol(run_dir)
    thresholds_by_protocol = load_thresholds_by_protocol(run_dir)
    protocols = sorted(models_by_protocol.keys())
    default_protocol = choose_default_protocol(protocols)

    test_cols = read_csv_columns(test_csv)
    metadata_cols = [
        "label",
        "protocol_hint",
        "attack_family",
        "attack_name",
        "source_relpath",
        "source_row_index",
    ]
    available_meta = [c for c in metadata_cols if c in test_cols]
    available_features = [c for c in feature_columns if c in test_cols]
    usecols = available_meta + available_features

    case_counts: Dict[Tuple[str, str], int] = {(p, o): 0 for p in protocols for o in OUTCOME_ORDER}
    global_samples: Dict[str, List[np.ndarray]] = {p: [] for p in protocols}
    global_sample_count: Dict[str, int] = {p: 0 for p in protocols}
    reference_rows: List[Dict[str, object]] = []
    reference_count: Dict[str, int] = {p: 0 for p in protocols}

    local_case_summary: List[Dict[str, object]] = []
    local_case_feature_rows: List[np.ndarray] = []
    local_case_raw_rows: List[Dict[str, object]] = []

    rows_processed = 0
    has_label = "label" in usecols

    for chunk in pd.read_csv(test_csv, usecols=usecols, chunksize=max(1, int(args.chunk_size))):
        if chunk.empty:
            continue
        chunk = chunk.reset_index(drop=True)
        pred_df = routed_predict(
            df=chunk.copy(),
            feature_columns=feature_columns,
            models_by_protocol=models_by_protocol,
            thresholds_by_protocol=thresholds_by_protocol,
            default_protocol=default_protocol,
        )
        chunk_enriched = pd.concat([chunk, pred_df], axis=1)
        x_chunk = prepare_feature_matrix(chunk.copy(), feature_columns)
        rows_processed += len(chunk_enriched)

        if has_label:
            y_true = chunk_enriched["label"].map(to_int_label).to_numpy(np.int8, copy=False)
        else:
            y_true = np.zeros(len(chunk_enriched), dtype=np.int8)
        y_pred = chunk_enriched["prediction"].to_numpy(np.int8, copy=False)
        protocol_used = chunk_enriched["protocol_used"].astype(str).to_numpy()
        outcomes = np.array(
            [classify_outcome(int(y_true[i]), int(y_pred[i])) for i in range(len(chunk_enriched))],
            dtype=object,
        )

        for proto in protocols:
            proto_idx = np.where(protocol_used == proto)[0]
            if proto_idx.size == 0:
                continue

            need_global = int(args.global_sample_per_protocol) - global_sample_count[proto]
            if need_global > 0:
                take = proto_idx[:need_global]
                if take.size > 0:
                    global_samples[proto].append(x_chunk[take].copy())
                    global_sample_count[proto] += int(take.size)

            need_ref = int(args.reference_rows_per_protocol) - reference_count[proto]
            if need_ref > 0:
                take_ref = proto_idx[:need_ref]
                for idx in take_ref.tolist():
                    row = chunk_enriched.iloc[idx].to_dict()
                    reference_rows.append(enrich_row_for_output(row))
                    reference_count[proto] += 1

            if has_label:
                for outcome in OUTCOME_ORDER:
                    need_case = int(args.cases_per_outcome) - case_counts[(proto, outcome)]
                    if need_case <= 0:
                        continue
                    match_idx = proto_idx[outcomes[proto_idx] == outcome]
                    take_case = match_idx[:need_case]
                    for idx in take_case.tolist():
                        row = chunk_enriched.iloc[idx]
                        case_counts[(proto, outcome)] += 1
                        case_id = f"{proto}_{outcome}_{case_counts[(proto, outcome)]:03d}"
                        local_case_summary.append(
                            {
                                "case_id": case_id,
                                "protocol_used": proto,
                                "outcome": outcome,
                                "label_true": int(to_int_label(row.get("label", 0))),
                                "prediction": int(to_int_label(row.get("prediction", 0))),
                                "score_attack": float(row.get("score_attack", 0.0)),
                                "threshold": float(row.get("threshold", thresholds_by_protocol.get(proto, 0.5))),
                                "attack_family": str(row.get("attack_family", "")),
                                "attack_name": str(row.get("attack_name", "")),
                                "source_relpath": str(row.get("source_relpath", "")),
                                "source_row_index": int(to_int_label(row.get("source_row_index", 0))),
                            }
                        )
                        local_case_feature_rows.append(x_chunk[idx].copy())
                        raw_case = row.to_dict()
                        raw_case["case_id"] = case_id
                        local_case_raw_rows.append(enrich_row_for_output(raw_case))

        cases_complete = all(
            case_counts[(p, o)] >= int(args.cases_per_outcome) for p in protocols for o in OUTCOME_ORDER
        ) if has_label else True
        global_complete = all(global_sample_count[p] >= int(args.global_sample_per_protocol) for p in protocols)
        ref_complete = all(reference_count[p] >= int(args.reference_rows_per_protocol) for p in protocols)
        if cases_complete and global_complete and ref_complete:
            break

    global_rows: List[pd.DataFrame] = []
    for proto in protocols:
        booster = models_by_protocol[proto]
        imp = global_feature_importance(booster, feature_columns)

        proto_samples = np.vstack(global_samples[proto]) if global_samples[proto] else np.empty((0, len(feature_columns)))
        mean_abs = mean_abs_contrib_by_feature(
            booster=booster,
            x_rows=proto_samples,
            feature_columns=feature_columns,
            max_rows=args.global_sample_per_protocol,
        )
        merged = imp.merge(mean_abs, on="feature", how="left")
        merged["mean_abs_contribution"] = merged["mean_abs_contribution"].fillna(0.0)
        merged["protocol_hint"] = proto
        merged["threshold"] = float(thresholds_by_protocol.get(proto, 0.5))
        merged = merged.sort_values(["mean_abs_contribution", "gain"], ascending=False).reset_index(drop=True)
        merged.insert(0, "rank", np.arange(1, len(merged) + 1))
        global_rows.append(merged)

    global_df = pd.concat(global_rows, ignore_index=True)
    global_path = output_dir / "global_feature_importance.csv"
    global_df.to_csv(global_path, index=False)

    contrib_rows: List[Dict[str, object]] = []
    for case, feat in zip(local_case_summary, local_case_feature_rows):
        proto = str(case["protocol_used"])
        booster = models_by_protocol[proto]
        contrib = local_feature_contributions(
            booster=booster,
            row_features=feat,
            feature_columns=feature_columns,
            top_n=args.top_features,
        )
        case_id = str(case["case_id"])
        for _, row in contrib.iterrows():
            contrib_rows.append(
                {
                    "case_id": case_id,
                    "protocol_used": proto,
                    "feature": str(row["feature"]),
                    "contribution": float(row["contribution"]),
                    "abs_contribution": float(row["abs_contribution"]),
                    "direction": str(row["direction"]),
                    "bias": float(row["bias"]),
                }
            )

    local_summary_cols = [
        "case_id",
        "protocol_used",
        "outcome",
        "label_true",
        "prediction",
        "score_attack",
        "threshold",
        "attack_family",
        "attack_name",
        "source_relpath",
        "source_row_index",
    ]
    local_case_summary_df = pd.DataFrame(local_case_summary, columns=local_summary_cols)
    local_summary_path = output_dir / "local_case_summary.csv"
    local_case_summary_df.to_csv(local_summary_path, index=False)

    local_contrib_cols = [
        "case_id",
        "protocol_used",
        "feature",
        "contribution",
        "abs_contribution",
        "direction",
        "bias",
    ]
    local_case_contrib_df = pd.DataFrame(contrib_rows, columns=local_contrib_cols)
    local_contrib_path = output_dir / "local_case_contributions.csv"
    local_case_contrib_df.to_csv(local_contrib_path, index=False)

    local_case_raw_df = pd.DataFrame(local_case_raw_rows)
    if local_case_raw_df.empty:
        local_case_raw_df = pd.DataFrame(columns=["case_id"])
    local_raw_path = output_dir / "local_case_reference_rows.csv"
    local_case_raw_df.to_csv(local_raw_path, index=False)

    reference_df = pd.DataFrame(reference_rows)
    if reference_df.empty:
        reference_df = pd.DataFrame(columns=["protocol_used", "score_attack", "threshold", "prediction"])
    reference_path = output_dir / "reference_rows.csv"
    reference_df.to_csv(reference_path, index=False)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "test_csv": str(test_csv),
        "rows_processed": int(rows_processed),
        "protocols": protocols,
        "default_protocol": default_protocol,
        "feature_count": int(len(feature_columns)),
        "thresholds_by_protocol": {k: float(v) for k, v in thresholds_by_protocol.items()},
        "global_sample_count_by_protocol": {k: int(v) for k, v in global_sample_count.items()},
        "reference_rows_by_protocol": {k: int(v) for k, v in reference_count.items()},
        "cases_by_protocol_outcome": {f"{k[0]}::{k[1]}": int(v) for k, v in case_counts.items()},
        "files": {
            "global_feature_importance_csv": str(global_path),
            "local_case_summary_csv": str(local_summary_path),
            "local_case_contributions_csv": str(local_contrib_path),
            "local_case_reference_rows_csv": str(local_raw_path),
            "reference_rows_csv": str(reference_path),
        },
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    manifest = {
        "kind": "xgb_protocol_explainability_bundle",
        "version": 1,
        "generated_at": summary["generated_at"],
        "run_dir": summary["run_dir"],
        "files": summary["files"],
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(str(output_dir))


if __name__ == "__main__":
    main()
