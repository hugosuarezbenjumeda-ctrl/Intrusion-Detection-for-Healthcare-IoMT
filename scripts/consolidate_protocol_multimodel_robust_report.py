                      

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pandas.errors import EmptyDataError


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="Path to protocol multimodel robust matrix run directory.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise RuntimeError(f"Run directory not found: {run_dir}")

    global_df = _read_csv(run_dir / "decision_table_global.csv")
    stability_df = _read_csv(run_dir / "stability_check_global.csv")
    consistency_df = _read_csv(run_dir / "stability_consistency_summary.csv")

    escalation_path = run_dir / "escalation_recommendation.json"
    escalation: Dict[str, Any] = {}
    if escalation_path.exists():
        with escalation_path.open("r", encoding="utf-8") as f:
            escalation = json.load(f)

    top_global: List[Dict[str, Any]] = []
    if not global_df.empty:
        top_global = global_df.sort_values("final_rank", ascending=True).head(10).to_dict(orient="records")

    stable_candidates: List[Dict[str, Any]] = []
    if not consistency_df.empty:
        stable_candidates = consistency_df[consistency_df["consistent_gate_pass"] == True].to_dict(orient="records")              

    out_payload = {
        "generated_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "top_global_candidates": top_global,
        "stable_candidates": stable_candidates,
        "global_rows": int(len(global_df)),
        "stability_rows": int(len(stability_df)),
        "consistency_rows": int(len(consistency_df)),
        "escalation_recommendation": escalation,
    }

    out_json = run_dir / "consolidated_report.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    out_md = run_dir / "consolidated_report.md"
    lines: List[str] = []
    lines.append("# Consolidated Robust Matrix Report")
    lines.append("")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Generated: `{out_payload['generated_at']}`")
    lines.append(f"- Global candidate rows: `{len(global_df)}`")
    lines.append(f"- Stability rows: `{len(stability_df)}`")
    lines.append("")
    lines.append("## Escalation")
    lines.append("")
    if escalation:
        lines.append(f"- Triggered: `{escalation.get('triggered')}`")
        lines.append(f"- Reason: `{escalation.get('reason')}`")
        lines.append(f"- Recommendation: `{escalation.get('recommendation')}`")
    else:
        lines.append("- No `escalation_recommendation.json` found.")
    lines.append("")
    lines.append("## Top Global Candidates")
    lines.append("")
    if top_global:
        for row in top_global[:5]:
            lines.append(
                f"- Rank {row.get('final_rank')}: `{row.get('candidate_key')}` "
                f"(gate_pass={row.get('gate_pass')}, worst_attacked_benign_fpr={row.get('worst_attacked_benign_fpr')}, "
                f"worst_adv_malicious_recall={row.get('worst_adv_malicious_recall')})"
            )
    else:
        lines.append("- No global rows found.")
    lines.append("")
    lines.append("## Stable Candidates")
    lines.append("")
    if stable_candidates:
        for row in stable_candidates:
            ident = row.get("candidate_group_key", row.get("candidate_key"))
            lines.append(f"- `{ident}` (num_seeds_checked={row.get('num_seeds_checked')})")
    else:
        lines.append("- No stable candidates.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(run_dir))


if __name__ == "__main__":
    main()
