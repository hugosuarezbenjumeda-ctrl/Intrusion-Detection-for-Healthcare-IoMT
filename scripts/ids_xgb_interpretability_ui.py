                      

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from xgb_protocol_ids_utils import (
    choose_default_protocol,
    discover_latest_hpo_run,
    global_feature_importance,
    load_feature_columns,
    load_thresholds_by_protocol,
    load_xgb_models_by_protocol,
    local_feature_contributions,
    prepare_feature_matrix,
    protocol_slug,
    routed_predict,
)


def set_app_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        :root {
            --bg-a: #f5fbf7;
            --bg-b: #e8f2ff;
            --ink: #102a43;
            --muted: #486581;
            --accent: #d64545;
            --ok: #1a7f5a;
            --panel: rgba(255, 255, 255, 0.82);
            --border: rgba(16, 42, 67, 0.14);
        }
        .stApp {
            background:
              radial-gradient(80rem 40rem at -10% -10%, rgba(26,127,90,0.12), transparent 60%),
              radial-gradient(80rem 40rem at 110% 120%, rgba(214,69,69,0.14), transparent 60%),
              linear-gradient(120deg, var(--bg-a), var(--bg-b));
            color: var(--ink);
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(160deg, #f0f7f2, #f3f8ff);
        }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 8px 12px;
            backdrop-filter: blur(4px);
        }
        .ids-card {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: var(--panel);
            padding: 0.8rem 1rem;
            margin-bottom: 0.8rem;
        }
        .ids-chip {
            display: inline-block;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.8rem;
            background: rgba(16, 42, 67, 0.08);
            color: var(--ink);
            border-radius: 999px;
            padding: 0.15rem 0.6rem;
            margin-right: 0.4rem;
        }
        .ids-alert {
            color: var(--accent);
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_runtime(run_dir: str) -> Dict[str, object]:
    run_path = Path(run_dir)
    feature_columns = load_feature_columns(run_path)
    models = load_xgb_models_by_protocol(run_path)
    thresholds = load_thresholds_by_protocol(run_path)
    default_protocol = choose_default_protocol(models.keys())
    return {
        "run_dir": run_path,
        "feature_columns": feature_columns,
        "models": models,
        "thresholds": thresholds,
        "default_protocol": default_protocol,
    }


@st.cache_data(show_spinner=False)
def read_optional_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def score_flows(df: pd.DataFrame, runtime: Dict[str, object]) -> pd.DataFrame:
    pred = routed_predict(
        df=df.copy(),
        feature_columns=runtime["feature_columns"],                       
        models_by_protocol=runtime["models"],                       
        thresholds_by_protocol=runtime["thresholds"],                       
        default_protocol=str(runtime["default_protocol"]),
    )
    out = pd.concat([df.reset_index(drop=True), pred], axis=1)
    out["prediction_label"] = np.where(out["prediction"] == 1, "attack", "benign")
    return out


def explain_row(
    row: pd.Series,
    runtime: Dict[str, object],
    top_n: int,
) -> Tuple[str, pd.DataFrame]:
    model_map = runtime["models"]                       
    feature_columns = runtime["feature_columns"]                       
    default_proto = str(runtime["default_protocol"])

    row_proto = protocol_slug(str(row.get("protocol_hint", "")))
    proto = row_proto if row_proto in model_map else default_proto
    row_df = pd.DataFrame([row.to_dict()])
    x_row = prepare_feature_matrix(row_df, feature_columns)
    contrib = local_feature_contributions(
        booster=model_map[proto],
        row_features=x_row[0],
        feature_columns=feature_columns,
        top_n=top_n,
    )
    return proto, contrib


def render_prediction_header(score: float, threshold: float, pred_label: str, proto: str) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Attack Score", f"{score:.4f}")
    col2.metric("Threshold", f"{threshold:.4f}")
    col3.metric("Prediction", pred_label)
    col4.metric("Model Protocol", proto)


def render_local_contrib(contrib: pd.DataFrame) -> None:
    if contrib.empty:
        st.info("No local contributions available for this row.")
        return
    st.dataframe(contrib, use_container_width=True, hide_index=True)
    chart_df = contrib[["feature", "contribution"]].copy().set_index("feature")
    st.bar_chart(chart_df, height=340)


def get_default_run_dir() -> str:
    latest = discover_latest_hpo_run("reports")
    return str(latest) if latest else "reports/full_gpu_hpo_models_20260306_195851"


def read_csv_window(path: str, start_row: int, nrows: int) -> pd.DataFrame:
    if nrows <= 0:
        return pd.DataFrame()
    if start_row <= 0:
        return pd.read_csv(path, nrows=nrows)
    return pd.read_csv(path, skiprows=range(1, start_row + 1), nrows=nrows)


def init_sim_state(
    source_mode: str,
    upload_df: pd.DataFrame,
    local_path: str,
    chunk_rows: int,
    flows_per_sec: float,
) -> None:
    st.session_state["ids_sim_state"] = {
        "source_mode": source_mode,
        "upload_df": upload_df if source_mode == "Upload CSV" else pd.DataFrame(),
        "local_path": local_path,
        "chunk_rows": int(chunk_rows),
        "flows_per_sec": float(max(flows_per_sec, 1.0)),
        "cursor": 0,
        "tick": 0,
        "processed_rows": 0,
        "attack_rows": 0,
        "history": [],
        "recent_alerts": pd.DataFrame(),
        "ended": False,
    }


def get_next_sim_chunk(state: Dict[str, object]) -> pd.DataFrame:
    chunk_rows = int(state["chunk_rows"])
    cursor = int(state["cursor"])
    source_mode = str(state["source_mode"])
    if source_mode == "Upload CSV":
        upload_df = state["upload_df"]
        if not isinstance(upload_df, pd.DataFrame) or upload_df.empty:
            return pd.DataFrame()
        return upload_df.iloc[cursor : cursor + chunk_rows].copy()
    local_path = str(state["local_path"])
    if not local_path or not Path(local_path).exists():
        return pd.DataFrame()
    return read_csv_window(local_path, start_row=cursor, nrows=chunk_rows)


def run_sim_ticks(runtime: Dict[str, object], num_ticks: int) -> None:
    if "ids_sim_state" not in st.session_state:
        return
    state = st.session_state["ids_sim_state"]
    if not isinstance(state, dict):
        return

    for _ in range(max(1, int(num_ticks))):
        if bool(state.get("ended", False)):
            break
        chunk = get_next_sim_chunk(state)
        if chunk.empty:
            state["ended"] = True
            break

        scored = score_flows(chunk, runtime)
        rows = len(scored)
        attacks = int((scored["prediction"] == 1).sum())
        cursor = int(state["cursor"]) + rows
        tick = int(state["tick"]) + 1
        processed = int(state["processed_rows"]) + rows
        attack_total = int(state["attack_rows"]) + attacks
        flows_per_sec = float(state["flows_per_sec"])
        sim_elapsed = float(processed / max(flows_per_sec, 1.0))

        state["cursor"] = cursor
        state["tick"] = tick
        state["processed_rows"] = processed
        state["attack_rows"] = attack_total
        state["history"] = list(state["history"]) + [
            {
                "tick": tick,
                "rows": rows,
                "alerts": attacks,
                "alert_ratio": float(attacks / rows) if rows else 0.0,
                "sim_elapsed_sec": sim_elapsed,
            }
        ]

        alerts = scored[scored["prediction"] == 1].copy()
        if not alerts.empty:
            cols = [c for c in ["protocol_used", "score_attack", "threshold", "prediction_label", "attack_family", "source_relpath"] if c in alerts.columns]
            recent = alerts[cols].head(50)
            current_recent = state.get("recent_alerts", pd.DataFrame())
            if not isinstance(current_recent, pd.DataFrame):
                current_recent = pd.DataFrame()
            state["recent_alerts"] = pd.concat([recent, current_recent], ignore_index=True).head(200)

        if rows < int(state["chunk_rows"]):
            state["ended"] = True

    st.session_state["ids_sim_state"] = state


def main() -> None:
    st.set_page_config(page_title="XGBoost IDS Explainability Console", page_icon="IDS", layout="wide")
    set_app_style()

    st.title("Protocol-Routed XGBoost IDS Console")
    st.caption("Global and local explanations for the selected per-protocol XGBoost IDS models.")

    with st.sidebar:
        st.header("Runtime")
        run_dir = st.text_input("Run directory", value=get_default_run_dir())
        explain_dir_default = str(Path(run_dir) / "xgb_explainability")
        explain_dir = st.text_input("Explainability directory", value=explain_dir_default)
        top_n = st.slider("Top local features", min_value=5, max_value=30, value=15, step=1)
        if st.button("Reload models and artifacts"):
            st.cache_resource.clear()
            st.cache_data.clear()

    try:
        runtime = load_runtime(run_dir)
    except Exception as exc:
        st.error(f"Failed to load runtime from run dir: {exc}")
        st.stop()

    protocols = sorted((runtime["models"]).keys())                       
    thresholds = runtime["thresholds"]                       
    st.markdown(
        (
            "<div class='ids-card'>"
            + "".join([f"<span class='ids-chip'>{p}: thr={float(thresholds.get(p, 0.5)):.4f}</span>" for p in protocols])
            + "</div>"
        ),
        unsafe_allow_html=True,
    )

    explain_path = Path(explain_dir)
    global_df = read_optional_csv(str(explain_path / "global_feature_importance.csv"))
    local_summary_df = read_optional_csv(str(explain_path / "local_case_summary.csv"))
    local_contrib_df = read_optional_csv(str(explain_path / "local_case_contributions.csv"))
    local_ref_df = read_optional_csv(str(explain_path / "local_case_reference_rows.csv"))
    reference_rows_df = read_optional_csv(str(explain_path / "reference_rows.csv"))

    tab_single, tab_batch, tab_global, tab_cases = st.tabs(
        ["Single Flow", "Batch Scoring", "Global View", "Case Library"]
    )

    with tab_single:
        st.subheader("Single-flow prediction + local explanation")
        source = st.radio("Input source", ["Reference row", "Upload one-row CSV"], horizontal=True)

        row_df = pd.DataFrame()
        if source == "Reference row":
            if reference_rows_df.empty:
                st.warning("No reference_rows.csv found. Generate artifacts first or upload a one-row CSV.")
            else:
                ref_display = reference_rows_df.copy()
                if "protocol_used" in ref_display.columns:
                    ref_display["summary"] = (
                        ref_display["protocol_used"].astype(str)
                        + " | score="
                        + ref_display.get("score_attack", pd.Series([0] * len(ref_display))).astype(float).round(4).astype(str)
                    )
                else:
                    ref_display["summary"] = [f"row_{i}" for i in range(len(ref_display))]
                idx = st.selectbox("Choose reference row", options=list(range(len(ref_display))), format_func=lambda i: ref_display.loc[i, "summary"])
                row_df = reference_rows_df.iloc[[idx]].copy()
        else:
            upload = st.file_uploader("Upload CSV (first row used)", type=["csv"], key="single_upload")
            if upload is not None:
                row_df = pd.read_csv(upload).head(1)

        if not row_df.empty and st.button("Predict and explain", type="primary"):
            scored = score_flows(row_df, runtime)
            row = scored.iloc[0]
            proto, contrib = explain_row(row, runtime, top_n=top_n)
            render_prediction_header(
                score=float(row["score_attack"]),
                threshold=float(row["threshold"]),
                pred_label=str(row["prediction_label"]),
                proto=proto,
            )
            render_local_contrib(contrib)
            with st.expander("Scored row data"):
                st.dataframe(scored, use_container_width=True, hide_index=True)

    with tab_batch:
        st.subheader("Batch scoring")
        st.caption("Use local CSV path for large files. Uploads are still constrained by Streamlit/browser limits.")
        source_mode = st.radio("Batch source", ["Upload CSV", "Local CSV path"], horizontal=True, key="batch_source_mode")

        upload_df = pd.DataFrame()
        local_csv_path = ""
        if source_mode == "Upload CSV":
            batch_upload = st.file_uploader("Upload CSV for batch scoring", type=["csv"], key="batch_upload")
            if batch_upload is not None:
                upload_df = pd.read_csv(batch_upload)
                st.write(f"Rows loaded from upload: {len(upload_df):,}")
        else:
            local_csv_path = st.text_input("Local CSV path", value="data/merged/metadata_test.csv", key="batch_local_path")
            local_path_obj = Path(local_csv_path)
            if local_csv_path:
                if local_path_obj.exists():
                    st.success(f"Local file found: {local_path_obj}")
                else:
                    st.error("Local path does not exist.")

        st.markdown("### One-shot scoring")
        local_one_shot_nrows = st.number_input(
            "Rows to load for one-shot scoring (Local path mode only)",
            min_value=1_000,
            max_value=5_000_000,
            value=100_000,
            step=1_000,
            key="batch_local_nrows",
        )
        if st.button("Run one-shot batch scoring", type="primary"):
            batch_df = pd.DataFrame()
            if source_mode == "Upload CSV":
                batch_df = upload_df.copy()
            else:
                path_obj = Path(local_csv_path)
                if path_obj.exists():
                    batch_df = pd.read_csv(path_obj, nrows=int(local_one_shot_nrows))
            if batch_df.empty:
                st.warning("No batch data available. Upload a CSV or provide a valid local path.")
            else:
                scored = score_flows(batch_df, runtime)
                st.dataframe(scored.head(200), use_container_width=True, hide_index=True)
                attack_rate = float((scored["prediction"] == 1).mean()) if len(scored) else 0.0
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", f"{len(scored):,}")
                c2.metric("Predicted attacks", f"{int((scored['prediction'] == 1).sum()):,}")
                c3.metric("Attack ratio", f"{attack_rate:.4f}")

                csv_text = scored.to_csv(index=False)
                st.download_button(
                    "Download scored CSV",
                    data=csv_text.encode("utf-8"),
                    file_name="ids_scored_batch.csv",
                    mime="text/csv",
                )

                explain_idx = st.number_input(
                    "Inspect row for local explanation",
                    min_value=0,
                    max_value=max(0, len(scored) - 1),
                    value=0,
                    step=1,
                    key="batch_explain_idx",
                )
                if len(scored) > 0:
                    row = scored.iloc[int(explain_idx)]
                    proto, contrib = explain_row(row, runtime, top_n=top_n)
                    render_prediction_header(
                        score=float(row["score_attack"]),
                        threshold=float(row["threshold"]),
                        pred_label=str(row["prediction_label"]),
                        proto=proto,
                    )
                    render_local_contrib(contrib)

        st.markdown("### Sequential flow simulation")
        st.caption("Replay flows in time-slices to simulate streaming IDS behavior.")
        sim_chunk_rows = st.number_input("Rows per tick", min_value=100, max_value=200_000, value=5_000, step=100)
        sim_flows_per_sec = st.number_input("Simulated flows per second", min_value=1.0, max_value=500_000.0, value=2_500.0, step=100.0)
        c_init, c_tick, c_burst, c_reset = st.columns(4)

        if c_init.button("Initialize simulation"):
            if source_mode == "Upload CSV" and upload_df.empty:
                st.warning("Upload CSV first.")
            elif source_mode == "Local CSV path" and (not local_csv_path or not Path(local_csv_path).exists()):
                st.warning("Provide a valid local CSV path first.")
            else:
                init_sim_state(
                    source_mode=source_mode,
                    upload_df=upload_df,
                    local_path=local_csv_path,
                    chunk_rows=int(sim_chunk_rows),
                    flows_per_sec=float(sim_flows_per_sec),
                )
                st.success("Simulation initialized.")

        if c_tick.button("Run next tick"):
            run_sim_ticks(runtime, num_ticks=1)
        if c_burst.button("Run 10 ticks"):
            run_sim_ticks(runtime, num_ticks=10)
        if c_reset.button("Reset"):
            st.session_state.pop("ids_sim_state", None)

        sim_state = st.session_state.get("ids_sim_state")
        if isinstance(sim_state, dict):
            processed_rows = int(sim_state.get("processed_rows", 0))
            attack_rows = int(sim_state.get("attack_rows", 0))
            flows_per_sec = float(sim_state.get("flows_per_sec", 1.0))
            sim_elapsed = processed_rows / max(flows_per_sec, 1.0)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Ticks", f"{int(sim_state.get('tick', 0))}")
            s2.metric("Processed flows", f"{processed_rows:,}")
            s3.metric("Alerts", f"{attack_rows:,}")
            s4.metric("Simulated elapsed", f"{sim_elapsed:.1f}s")

            history = sim_state.get("history", [])
            if isinstance(history, list) and history:
                hist_df = pd.DataFrame(history)
                st.dataframe(hist_df.tail(30), use_container_width=True, hide_index=True)
                st.line_chart(hist_df.set_index("tick")[["alerts", "alert_ratio"]], height=260)

            recent_alerts = sim_state.get("recent_alerts")
            if isinstance(recent_alerts, pd.DataFrame) and not recent_alerts.empty:
                st.markdown("**Recent alerts**")
                st.dataframe(recent_alerts.head(200), use_container_width=True, hide_index=True)

            if bool(sim_state.get("ended", False)):
                st.info("Simulation reached end of input.")

    with tab_global:
        st.subheader("Global explanations")
        if global_df.empty:
            st.info("global_feature_importance.csv not found. Showing live gain/weight importance from model files.")
            all_global = []
            for proto in protocols:
                imp = global_feature_importance(runtime["models"][proto], runtime["feature_columns"])                       
                imp["protocol_hint"] = proto
                all_global.append(imp)
            global_df = pd.concat(all_global, ignore_index=True)

        selected_proto = st.selectbox("Protocol", options=protocols, key="global_proto")
        available_metrics = [c for c in ["mean_abs_contribution", "gain", "weight", "cover"] if c in global_df.columns]
        metric = st.selectbox("Ranking metric", options=available_metrics, index=0)
        top_k = st.slider("Top features", min_value=5, max_value=40, value=20, step=1)

        proto_df = global_df[global_df["protocol_hint"].astype(str) == selected_proto].copy()
        if proto_df.empty:
            st.warning("No global rows found for this protocol.")
        else:
            proto_df = proto_df.sort_values(metric, ascending=False).head(top_k)
            st.dataframe(proto_df, use_container_width=True, hide_index=True)
            chart = proto_df[["feature", metric]].set_index("feature")
            st.bar_chart(chart, height=420)

    with tab_cases:
        st.subheader("Local case library")
        if local_summary_df.empty or local_contrib_df.empty:
            st.warning("No local case artifacts found. Run explainability artifact generation first.")
        else:
            filt_proto = st.selectbox("Protocol filter", options=["all"] + protocols, index=0)
            outcome_options = ["all"] + sorted(local_summary_df["outcome"].astype(str).unique().tolist())
            filt_outcome = st.selectbox("Outcome filter", options=outcome_options, index=0)

            view = local_summary_df.copy()
            if filt_proto != "all":
                view = view[view["protocol_used"].astype(str) == filt_proto]
            if filt_outcome != "all":
                view = view[view["outcome"].astype(str) == filt_outcome]

            if view.empty:
                st.info("No cases match current filters.")
            else:
                st.dataframe(view, use_container_width=True, hide_index=True)
                case_ids = view["case_id"].astype(str).tolist()
                selected_case = st.selectbox("Select case", options=case_ids)

                case_row = local_summary_df[local_summary_df["case_id"].astype(str) == selected_case].iloc[0]
                render_prediction_header(
                    score=float(case_row["score_attack"]),
                    threshold=float(case_row["threshold"]),
                    pred_label="attack" if int(case_row["prediction"]) == 1 else "benign",
                    proto=str(case_row["protocol_used"]),
                )

                cdf = local_contrib_df[local_contrib_df["case_id"].astype(str) == selected_case].copy()
                cdf = cdf.sort_values("abs_contribution", ascending=False)
                render_local_contrib(cdf)

                if not local_ref_df.empty and "case_id" in local_ref_df.columns:
                    raw_case = local_ref_df[local_ref_df["case_id"].astype(str) == selected_case]
                    if not raw_case.empty:
                        with st.expander("Reference row"):
                            st.dataframe(raw_case, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
