                      
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    xgb = None
    HAS_XGBOOST = False

import train_protocol_multimodel_robust_matrix as matrix


class ProtocolMatrixTests(unittest.TestCase):
    @unittest.skipUnless(HAS_XGBOOST, "xgboost is required for parity test")
    def test_xgb_attack_scorer_parity(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(64, 6)).astype(np.float32)
        y = (x[:, 0] + 0.2 * x[:, 1] > 0.0).astype(np.float32)
        dtrain = xgb.DMatrix(x, label=y)
        booster = xgb.train(
            params={
                "objective": "binary:logistic",
                "tree_method": "hist",
                "device": "cpu",
                "max_depth": 3,
                "eta": 0.2,
                "seed": 7,
            },
            dtrain=dtrain,
            num_boost_round=12,
            verbose_eval=False,
        )
        threshold = 0.42
        old_margin = matrix.score_margin_batch(booster, x, threshold=threshold)
        scorer = matrix._make_xgb_attack_scorer(booster=booster, threshold=threshold, source_detail="test")
        new_margin = scorer.score_margin_fn(x)
        self.assertTrue(np.allclose(old_margin, new_margin, atol=1e-10, rtol=0.0))

    def test_threshold_selector_gate_pass_mode(self) -> None:
        y_clean = np.array([0, 0, 1, 1], dtype=np.int8)
        p_clean = np.array([0.05, 0.20, 0.85, 0.95], dtype=np.float64)
        p_adv_benign = np.array([0.01, 0.02, 0.40, 0.55], dtype=np.float64)
        p_adv_malicious = np.array([0.90, 0.92, 0.95, 0.97], dtype=np.float64)
        out = matrix._select_threshold_gate_lexicographic(
            y_clean=y_clean,
            p_clean=p_clean,
            p_adv_benign=p_adv_benign,
            p_adv_malicious=p_adv_malicious,
            grid_size=50,
            base_threshold=0.5,
            gate_clean_fpr_max=0.25,
            gate_attacked_benign_fpr_max=0.25,
            gate_adv_malicious_recall_min=0.75,
        )
        self.assertEqual(out["selection_mode"], "internal_margin_gate_pass_lexicographic")
        self.assertGreater(out["num_internal_gate_pass"], 0)
        self.assertGreater(out["num_gate_pass"], 0)
        self.assertTrue(bool(out["selected_metrics"]["gate_pass"]))

    def test_threshold_selector_margin_fallback_to_standard_gate_pass(self) -> None:
        y_clean = np.array([0, 0, 1, 1], dtype=np.int8)
        p_clean = np.array([0.10, 0.20, 0.80, 0.90], dtype=np.float64)
        p_adv_benign = np.array([0.05, 0.15, 0.25, 0.35], dtype=np.float64)
        p_adv_malicious = np.array([0.31, 0.32, 0.33, 0.34], dtype=np.float64)
        out = matrix._select_threshold_gate_lexicographic(
            y_clean=y_clean,
            p_clean=p_clean,
            p_adv_benign=p_adv_benign,
            p_adv_malicious=p_adv_malicious,
            grid_size=60,
            base_threshold=0.3,
            gate_clean_fpr_max=0.5,
            gate_attacked_benign_fpr_max=0.25,
            gate_adv_malicious_recall_min=0.75,
            gate_attacked_benign_margin=0.8,
        )
        self.assertEqual(out["selection_mode"], "gate_pass_lexicographic")
        self.assertEqual(int(out["num_internal_gate_pass"]), 0)
        self.assertGreater(int(out["num_gate_pass"]), 0)

    def test_threshold_selector_no_pass_adv_priority(self) -> None:
        y_clean = np.array([0, 0, 1, 1], dtype=np.int8)
        p_clean = np.array([0.05, 0.95, 0.90, 0.92], dtype=np.float64)
        p_adv_benign = np.array([0.30, 0.35, 0.40, 0.45], dtype=np.float64)
        p_adv_malicious = np.array([0.18, 0.22, 0.25, 0.27], dtype=np.float64)
        out = matrix._select_threshold_gate_lexicographic(
            y_clean=y_clean,
            p_clean=p_clean,
            p_adv_benign=p_adv_benign,
            p_adv_malicious=p_adv_malicious,
            grid_size=80,
            base_threshold=0.8,
            gate_clean_fpr_max=0.005,
            gate_attacked_benign_fpr_max=0.005,
            gate_adv_malicious_recall_min=0.99,
        )
        self.assertEqual(out["selection_mode"], "no_pass_adv_priority")
        self.assertEqual(out["num_gate_pass"], 0)
        self.assertLessEqual(float(out["selected_adv_shortfall"]), 0.8)
                                                                                   
        self.assertLess(float(out["selected_threshold"]), 0.8)

    def test_bluetooth_hardneg_cap_enforced(self) -> None:
        rng = np.random.default_rng(123)
        x_hardneg = rng.normal(size=(100, 8)).astype(np.float32)
        capped_a, info_a = matrix._cap_bluetooth_hardneg_rows(
            protocol="bluetooth",
            x_hardneg=x_hardneg,
            sampled_benign_count=120,
            max_fraction=0.5,
            seed=42,
        )
        capped_b, info_b = matrix._cap_bluetooth_hardneg_rows(
            protocol="bluetooth",
            x_hardneg=x_hardneg,
            sampled_benign_count=120,
            max_fraction=0.5,
            seed=42,
        )
        self.assertEqual(capped_a.shape[0], 60)
        self.assertTrue(np.array_equal(capped_a, capped_b))
        self.assertTrue(bool(info_a["capped"]))
        self.assertEqual(info_a["after_rows"], 60)
        self.assertEqual(info_b["after_rows"], 60)

    def test_bluetooth_recovery_fallback_family_pack(self) -> None:
        args = SimpleNamespace(
            family_pack="bluetooth_recovery_fallback",
            family_a_hardneg_weight=1.5,
            family_a_maladv_weight=1.5,
            family_b_hardneg_weight=2.0,
            family_b_maladv_weight=2.0,
            family_b_extra_topk_per_epsilon=500,
            family_c_hardneg_weight=2.5,
            family_c_maladv_weight=2.5,
            family_c_extra_topk_per_epsilon=2500,
        )
        families = matrix._family_matrix_configs(args)
        self.assertEqual(len(families), 5)
        fam_c = next(cfg for cfg in families if cfg["family_id"] == "C")
        self.assertEqual(fam_c["family_description"], "bluetooth_recovery_fallback_tuned")
        self.assertAlmostEqual(float(fam_c["hardneg_weight"]), 1.9, places=6)
        self.assertAlmostEqual(float(fam_c["maladv_weight"]), 2.1, places=6)
        self.assertEqual(int(fam_c["extra_topk_per_epsilon"]), 500)

    def test_shared_val_seed_split_is_identical_across_stability_seeds(self) -> None:
        df = pd.DataFrame(
            {
                "source_relpath": [
                    "wifi_a.csv",
                    "wifi_a.csv",
                    "wifi_b.csv",
                    "wifi_b.csv",
                    "bluetooth_a.csv",
                    "bluetooth_a.csv",
                    "bluetooth_b.csv",
                    "bluetooth_b.csv",
                ],
                "source_row_index": [0, 1, 0, 1, 0, 1, 0, 1],
                "protocol_hint": ["wifi", "wifi", "wifi", "wifi", "bluetooth", "bluetooth", "bluetooth", "bluetooth"],
                "label": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        split_seed_43 = matrix._resolve_stability_split_seed(
            split_mode="shared_val_seed",
            shared_split_seed=42,
            stability_seed=43,
        )
        split_seed_44 = matrix._resolve_stability_split_seed(
            split_mode="shared_val_seed",
            shared_split_seed=42,
            stability_seed=44,
        )
        self.assertEqual(split_seed_43, 42)
        self.assertEqual(split_seed_44, 42)
        val_mask_43, _ = matrix._build_protocol_class_val_mask(df, seed=split_seed_43, val_mod=2, allow_row_fallback=True)
        val_mask_44, _ = matrix._build_protocol_class_val_mask(df, seed=split_seed_44, val_mod=2, allow_row_fallback=True)
        self.assertTrue(np.array_equal(val_mask_43, val_mask_44))

    def test_bluetooth_benign_train_floor_is_deterministic(self) -> None:
        n_dom = 18
        n_other = 4
        df = pd.DataFrame(
            {
                "source_relpath": (["dom.csv"] * n_dom) + (["other.csv"] * n_other) + (["mal.csv"] * 6),
                "source_row_index": list(range(n_dom + n_other + 6)),
                "protocol_hint": (["bluetooth"] * (n_dom + n_other + 6)),
                "label": ([0] * (n_dom + n_other)) + ([1] * 6),
            }
        )
        base_mask = np.zeros(len(df), dtype=bool)
                                                                                 
        base_mask[:16] = True

        adjusted_a, ev_a = matrix._apply_bluetooth_benign_train_floor(
            df=df,
            val_mask=base_mask,
            min_train_rows=10,
            seed=42,
        )
        adjusted_b, ev_b = matrix._apply_bluetooth_benign_train_floor(
            df=df,
            val_mask=base_mask,
            min_train_rows=10,
            seed=42,
        )

        self.assertTrue(np.array_equal(adjusted_a, adjusted_b))
        self.assertIsNotNone(ev_a)
        self.assertIsNotNone(ev_b)
        self.assertTrue(bool(ev_a["floor_applied"]))
        self.assertEqual(int(ev_a["train_rows_after_floor"]), 10)
        moved_idx = np.flatnonzero(base_mask & (~adjusted_a))
        moved_paths = set(df.loc[moved_idx, "source_relpath"].astype(str).tolist())
        self.assertEqual(moved_paths, {"dom.csv"})

    def test_stability_sampling_fixed_mode_indices_identical(self) -> None:
        proto_data = {
            "train_benign_all_idx": np.arange(0, 50, dtype=np.int64),
            "train_malicious_all_idx": np.arange(50, 100, dtype=np.int64),
            "val_benign_all_idx": np.arange(0, 40, dtype=np.int64),
            "val_malicious_all_idx": np.arange(40, 80, dtype=np.int64),
        }
        fixed_seed = 42
        seed_43 = matrix._resolve_stability_sampling_seed(
            sampling_mode="fixed",
            shared_sampling_seed=fixed_seed,
            stability_seed=43,
            protocol="bluetooth",
        )
        seed_44 = matrix._resolve_stability_sampling_seed(
            sampling_mode="fixed",
            shared_sampling_seed=fixed_seed,
            stability_seed=44,
            protocol="wifi",
        )
        a = matrix._sample_protocol_indices_for_run(
            proto="bluetooth",
            proto_data=proto_data,
            run_seed=43,
            hardneg_train_benign_sample=20,
            hardneg_train_malicious_sample=20,
            hardneg_val_benign_sample=15,
            val_malicious_sample=15,
            sampling_seed=seed_43,
        )
        b = matrix._sample_protocol_indices_for_run(
            proto="bluetooth",
            proto_data=proto_data,
            run_seed=44,
            hardneg_train_benign_sample=20,
            hardneg_train_malicious_sample=20,
            hardneg_val_benign_sample=15,
            val_malicious_sample=15,
            sampling_seed=seed_44,
        )
        self.assertTrue(np.array_equal(a["train_benign_idx"], b["train_benign_idx"]))
        self.assertTrue(np.array_equal(a["train_malicious_idx"], b["train_malicious_idx"]))
        self.assertTrue(np.array_equal(a["val_benign_idx"], b["val_benign_idx"]))
        self.assertTrue(np.array_equal(a["val_malicious_idx"], b["val_malicious_idx"]))

    def test_adaptive_sampler_is_deterministic_and_clamped(self) -> None:
        proto_data = {
            "train_benign_all_idx": np.arange(0, 40, dtype=np.int64),
            "train_malicious_all_idx": np.arange(40, 80, dtype=np.int64),
            "val_benign_all_idx": np.arange(0, 20, dtype=np.int64),
            "val_malicious_all_idx": np.arange(20, 30, dtype=np.int64),
        }
        sampling_buckets = {
            "train_benign": {"fraction": 0.10, "min_rows": 6, "max_rows": 9},
            "train_malicious": {"fraction": 0.05, "min_rows": 3, "max_rows": 4},
            "val_benign": {"fraction": 0.30, "min_rows": 8, "max_rows": 10},
            "val_malicious": {"fraction": 0.10, "min_rows": 20, "max_rows": 50},
        }
        a = matrix._sample_protocol_indices_for_run(
            proto="wifi",
            proto_data=proto_data,
            run_seed=43,
            hardneg_train_benign_sample=999,
            hardneg_train_malicious_sample=999,
            hardneg_val_benign_sample=999,
            val_malicious_sample=999,
            sampling_seed=42,
            sampling_policy="adaptive_fraction",
            sampling_buckets=sampling_buckets,
        )
        b = matrix._sample_protocol_indices_for_run(
            proto="wifi",
            proto_data=proto_data,
            run_seed=44,
            hardneg_train_benign_sample=1,
            hardneg_train_malicious_sample=1,
            hardneg_val_benign_sample=1,
            val_malicious_sample=1,
            sampling_seed=42,
            sampling_policy="adaptive_fraction",
            sampling_buckets=sampling_buckets,
        )
        self.assertTrue(np.array_equal(a["train_benign_idx"], b["train_benign_idx"]))
        self.assertTrue(np.array_equal(a["train_malicious_idx"], b["train_malicious_idx"]))
        self.assertTrue(np.array_equal(a["val_benign_idx"], b["val_benign_idx"]))
        self.assertTrue(np.array_equal(a["val_malicious_idx"], b["val_malicious_idx"]))
        self.assertEqual(int(a["train_benign_idx"].shape[0]), 6)
        self.assertEqual(int(a["train_malicious_idx"].shape[0]), 3)
        self.assertEqual(int(a["val_benign_idx"].shape[0]), 8)
                                                        
        self.assertEqual(int(a["val_malicious_idx"].shape[0]), 10)

    def test_adaptive_sample_target_pool_smaller_than_min_uses_full_pool(self) -> None:
        out = matrix._resolve_adaptive_sample_target(
            pool_size=5,
            fraction=0.10,
            min_rows=10,
            max_rows=100,
        )
        self.assertEqual(int(out), 5)

    def test_stability_sampling_per_seed_and_hybrid_resolution(self) -> None:
        seed_bt = matrix._resolve_stability_sampling_seed(
            sampling_mode="hybrid",
            shared_sampling_seed=42,
            stability_seed=47,
            protocol="bluetooth",
        )
        seed_wifi = matrix._resolve_stability_sampling_seed(
            sampling_mode="hybrid",
            shared_sampling_seed=42,
            stability_seed=47,
            protocol="wifi",
        )
        seed_per = matrix._resolve_stability_sampling_seed(
            sampling_mode="per_seed",
            shared_sampling_seed=42,
            stability_seed=47,
            protocol="bluetooth",
        )
        self.assertEqual(seed_bt, 42)
        self.assertEqual(seed_wifi, 47)
        self.assertEqual(seed_per, 47)

    def test_data_cap_summary_contains_bluetooth_floor_fields(self) -> None:
        df = pd.DataFrame(
            {
                "source_relpath": ["bt_a.csv", "bt_a.csv", "bt_b.csv", "wifi.csv"],
                "source_row_index": [0, 1, 0, 0],
                "protocol_hint": ["bluetooth", "bluetooth", "bluetooth", "wifi"],
                "label": [0, 0, 1, 0],
            }
        )
        val_mask = np.array([True, False, True, True], dtype=bool)
        rows = matrix._build_data_cap_summary_rows(
            df=df,
            val_mask=val_mask,
            protocols=["bluetooth", "wifi"],
            fallback_events=[
                {
                    "protocol_hint": "bluetooth",
                    "label": 0,
                    "fallback_type": "bluetooth_benign_min_train_floor",
                    "floor_target": 8000,
                    "floor_applied": True,
                    "rows_moved_to_train": 120,
                    "train_rows_before_floor": 2,
                    "train_rows_after_floor": 122,
                    "val_rows_before_floor": 500,
                    "val_rows_after_floor": 380,
                }
            ],
            split_seed_used=42,
        )
        bt_row = next(r for r in rows if str(r["protocol"]) == "bluetooth" and int(r["label"]) == 0)
        self.assertEqual(int(bt_row["floor_target"]), 8000)
        self.assertTrue(bool(bt_row["floor_applied"]))
        self.assertEqual(int(bt_row["rows_moved_to_train"]), 120)

    def test_fpr_feasibility_checker(self) -> None:
        infeasible = matrix._compute_fpr_feasibility(
            val_benign_count=199,
            min_required=200,
            enabled=True,
        )
        feasible = matrix._compute_fpr_feasibility(
            val_benign_count=200,
            min_required=200,
            enabled=True,
        )
        self.assertFalse(bool(infeasible["strict_fpr_feasible"]))
        self.assertTrue(bool(feasible["strict_fpr_feasible"]))
        self.assertAlmostEqual(float(feasible["fpr_resolution"]), 1.0 / 200.0, places=12)

    def test_gate_override_unchanged_when_feasible(self) -> None:
        selected = {
            "gate_pass": True,
            "gate_failure_category": "none",
            "gate_clean_fpr": True,
            "gate_attacked_benign_fpr": True,
            "gate_adv_malicious_recall": True,
        }
        unchanged = matrix._apply_data_cap_gate_override(selected, strict_fpr_feasible=True)
        forced_fail = matrix._apply_data_cap_gate_override(selected, strict_fpr_feasible=False)
        self.assertTrue(bool(unchanged["gate_pass"]))
        self.assertEqual(str(unchanged["gate_failure_category"]), "none")
        self.assertFalse(bool(forced_fail["gate_pass"]))
        self.assertEqual(str(forced_fail["gate_failure_category"]), "data_cap_infeasible")

    def test_stage_profile_override_precedence(self) -> None:
        args = SimpleNamespace(
            sampling_policy="adaptive_fraction",
            hardneg_train_benign_sample=12000,
            hardneg_train_malicious_sample=12000,
            hardneg_val_benign_sample=4000,
            val_malicious_sample=4000,
            hardneg_train_benign_fraction=0.20,
            hardneg_train_benign_min=4000,
            hardneg_train_benign_max=30000,
            hardneg_train_malicious_fraction=0.08,
            hardneg_train_malicious_min=4000,
            hardneg_train_malicious_max=20000,
            hardneg_val_benign_fraction=0.20,
            hardneg_val_benign_min=3000,
            hardneg_val_benign_max=10000,
            val_malicious_fraction=0.08,
            val_malicious_min=3000,
            val_malicious_max=8000,
            hardneg_query_budget=120,
            hardneg_query_max_steps=60,
            hardneg_candidates_per_step=3,
            hardneg_feature_subset_size=3,
            query_score_batch_rows=64,
            query_fast_projection=True,
            query_refine_topk=2,
            hardneg_query_max_active_rows_per_step=0,
            hardneg_query_stagnation_patience=0,
            hardneg_query_stagnation_min_delta=1e-6,
            val_malicious_query_budget=None,
            val_malicious_query_max_steps=None,
            val_malicious_candidates_per_step=None,
            val_malicious_feature_subset_size=None,
            val_malicious_score_batch_rows=None,
            val_malicious_query_max_active_rows_per_step=None,
            val_malicious_query_stagnation_patience=None,
            val_malicious_query_stagnation_min_delta=None,
            coarse_hardneg_train_benign_sample=2500,
            coarse_hardneg_train_malicious_sample=2500,
            coarse_hardneg_val_benign_sample=1000,
            coarse_val_malicious_sample=1000,
            coarse_hardneg_train_benign_fraction=0.15,
            coarse_hardneg_train_benign_min=2000,
            coarse_hardneg_train_benign_max=9000,
            coarse_hardneg_train_malicious_fraction=0.05,
            coarse_hardneg_train_malicious_min=1500,
            coarse_hardneg_train_malicious_max=8000,
            coarse_hardneg_val_benign_fraction=0.18,
            coarse_hardneg_val_benign_min=900,
            coarse_hardneg_val_benign_max=3000,
            coarse_val_malicious_fraction=0.06,
            coarse_val_malicious_min=700,
            coarse_val_malicious_max=2500,
            coarse_hardneg_query_budget=24,
            coarse_hardneg_query_max_steps=8,
            coarse_hardneg_candidates_per_step=16,
            coarse_query_score_batch_rows=256,
            coarse_query_refine_topk=3,
            coarse_val_malicious_query_budget=24,
            coarse_val_malicious_query_max_steps=8,
            coarse_val_malicious_candidates_per_step=16,
            coarse_val_malicious_score_batch_rows=256,
            stability_hardneg_train_benign_sample=None,
            stability_hardneg_train_malicious_sample=None,
            stability_hardneg_val_benign_sample=None,
            stability_val_malicious_sample=None,
            stability_hardneg_train_benign_fraction=None,
            stability_hardneg_train_benign_min=None,
            stability_hardneg_train_benign_max=None,
            stability_hardneg_train_malicious_fraction=None,
            stability_hardneg_train_malicious_min=None,
            stability_hardneg_train_malicious_max=None,
            stability_hardneg_val_benign_fraction=None,
            stability_hardneg_val_benign_min=None,
            stability_hardneg_val_benign_max=None,
            stability_val_malicious_fraction=None,
            stability_val_malicious_min=None,
            stability_val_malicious_max=None,
            stability_hardneg_query_budget=None,
            stability_hardneg_query_max_steps=None,
            stability_hardneg_candidates_per_step=None,
            stability_query_score_batch_rows=None,
            stability_query_refine_topk=None,
            stability_val_malicious_query_budget=None,
            stability_val_malicious_query_max_steps=None,
            stability_val_malicious_candidates_per_step=None,
            stability_val_malicious_score_batch_rows=None,
            models="xgboost,catboost",
            coarse_models="xgboost,catboost",
            stability_models="",
        )
        coarse = matrix._resolve_stage_search_profile(args, "coarse")
        stability = matrix._resolve_stage_search_profile(args, "stability")

        self.assertEqual(str(coarse["sampling_policy"]), "adaptive_fraction")
        self.assertEqual(int(coarse["hardneg_train_benign_sample"]), 2500)
        self.assertEqual(int(coarse["hardneg_val_benign_sample"]), 1000)
        self.assertEqual(int(coarse["query_budget_benign"]), 24)
        self.assertEqual(int(coarse["query_max_steps_benign"]), 8)
        self.assertEqual(int(coarse["query_candidates_benign"]), 16)
        self.assertEqual(int(coarse["query_score_batch_benign"]), 256)
        self.assertTrue(bool(coarse["query_fast_projection"]))
        self.assertEqual(int(coarse["query_refine_topk"]), 3)
        self.assertAlmostEqual(float(coarse["sampling_buckets"]["train_benign"]["fraction"]), 0.15, places=9)
        self.assertEqual(int(coarse["sampling_buckets"]["train_benign"]["min_rows"]), 2000)
        self.assertEqual(int(coarse["sampling_buckets"]["train_benign"]["max_rows"]), 9000)
        self.assertEqual(int(stability["hardneg_train_benign_sample"]), 12000)
        self.assertEqual(int(stability["query_budget_benign"]), 120)
        self.assertEqual(int(stability["query_score_batch_benign"]), 64)
        self.assertEqual(int(stability["query_refine_topk"]), 2)
        self.assertAlmostEqual(float(stability["sampling_buckets"]["train_benign"]["fraction"]), 0.20, places=9)
        self.assertEqual(int(stability["sampling_buckets"]["train_benign"]["min_rows"]), 4000)
        self.assertEqual(int(stability["sampling_buckets"]["train_benign"]["max_rows"]), 30000)

    def test_stability_models_default_to_shortlist(self) -> None:
        args = SimpleNamespace(
            models="xgboost,catboost",
            coarse_models="xgboost,catboost",
            stability_models="",
        )
        coarse_models = matrix._resolve_stage_models(args, stage_name="coarse")
        self.assertEqual(coarse_models, ["xgboost", "catboost"])

        shortlist = ["catboost", "xgboost", "catboost"]
        stability_models = matrix._resolve_stage_models(
            args,
            stage_name="stability",
            shortlist_models=shortlist,
        )
        self.assertEqual(stability_models, ["catboost", "xgboost"])

        args_explicit = SimpleNamespace(
            models="xgboost,catboost",
            coarse_models="xgboost,catboost",
            stability_models="xgboost",
        )
        explicit_models = matrix._resolve_stage_models(
            args_explicit,
            stage_name="stability",
            shortlist_models=shortlist,
        )
        self.assertEqual(explicit_models, ["xgboost"])

    def test_model_restriction_rejects_unsupported_models(self) -> None:
        args_bad = SimpleNamespace(
            models="xgboost,mlp",
            coarse_models="",
            stability_models="",
        )
        with self.assertRaises(RuntimeError):
            matrix._resolve_stage_models(args_bad, stage_name="coarse")
        args_bad_stability = SimpleNamespace(
            models="xgboost,catboost",
            coarse_models="xgboost,catboost",
            stability_models="mlp",
        )
        with self.assertRaises(RuntimeError):
            matrix._resolve_stage_models(
                args_bad_stability,
                stage_name="stability",
                shortlist_models=["xgboost", "catboost"],
            )

    def test_shortlist_diversity_includes_each_enabled_model_before_fill(self) -> None:
        family_cfgs = [
            {"family_id": "A", "family_name": "fpr_priority"},
            {"family_id": "B", "family_name": "balanced"},
        ]
        global_rows = [
            {
                "candidate_group_key": "xgboost__A",
                "model_name": "xgboost",
                "family_id": "A",
            },
            {
                "candidate_group_key": "xgboost__B",
                "model_name": "xgboost",
                "family_id": "B",
            },
            {
                "candidate_group_key": "catboost__A",
                "model_name": "catboost",
                "family_id": "A",
            },
        ]
        shortlist_specs = matrix._build_stability_shortlist_specs(
            global_rows=global_rows,
            family_cfgs=family_cfgs,
            stage2_topk_global=1,
            enabled_models=["xgboost", "catboost"],
        )
        shortlist_models = [str(spec.model_name) for spec in shortlist_specs]
        self.assertGreaterEqual(len(shortlist_specs), 2)
        self.assertIn("xgboost", shortlist_models)
        self.assertIn("catboost", shortlist_models)

    def test_topk_two_finalists_keep_ranking_order(self) -> None:
        protocol_rows = [
            {
                "protocol": "wifi",
                "model_name": "xgboost",
                "family_id": "A",
                "seed": 42,
                "gate_pass": True,
                "gate_failure_category": "none",
                "attacked_benign_fpr": 0.001,
                "clean_fpr": 0.001,
                "adv_malicious_recall": 0.995,
                "clean_f1": 0.91,
                "robust_f1": 0.88,
            },
            {
                "protocol": "mqtt",
                "model_name": "xgboost",
                "family_id": "A",
                "seed": 42,
                "gate_pass": True,
                "gate_failure_category": "none",
                "attacked_benign_fpr": 0.002,
                "clean_fpr": 0.001,
                "adv_malicious_recall": 0.994,
                "clean_f1": 0.90,
                "robust_f1": 0.87,
            },
            {
                "protocol": "wifi",
                "model_name": "catboost",
                "family_id": "B",
                "seed": 42,
                "gate_pass": True,
                "gate_failure_category": "none",
                "attacked_benign_fpr": 0.003,
                "clean_fpr": 0.002,
                "adv_malicious_recall": 0.993,
                "clean_f1": 0.89,
                "robust_f1": 0.86,
            },
            {
                "protocol": "mqtt",
                "model_name": "catboost",
                "family_id": "B",
                "seed": 42,
                "gate_pass": True,
                "gate_failure_category": "none",
                "attacked_benign_fpr": 0.003,
                "clean_fpr": 0.002,
                "adv_malicious_recall": 0.993,
                "clean_f1": 0.89,
                "robust_f1": 0.86,
            },
            {
                "protocol": "wifi",
                "model_name": "mlp",
                "family_id": "C",
                "seed": 42,
                "gate_pass": False,
                "gate_failure_category": "attacked_benign_fpr",
                "attacked_benign_fpr": 0.010,
                "clean_fpr": 0.004,
                "adv_malicious_recall": 0.990,
                "clean_f1": 0.85,
                "robust_f1": 0.80,
            },
            {
                "protocol": "mqtt",
                "model_name": "mlp",
                "family_id": "C",
                "seed": 42,
                "gate_pass": False,
                "gate_failure_category": "attacked_benign_fpr",
                "attacked_benign_fpr": 0.011,
                "clean_fpr": 0.004,
                "adv_malicious_recall": 0.989,
                "clean_f1": 0.84,
                "robust_f1": 0.79,
            },
        ]
        ranked = matrix._build_global_rows(protocol_rows=protocol_rows, protocols=["wifi", "mqtt"])
        shortlist = ranked[:2]
        self.assertEqual(len(shortlist), 2)
        self.assertEqual(str(shortlist[0]["candidate_group_key"]), "xgboost__A")
        self.assertEqual(str(shortlist[1]["candidate_group_key"]), "catboost__B")


if __name__ == "__main__":
    unittest.main()
