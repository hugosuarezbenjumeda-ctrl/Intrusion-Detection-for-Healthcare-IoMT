                      
import unittest

import numpy as np

import evaluate_xgb_robustness as eval_xgb


def _simple_profile(n_features: int) -> dict:
    return {
        "mutable_idx": np.arange(n_features, dtype=np.int64),
        "locked_idx": np.empty(0, dtype=np.int64),
        "locked_mask": np.zeros(n_features, dtype=bool),
        "span": np.ones(n_features, dtype=np.float64),
        "relative_caps": np.full(n_features, 10.0, dtype=np.float64),
        "lower": np.full(n_features, -10.0, dtype=np.float64),
        "upper": np.full(n_features, 10.0, dtype=np.float64),
        "relation_idx": {
            "min": None,
            "avg": None,
            "max": None,
            "tot_sum": None,
            "number": None,
            "srate": None,
            "drate": None,
            "rate": None,
        },
        "min_avg_max_enabled": False,
        "ratio_tot_num_avg_enabled": False,
        "ratio_tot_num_avg_low": None,
        "ratio_tot_num_avg_high": None,
        "ratio_rate_enabled": False,
        "ratio_rate_low": None,
        "ratio_rate_high": None,
    }


class EvaluateXgbHillclimbTests(unittest.TestCase):
    def test_fast_projection_refine_respects_budget(self) -> None:
        rng = np.random.default_rng(19)
        x_orig = rng.normal(size=(24, 6)).astype(np.float32)
        baseline_margin = np.full(24, 1.0, dtype=np.float64)
        profile = _simple_profile(x_orig.shape[1])

        out = eval_xgb.run_query_sparse_hillclimb(
            x_orig=x_orig,
            baseline_margin=baseline_margin,
            booster=None,
            threshold=0.5,
            profile=profile,
            epsilon=0.10,
            objective="minimize",
            query_budget=9,
            max_steps=5,
            candidates_per_step=4,
            feature_subset_size=3,
            score_margin_fn=lambda x: np.asarray(x[:, 0], dtype=np.float64),
            score_batch_rows=12,
            fast_projection=True,
            refine_topk=2,
            rng=np.random.default_rng(1234),
            progress_prefix="test_fast_proj",
            start_ts=0.0,
        )
        self.assertLessEqual(int(np.max(out["queries_used"])), 9)
        summary = out["summary"]
        self.assertTrue(bool(summary.get("fast_projection", False)))
        self.assertEqual(int(summary.get("refine_topk", 0)), 2)
        self.assertGreater(int(summary.get("fast_projection_rows", 0)), 0)

    def test_active_row_cap_is_respected(self) -> None:
        rng = np.random.default_rng(11)
        x_orig = rng.normal(size=(20, 5)).astype(np.float32)
        baseline_margin = np.full(20, 1.0, dtype=np.float64)
        profile = _simple_profile(x_orig.shape[1])

        out = eval_xgb.run_query_sparse_hillclimb(
            x_orig=x_orig,
            baseline_margin=baseline_margin,
            booster=None,
            threshold=0.5,
            profile=profile,
            epsilon=0.10,
            objective="minimize",
            query_budget=12,
            max_steps=4,
            candidates_per_step=2,
            feature_subset_size=2,
            score_margin_fn=lambda x: np.asarray(x[:, 0], dtype=np.float64),
            score_batch_rows=8,
            max_active_rows_per_step=4,
            rng=np.random.default_rng(123),
            progress_prefix="test_cap",
            start_ts=0.0,
        )
        summary = out["summary"]
        self.assertEqual(int(summary["max_active_rows_per_step"]), 4)
        self.assertLessEqual(int(summary["active_rows_max_per_step"]), 4)
        self.assertGreaterEqual(int(summary["steps_executed"]), 1)

    def test_stagnation_early_stop_triggers(self) -> None:
        rng = np.random.default_rng(13)
        x_orig = rng.normal(size=(16, 4)).astype(np.float32)
        baseline_margin = np.full(16, 1.0, dtype=np.float64)
        profile = _simple_profile(x_orig.shape[1])

        out = eval_xgb.run_query_sparse_hillclimb(
            x_orig=x_orig,
            baseline_margin=baseline_margin,
            booster=None,
            threshold=0.5,
            profile=profile,
            epsilon=0.10,
            objective="minimize",
            query_budget=100,
            max_steps=20,
            candidates_per_step=2,
            feature_subset_size=2,
            score_margin_fn=lambda x: np.ones(x.shape[0], dtype=np.float64),
            score_batch_rows=8,
            stagnation_patience=2,
            stagnation_min_delta=1e-6,
            rng=np.random.default_rng(7),
            progress_prefix="test_stagnation",
            start_ts=0.0,
        )
        summary = out["summary"]
        self.assertTrue(bool(summary["stagnation_triggered"]))
        self.assertLessEqual(int(summary["steps_executed"]), 2)
        self.assertEqual(int(summary["steps_with_progress"]), 0)

    def test_backward_compatibility_when_controls_unset(self) -> None:
        rng = np.random.default_rng(17)
        x_orig = rng.normal(size=(12, 3)).astype(np.float32)
        baseline_margin = np.full(12, 1.0, dtype=np.float64)
        profile = _simple_profile(x_orig.shape[1])
        scorer = lambda x: np.asarray(x[:, 0], dtype=np.float64)

        out_default = eval_xgb.run_query_sparse_hillclimb(
            x_orig=x_orig,
            baseline_margin=baseline_margin,
            booster=None,
            threshold=0.5,
            profile=profile,
            epsilon=0.05,
            objective="minimize",
            query_budget=8,
            max_steps=5,
            candidates_per_step=2,
            feature_subset_size=2,
            score_margin_fn=scorer,
            score_batch_rows=4,
            rng=np.random.default_rng(99),
            progress_prefix="test_backcompat_a",
            start_ts=0.0,
        )
        out_explicit = eval_xgb.run_query_sparse_hillclimb(
            x_orig=x_orig,
            baseline_margin=baseline_margin,
            booster=None,
            threshold=0.5,
            profile=profile,
            epsilon=0.05,
            objective="minimize",
            query_budget=8,
            max_steps=5,
            candidates_per_step=2,
            feature_subset_size=2,
            score_margin_fn=scorer,
            score_batch_rows=4,
            max_active_rows_per_step=None,
            stagnation_patience=None,
            stagnation_min_delta=1e-6,
            rng=np.random.default_rng(99),
            progress_prefix="test_backcompat_b",
            start_ts=0.0,
        )
        self.assertTrue(np.array_equal(out_default["queries_used"], out_explicit["queries_used"]))
        self.assertTrue(np.array_equal(out_default["accepted_moves"], out_explicit["accepted_moves"]))
        self.assertTrue(np.array_equal(out_default["success"], out_explicit["success"]))
        self.assertTrue(np.allclose(out_default["final_margin"], out_explicit["final_margin"], atol=1e-12, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
