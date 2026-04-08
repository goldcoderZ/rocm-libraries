# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the ML recommender tile prediction module.

Verifies layout classification, tile prediction, config ranking, and
weight loading via the origami.ml_recommender Python bindings.
"""

import pytest
import origami
from helpers import HARDWARE


# ---- Fixtures ----------------------------------------------------------------


@pytest.fixture
def hardware_gfx950():
    return HARDWARE["gfx950"]


def _make_bf16_problem(m, n, k, transA, transB, batch=1):
    """Helper to create a BFloat16 GEMM problem."""
    p = origami.problem_t()
    p.size = origami.dim3_t(m, n, k)
    p.batch = batch
    p.a_transpose = transA
    p.b_transpose = transB
    p.a_dtype = origami.data_type_t.BFloat16
    p.b_dtype = origami.data_type_t.BFloat16
    p.c_dtype = origami.data_type_t.BFloat16
    p.d_dtype = origami.data_type_t.BFloat16
    p.mi_dtype = origami.data_type_t.BFloat16
    return p


def _make_tf32_problem(m, n, k, transA, transB, batch=1):
    """Helper to create a TF32 (XFloat32) GEMM problem."""
    p = origami.problem_t()
    p.size = origami.dim3_t(m, n, k)
    p.batch = batch
    p.a_transpose = transA
    p.b_transpose = transB
    p.a_dtype = origami.data_type_t.Float
    p.b_dtype = origami.data_type_t.Float
    p.c_dtype = origami.data_type_t.Float
    p.d_dtype = origami.data_type_t.Float
    p.mi_dtype = origami.data_type_t.XFloat32
    return p


# ---- Layout classification --------------------------------------------------


class TestClassifyLayout:
    """Tests for origami.ml_recommender.classify_layout()."""

    def test_bbs_tn(self):
        p = _make_bf16_problem(1024, 2048, 512, origami.transpose_t.T, origami.transpose_t.N)
        assert origami.ml_recommender.classify_layout(p) == origami.ml_recommender.BBS_TN

    def test_bbs_nn(self):
        p = _make_bf16_problem(1024, 2048, 512, origami.transpose_t.N, origami.transpose_t.N)
        assert origami.ml_recommender.classify_layout(p) == origami.ml_recommender.BBS_NN

    def test_sssmx_nt(self):
        p = _make_tf32_problem(1024, 2048, 512, origami.transpose_t.N, origami.transpose_t.T)
        assert origami.ml_recommender.classify_layout(p) == origami.ml_recommender.SSS_MX_NT

    def test_half_maps_to_bbs_tn(self):
        p = _make_bf16_problem(1024, 2048, 512, origami.transpose_t.T, origami.transpose_t.N)
        p.a_dtype = origami.data_type_t.Half
        p.b_dtype = origami.data_type_t.Half
        assert origami.ml_recommender.classify_layout(p) == origami.ml_recommender.BBS_TN

    def test_fallback_for_unsupported_layout(self):
        """BBS_TT (both transposed) has no trained model; should fall back to BBS_NN."""
        p = _make_bf16_problem(1024, 2048, 512, origami.transpose_t.T, origami.transpose_t.T)
        assert origami.ml_recommender.classify_layout(p) == origami.ml_recommender.BBS_NN

    def test_f32_sgemm_falls_back(self):
        """Pure SGEMM (mi_dtype=Float) should not match SSS_MX_NT."""
        p = origami.problem_t()
        p.size = origami.dim3_t(1024, 2048, 512)
        p.batch = 1
        p.a_transpose = origami.transpose_t.N
        p.b_transpose = origami.transpose_t.T
        p.a_dtype = origami.data_type_t.Float
        p.b_dtype = origami.data_type_t.Float
        p.mi_dtype = origami.data_type_t.Float
        assert origami.ml_recommender.classify_layout(p) == origami.ml_recommender.BBS_NN


# ---- Tile prediction ---------------------------------------------------------


class TestPredictTile:
    """Tests for origami.ml_recommender.predict_tile()."""

    def test_returns_valid_dims(self):
        p = _make_bf16_problem(2048, 4096, 1024, origami.transpose_t.T, origami.transpose_t.N)
        tile = origami.ml_recommender.predict_tile(p)
        assert tile.mt_m > 0
        assert tile.mt_n > 0
        assert tile.mt_k > 0

    def test_deterministic(self):
        p = _make_bf16_problem(512, 1024, 256, origami.transpose_t.T, origami.transpose_t.N)
        t1 = origami.ml_recommender.predict_tile(p)
        t2 = origami.ml_recommender.predict_tile(p)
        assert t1.mt_m == t2.mt_m
        assert t1.mt_n == t2.mt_n
        assert t1.mt_k == t2.mt_k
        assert t1.score == pytest.approx(t2.score)

    @pytest.mark.skipif(
        not origami.ml_recommender.weights_loaded(),
        reason="Weights not loaded",
    )
    def test_different_shapes_different_tiles(self):
        small = _make_bf16_problem(32, 64, 128, origami.transpose_t.T, origami.transpose_t.N)
        large = _make_bf16_problem(16384, 16384, 4096, origami.transpose_t.T, origami.transpose_t.N)
        ts = origami.ml_recommender.predict_tile(small)
        tl = origami.ml_recommender.predict_tile(large)
        assert (ts.mt_m, ts.mt_n, ts.mt_k) != (tl.mt_m, tl.mt_n, tl.mt_k)

    @pytest.mark.skipif(
        not origami.ml_recommender.weights_loaded(),
        reason="Weights not loaded",
    )
    def test_batched_problem(self):
        p = _make_bf16_problem(64, 64, 128, origami.transpose_t.T, origami.transpose_t.N, batch=2048)
        tile = origami.ml_recommender.predict_tile(p)
        assert tile.mt_m > 0
        assert tile.score != pytest.approx(-1.0)


# ---- Config ranking ----------------------------------------------------------


class TestRankConfigs:
    """Tests for origami.ml_recommender.rank_configs()."""

    def _make_configs(self):
        configs = []
        for mt_m, mt_n, mt_k in [
            (256, 256, 64),
            (128, 128, 64),
            (192, 256, 64),
            (64, 64, 32),
            (128, 256, 32),
        ]:
            c = origami.config_t()
            c.mt = origami.dim3_t(mt_m, mt_n, mt_k)
            c.mi = origami.dim3_t(16, 16, 16)
            c.occupancy = 1
            configs.append(c)
        return configs

    def test_returns_results(self, hardware_gfx950):
        p = _make_bf16_problem(2048, 2048, 2048, origami.transpose_t.T, origami.transpose_t.N)
        configs = self._make_configs()
        ranked = origami.ml_recommender.rank_configs(p, hardware_gfx950, configs)
        assert len(ranked) > 0
        assert len(ranked) <= len(configs)

    def test_exact_match_ranked_first(self, hardware_gfx950):
        p = _make_bf16_problem(2048, 2048, 2048, origami.transpose_t.T, origami.transpose_t.N)
        predicted = origami.ml_recommender.predict_tile(p)

        configs = self._make_configs()
        exact = origami.config_t()
        exact.mt = origami.dim3_t(predicted.mt_m, predicted.mt_n, predicted.mt_k)
        exact.mi = origami.dim3_t(16, 16, 16)
        exact.occupancy = 1
        configs.append(exact)

        ranked = origami.ml_recommender.rank_configs(p, hardware_gfx950, configs)
        assert ranked[0].config.mt.m == predicted.mt_m
        assert ranked[0].config.mt.n == predicted.mt_n
        assert ranked[0].config.mt.k == predicted.mt_k

    def test_empty_configs(self, hardware_gfx950):
        p = _make_bf16_problem(1024, 1024, 1024, origami.transpose_t.T, origami.transpose_t.N)
        ranked = origami.ml_recommender.rank_configs(p, hardware_gfx950, [])
        assert len(ranked) == 0

    def test_stable_ordering(self, hardware_gfx950):
        p = _make_bf16_problem(4096, 4096, 512, origami.transpose_t.T, origami.transpose_t.N)
        configs = self._make_configs()
        r1 = origami.ml_recommender.rank_configs(p, hardware_gfx950, configs)
        r2 = origami.ml_recommender.rank_configs(p, hardware_gfx950, configs)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.config.mt.m == b.config.mt.m
            assert a.config.mt.n == b.config.mt.n
            assert a.config.mt.k == b.config.mt.k


# ---- Weight loading ----------------------------------------------------------


class TestWeights:
    """Tests for weight loading functions."""

    def test_weights_loaded_returns_bool(self):
        result = origami.ml_recommender.weights_loaded()
        assert isinstance(result, bool)

    def test_load_nonexistent_returns_false(self):
        result = origami.ml_recommender.load_weights("/nonexistent/path.bin")
        assert result is False


# ---- prediction_modes_t enum ------------------------------------------------


class TestPredictionMode:
    """Verify ml_recommender is available in prediction_modes_t."""

    def test_ml_recommender_enum_exists(self):
        mode = origami.prediction_modes_t.ml_recommender
        assert mode is not None
        assert int(mode) == 2
