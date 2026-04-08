/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "common.hpp"
#include "origami/ml_recommender.hpp"

using Catch::Approx;
namespace ml = origami::ml_recommender;

// ---------------------------------------------------------------------------
// classify_layout
// ---------------------------------------------------------------------------

TEST_CASE("ML Recommender: classify_layout BBS_TN", "[ml_recommender]") {
  auto problem    = make_problem(1024, 2048, 512, origami::transpose_t::T, origami::transpose_t::N);
  problem.a_dtype = origami::data_type_t::BFloat16;
  problem.b_dtype = origami::data_type_t::BFloat16;
  auto key        = ml::classify_layout(problem);
  REQUIRE(key == ml::layout_key_t::BBS_TN);
}

TEST_CASE("ML Recommender: classify_layout BBS_NN", "[ml_recommender]") {
  auto problem    = make_problem(1024, 2048, 512, origami::transpose_t::N, origami::transpose_t::N);
  problem.a_dtype = origami::data_type_t::BFloat16;
  problem.b_dtype = origami::data_type_t::BFloat16;
  auto key        = ml::classify_layout(problem);
  REQUIRE(key == ml::layout_key_t::BBS_NN);
}

TEST_CASE("ML Recommender: classify_layout SSS_MX_NT", "[ml_recommender]") {
  auto problem    = make_problem(1024, 2048, 512, origami::transpose_t::N, origami::transpose_t::T);
  problem.a_dtype = origami::data_type_t::Float;
  problem.b_dtype = origami::data_type_t::Float;
  problem.mi_dtype = origami::data_type_t::XFloat32;
  auto key         = ml::classify_layout(problem);
  REQUIRE(key == ml::layout_key_t::SSS_MX_NT);
}

TEST_CASE("ML Recommender: classify_layout fallback to BBS_NN", "[ml_recommender]") {
  auto problem    = make_problem(1024, 2048, 512, origami::transpose_t::T, origami::transpose_t::T);
  problem.a_dtype = origami::data_type_t::BFloat16;
  problem.b_dtype = origami::data_type_t::BFloat16;
  auto key        = ml::classify_layout(problem);
  REQUIRE(key == ml::layout_key_t::BBS_NN);
}

TEST_CASE("ML Recommender: classify_layout Half maps to BBS_TN", "[ml_recommender]") {
  auto problem    = make_problem(1024, 2048, 512, origami::transpose_t::T, origami::transpose_t::N);
  problem.a_dtype = origami::data_type_t::Half;
  problem.b_dtype = origami::data_type_t::Half;
  auto key        = ml::classify_layout(problem);
  REQUIRE(key == ml::layout_key_t::BBS_TN);
}

// ---------------------------------------------------------------------------
// predict_tile (requires binary weights at runtime)
// ---------------------------------------------------------------------------

TEST_CASE("ML Recommender: predict_tile returns valid dimensions", "[ml_recommender]") {
  auto problem    = make_problem(2048, 4096, 1024);
  problem.a_dtype = origami::data_type_t::BFloat16;
  problem.b_dtype = origami::data_type_t::BFloat16;
  auto tile       = ml::predict_tile(problem);

  // Even without weights, predict_tile returns a safe default
  REQUIRE(tile.mt_m > 0);
  REQUIRE(tile.mt_n > 0);
  REQUIRE(tile.mt_k > 0);
}

TEST_CASE("ML Recommender: predict_tile deterministic", "[ml_recommender]") {
  auto problem    = make_problem(512, 1024, 256);
  problem.a_dtype = origami::data_type_t::BFloat16;
  problem.b_dtype = origami::data_type_t::BFloat16;

  auto tile1 = ml::predict_tile(problem);
  auto tile2 = ml::predict_tile(problem);

  REQUIRE(tile1.mt_m == tile2.mt_m);
  REQUIRE(tile1.mt_n == tile2.mt_n);
  REQUIRE(tile1.mt_k == tile2.mt_k);
  REQUIRE(tile1.score == Approx(tile2.score));
}

TEST_CASE("ML Recommender: predict_tile different shapes produce different tiles",
          "[ml_recommender]") {
  auto small_problem    = make_problem(32, 64, 128);
  small_problem.a_dtype = origami::data_type_t::BFloat16;
  small_problem.b_dtype = origami::data_type_t::BFloat16;

  auto large_problem    = make_problem(16384, 16384, 4096);
  large_problem.a_dtype = origami::data_type_t::BFloat16;
  large_problem.b_dtype = origami::data_type_t::BFloat16;

  auto tile_small = ml::predict_tile(small_problem);
  auto tile_large = ml::predict_tile(large_problem);

  if (ml::weights_loaded()) {
    bool different = (tile_small.mt_m != tile_large.mt_m) || (tile_small.mt_n != tile_large.mt_n) ||
                     (tile_small.mt_k != tile_large.mt_k);
    REQUIRE(different);
  }
}

// ---------------------------------------------------------------------------
// rank_configs
// ---------------------------------------------------------------------------

TEST_CASE("ML Recommender: rank_configs returns all valid configs", "[ml_recommender]") {
  auto hardware   = make_hardware(950);
  auto problem    = make_problem(2048, 2048, 2048);
  problem.a_dtype = origami::data_type_t::BFloat16;
  problem.b_dtype = origami::data_type_t::BFloat16;

  std::vector<origami::config_t> configs;
  configs.push_back(make_config(256, 256, 64));
  configs.push_back(make_config(128, 128, 64));
  configs.push_back(make_config(192, 256, 64));
  configs.push_back(make_config(64, 64, 32));

  auto ranked = ml::rank_configs(problem, hardware, configs);
  REQUIRE(ranked.size() > 0);
  REQUIRE(ranked.size() <= configs.size());
}

TEST_CASE("ML Recommender: rank_configs prefers exact tile match", "[ml_recommender]") {
  auto hardware   = make_hardware(950);
  auto problem    = make_problem(2048, 2048, 2048);
  problem.a_dtype = origami::data_type_t::BFloat16;
  problem.b_dtype = origami::data_type_t::BFloat16;

  auto predicted = ml::predict_tile(problem);

  // Use small tiles that always pass LDS capacity, plus the predicted tile
  std::vector<origami::config_t> configs;
  configs.push_back(make_config(64, 64, 32));
  configs.push_back(make_config(128, 128, 32));
  configs.push_back(
      make_config(predicted.mt_m, predicted.mt_n, predicted.mt_k));

  auto ranked = ml::rank_configs(problem, hardware, configs);
  REQUIRE(ranked.size() > 0);

  bool found_predicted = false;
  for (const auto& r : ranked) {
    if (r.config.mt.m == predicted.mt_m && r.config.mt.n == predicted.mt_n &&
        r.config.mt.k == predicted.mt_k) {
      found_predicted = true;
      break;
    }
  }
  REQUIRE(found_predicted);

  if (ranked[0].config.mt.m == predicted.mt_m &&
      ranked[0].config.mt.n == predicted.mt_n) {
    REQUIRE(ranked[0].config.mt.k == predicted.mt_k);
  }
}

TEST_CASE("ML Recommender: rank_configs handles empty configs", "[ml_recommender]") {
  auto hardware   = make_hardware(950);
  auto problem    = make_problem(1024, 1024, 1024);
  problem.a_dtype = origami::data_type_t::BFloat16;
  problem.b_dtype = origami::data_type_t::BFloat16;

  std::vector<origami::config_t> configs;
  auto ranked = ml::rank_configs(problem, hardware, configs);
  REQUIRE(ranked.empty());
}

TEST_CASE("ML Recommender: rank_configs stable across calls", "[ml_recommender]") {
  auto hardware   = make_hardware(950);
  auto problem    = make_problem(4096, 4096, 512);
  problem.a_dtype = origami::data_type_t::BFloat16;
  problem.b_dtype = origami::data_type_t::BFloat16;

  std::vector<origami::config_t> configs;
  configs.push_back(make_config(256, 256, 64));
  configs.push_back(make_config(128, 256, 64));
  configs.push_back(make_config(256, 128, 64));
  configs.push_back(make_config(192, 192, 64));

  auto ranked1 = ml::rank_configs(problem, hardware, configs);
  auto ranked2 = ml::rank_configs(problem, hardware, configs);

  REQUIRE(ranked1.size() == ranked2.size());
  for (std::size_t i = 0; i < ranked1.size(); ++i) {
    REQUIRE(ranked1[i].config.mt.m == ranked2[i].config.mt.m);
    REQUIRE(ranked1[i].config.mt.n == ranked2[i].config.mt.n);
    REQUIRE(ranked1[i].config.mt.k == ranked2[i].config.mt.k);
  }
}

// ---------------------------------------------------------------------------
// Multi-layout dispatch
// ---------------------------------------------------------------------------

TEST_CASE("ML Recommender: different layouts use different models", "[ml_recommender]") {
  auto tn_problem = make_problem(1024, 2048, 512, origami::transpose_t::T, origami::transpose_t::N);
  tn_problem.a_dtype = origami::data_type_t::BFloat16;
  tn_problem.b_dtype = origami::data_type_t::BFloat16;

  auto nn_problem = make_problem(1024, 2048, 512, origami::transpose_t::N, origami::transpose_t::N);
  nn_problem.a_dtype = origami::data_type_t::BFloat16;
  nn_problem.b_dtype = origami::data_type_t::BFloat16;

  auto tn_key = ml::classify_layout(tn_problem);
  auto nn_key = ml::classify_layout(nn_problem);
  REQUIRE(tn_key != nn_key);

  if (ml::weights_loaded()) {
    const auto& tn_w = ml::get_layout_weights(tn_key);
    const auto& nn_w = ml::get_layout_weights(nn_key);
    REQUIRE(tn_w.num_clusters > 0);
    REQUIRE(nn_w.num_clusters > 0);
  }
}

// ---------------------------------------------------------------------------
// weights_loaded / load_weights
// ---------------------------------------------------------------------------

TEST_CASE("ML Recommender: weights_loaded returns bool", "[ml_recommender]") {
  bool loaded = ml::weights_loaded();
  (void)loaded;
}
