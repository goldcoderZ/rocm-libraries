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

#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "origami/hardware.hpp"
#include "origami/types.hpp"

namespace origami {
namespace ml_recommender {

/**
 * @brief Supported layout/datatype combinations for ML-based tile prediction.
 *
 * Each layout key corresponds to a separately trained InteractionTwoTower model
 * with its own Voronoi centroids and neural network weights. Untrained
 * layout/datatype combinations fall back to BBS_NN.
 */
enum class layout_key_t : int {
  BBS_TN      = 0,  ///< BFloat16/Half, transA=T transB=N
  BBS_NN      = 1,  ///< BFloat16/Half, transA=N transB=N (also used as fallback)
  SSS_MX_NT   = 2,  ///< XFloat32 (TF32), transA=N transB=T
  NUM_LAYOUTS = 3
};

/**
 * @brief Per-layout model weights for the InteractionTwoTower predictor.
 *
 * Holds pointers to all arrays required for inference: Voronoi centroids,
 * tile geometry tables, pre-computed tile embeddings, query-tower MLP weights,
 * interaction MLP weights, and normalization constants. Float pointers are
 * populated at runtime from a binary FP16 weight file; integer arrays are
 * compiled in as constexpr from the header.
 */
struct layout_weights_t {
  std::size_t num_clusters;  ///< Number of trained Voronoi clusters.
  std::size_t n_centroids;   ///< Number of Voronoi centroids (>= num_clusters).
  std::size_t total_tiles;   ///< Total candidate tiles across all clusters.
  std::size_t inter_hidden;  ///< Hidden dimension of the interaction MLP.
  float w_b;                 ///< Batch-dimension weight in log-space distance.

  const float* centroids;          ///< [n_centroids * 4] 4D log-space centroids.
  const int* centroid_to_cluster;  ///< [n_centroids] centroid -> cluster index (-1 = unused).
  const int* tile_offsets;         ///< [num_clusters] start index into tile arrays.
  const int* tile_counts;          ///< [num_clusters] number of tiles per cluster.
  const int* tile_mt_m;            ///< [total_tiles] macro-tile M dimension.
  const int* tile_mt_n;            ///< [total_tiles] macro-tile N dimension.
  const int* tile_mt_k;            ///< [total_tiles] macro-tile K dimension.
  const float* tile_embeddings;    ///< [total_tiles * EMBED_DIM] pre-computed item embeddings.
  const float* gemm_mean;          ///< [num_clusters * GEMM_DIM] per-cluster feature means.
  const float* gemm_std;           ///< [num_clusters * GEMM_DIM] per-cluster feature stds.
  const float* inter_mean;         ///< [num_clusters * INTER_DIM] interaction feature means.
  const float* inter_std;          ///< [num_clusters * INTER_DIM] interaction feature stds.
  const float* qw1;                ///< Query tower layer 1 weights [HIDDEN_DIM x GEMM_DIM].
  const float* qb1;                ///< Query tower layer 1 biases [HIDDEN_DIM].
  const float* qw2;                ///< Query tower layer 2 weights [HIDDEN_DIM x HIDDEN_DIM].
  const float* qb2;                ///< Query tower layer 2 biases [HIDDEN_DIM].
  const float* qw3;                ///< Query tower layer 3 weights [HIDDEN_DIM x HIDDEN_DIM].
  const float* qb3;                ///< Query tower layer 3 biases [HIDDEN_DIM].
  const float* qw4;                ///< Query tower layer 4 weights [EMBED_DIM x HIDDEN_DIM].
  const float* qb4;                ///< Query tower layer 4 biases [EMBED_DIM].
  const float* iw1;                ///< Interaction MLP layer 1 weights [inter_hidden x INTER_DIM].
  const float* ib1;                ///< Interaction MLP layer 1 biases [inter_hidden].
  const float* iw2;                ///< Interaction MLP layer 2 weights [1 x inter_hidden].
  const float* ib2;                ///< Interaction MLP layer 2 biases [1].
};

/**
 * @brief Predicted macro-tile dimensions and confidence score.
 *
 */
struct predicted_tile_t {
  std::size_t mt_m;  ///< Predicted macro-tile M dimension.
  std::size_t mt_n;  ///< Predicted macro-tile N dimension.
  std::size_t mt_k;  ///< Predicted macro-tile K dimension.
  float score;       ///< Model confidence score (higher = more confident).
};

/**
 * @brief Determine which trained layout model to use for a given GEMM problem.
 *
 * Inspects the problem's transpose flags and data types to select the best
 * matching pre-trained model. Returns BBS_NN as fallback for untrained combos.
 *
 * @param problem GEMM problem description.
 * @return layout_key_t The layout key for weight lookup.
 */
layout_key_t classify_layout(const problem_t& problem);

/**
 * @brief Retrieve the weight set for a given layout.
 *
 * Triggers one-time binary weight loading on first call (thread-safe).
 * Searches for the weight file at: (1) $ML_RECOMMENDER_WEIGHTS env var,
 * (2) ./ml_recommender_weights.bin, (3) /opt/rocm/share/origami/.
 *
 * @param key Layout key from classify_layout().
 * @return const layout_weights_t& Reference to the layout's weight data.
 */
const layout_weights_t& get_layout_weights(layout_key_t key);

/**
 * @brief Predict the optimal macro-tile for a GEMM problem.
 *
 * Runs the full InteractionTwoTower inference pipeline:
 *   1. Classify layout -> select weight set
 *   2. Voronoi nearest-centroid lookup -> cluster index
 *   3. Compute 45-dim GEMM features, Z-score normalize
 *   4. Query tower 4-layer MLP -> 32-dim embedding
 *   5. Score all candidate tiles: dot(embedding, tile_emb) + interaction MLP
 *   6. Return the highest-scoring tile
 *
 * Returns {256, 256, 64, -1.0} as safe default if weights are unavailable.
 *
 * @param problem GEMM problem description (M, N, K, batch, transpose, dtype).
 * @return predicted_tile_t The predicted macro-tile dimensions and score.
 */
predicted_tile_t predict_tile(const problem_t& problem);

/**
 * @brief Rank kernel configurations by ML-predicted tile affinity.
 *
 * Uses predict_tile() to find the optimal macro-tile, then scores each
 * configuration by proximity to the predicted tile:
 *   - Exact (MT_M, MT_N, MT_K) match: highest priority
 *   - (MT_M, MT_N) match with MT_K tiebreaker: high priority
 *   - Partial (MT_M or MT_N) match: medium priority
 *   - No match: lowest priority
 *
 * Configurations that fail LDS capacity checks are filtered out. If all
 * configs are filtered, falls back to unfiltered ranking.
 *
 * If weights are unavailable, returns configs in original order with
 * uniform latency scores (graceful degradation to Origami analytical path).
 *
 * @param problem GEMM problem description.
 * @param hardware Hardware characteristics (@see origami::hardware_t).
 * @param configs Candidate kernel configurations to rank.
 * @return std::vector<prediction_result_t> Configs ranked by ML score (best first).
 */
std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const std::vector<config_t>& configs);

/**
 * @brief Check whether ML recommender weights have been loaded.
 *
 * @return true if binary weights were successfully loaded, false otherwise.
 */
bool weights_loaded();

/**
 * @brief Explicitly load ML recommender weights from a binary file.
 *
 * Normally weights are loaded lazily on first predict_tile()/rank_configs()
 * call. This function allows explicit loading, e.g. during library init.
 *
 * @param bin_path Path to the ml_recommender_weights.bin file.
 * @return true if loading succeeded, false otherwise.
 */
bool load_weights(const std::string& bin_path);

}  // namespace ml_recommender
}  // namespace origami
