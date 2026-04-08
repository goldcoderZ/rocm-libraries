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

#include "origami/ml_recommender.hpp"
#include "origami/gemm.hpp"
#include "origami/ml_recommender_weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <vector>

#ifdef __x86_64__
#include <cpuid.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace origami {
namespace ml_recommender {

namespace {

// ---------------------------------------------------------------------------
// SIMD runtime dispatch
// ---------------------------------------------------------------------------

float scalar_dot(const float* a, const float* b, std::size_t n) {
  float sum = 0.0f;
  for (std::size_t j = 0; j < n; ++j) sum += a[j] * b[j];
  return sum;
}

#ifdef __AVX512F__
float avx512_dot(const float* a, const float* b, std::size_t n) {
  __m512 acc    = _mm512_setzero_ps();
  std::size_t j = 0;
  for (; j + 16 <= n; j += 16)
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + j), _mm512_loadu_ps(b + j), acc);
  float sum = _mm512_reduce_add_ps(acc);
  for (; j < n; ++j) sum += a[j] * b[j];
  return sum;
}
#endif

using dot_fn_t = float (*)(const float*, const float*, std::size_t);

dot_fn_t select_dot_fn() {
#ifdef __AVX512F__
#ifdef __x86_64__
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) && (ebx & (1u << 16))) return avx512_dot;
#endif
#endif
  return scalar_dot;
}

static const dot_fn_t dot = select_dot_fn();

// ---------------------------------------------------------------------------
// Binary weight loading (FP16 -> FP32)
// ---------------------------------------------------------------------------

static std::vector<float> g_weight_storage;
static bool g_weights_loaded = false;
static std::once_flag g_load_once;

struct layout_float_ptrs_t {
  float* centroids       = nullptr;
  float* tile_embeddings = nullptr;
  float* gemm_mean       = nullptr;
  float* gemm_std        = nullptr;
  float* inter_mean      = nullptr;
  float* inter_std       = nullptr;
  float* qw1             = nullptr;
  float* qb1             = nullptr;
  float* qw2             = nullptr;
  float* qb2             = nullptr;
  float* qw3             = nullptr;
  float* qb3             = nullptr;
  float* qw4             = nullptr;
  float* qb4             = nullptr;
  float* iw1             = nullptr;
  float* ib1             = nullptr;
  float* iw2             = nullptr;
  float* ib2             = nullptr;
};

static layout_float_ptrs_t g_float_ptrs[3];
static layout_weights_t g_layout_table[3];

float half_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp  = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign << 31;
    } else {
      exp = 1;
      while (!(mant & 0x400)) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3ff;
      f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 0x1f) {
    f = (sign << 31) | 0x7f800000 | (mant << 13);
  } else {
    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float result;
  std::memcpy(&result, &f, sizeof(float));
  return result;
}

bool load_binary_weights(const char* bin_path) {
  std::ifstream file(bin_path, std::ios::binary);
  if (!file.is_open()) return false;

  char magic[4];
  file.read(magic, 4);
  if (std::strncmp(magic, "MLRW", 4) != 0) return false;

  uint32_t version, num_layouts;
  file.read(reinterpret_cast<char*>(&version), 4);
  file.read(reinterpret_cast<char*>(&num_layouts), 4);
  if (version != 2 || num_layouts > 3) return false;

  struct ArrayMapping {
    const char* name;
    std::size_t layout_idx;
    float** ptr;
  };

  // First pass: count total floats
  auto header_pos          = file.tellg();
  std::size_t total_floats = 0;
  for (uint32_t li = 0; li < num_layouts; ++li) {
    uint32_t name_len;
    file.read(reinterpret_cast<char*>(&name_len), 4);
    file.seekg(name_len, std::ios::cur);
    uint32_t num_arrays;
    file.read(reinterpret_cast<char*>(&num_arrays), 4);
    for (uint32_t ai = 0; ai < num_arrays; ++ai) {
      uint32_t arr_name_len;
      file.read(reinterpret_cast<char*>(&arr_name_len), 4);
      file.seekg(arr_name_len, std::ios::cur);
      uint32_t num_elements;
      file.read(reinterpret_cast<char*>(&num_elements), 4);
      total_floats += num_elements;
      file.seekg(num_elements * 2, std::ios::cur);
    }
  }

  g_weight_storage.resize(total_floats);

  // Layout name -> index mapping
  auto layout_name_to_idx = [](const std::string& name) -> int {
    if (name == "bbs_tn") return 0;
    if (name == "bbs_nn") return 1;
    if (name == "sssmx_nt") return 2;
    return -1;
  };

  // Array name -> pointer offset in layout_float_ptrs_t
  auto assign_ptr = [](layout_float_ptrs_t& ptrs, const std::string& name, float* p) {
    if (name == "CENTROIDS")
      ptrs.centroids = p;
    else if (name == "ALL_TILE_EMBEDDINGS")
      ptrs.tile_embeddings = p;
    else if (name == "GEMM_MEAN")
      ptrs.gemm_mean = p;
    else if (name == "GEMM_STD")
      ptrs.gemm_std = p;
    else if (name == "INTER_MEAN")
      ptrs.inter_mean = p;
    else if (name == "INTER_STD")
      ptrs.inter_std = p;
    else if (name == "QW1")
      ptrs.qw1 = p;
    else if (name == "QB1")
      ptrs.qb1 = p;
    else if (name == "QW2")
      ptrs.qw2 = p;
    else if (name == "QB2")
      ptrs.qb2 = p;
    else if (name == "QW3")
      ptrs.qw3 = p;
    else if (name == "QB3")
      ptrs.qb3 = p;
    else if (name == "QW4")
      ptrs.qw4 = p;
    else if (name == "QB4")
      ptrs.qb4 = p;
    else if (name == "IW1")
      ptrs.iw1 = p;
    else if (name == "IB1")
      ptrs.ib1 = p;
    else if (name == "IW2")
      ptrs.iw2 = p;
    else if (name == "IB2")
      ptrs.ib2 = p;
  };

  // Second pass: read and convert
  file.seekg(header_pos);
  std::size_t offset = 0;
  std::vector<uint16_t> read_buf;

  for (uint32_t li = 0; li < num_layouts; ++li) {
    uint32_t name_len;
    file.read(reinterpret_cast<char*>(&name_len), 4);
    std::string layout_name(name_len, '\0');
    file.read(&layout_name[0], name_len);

    int layout_idx = layout_name_to_idx(layout_name);
    if (layout_idx < 0) continue;

    uint32_t num_arrays;
    file.read(reinterpret_cast<char*>(&num_arrays), 4);

    for (uint32_t ai = 0; ai < num_arrays; ++ai) {
      uint32_t arr_name_len;
      file.read(reinterpret_cast<char*>(&arr_name_len), 4);
      std::string arr_name(arr_name_len, '\0');
      file.read(&arr_name[0], arr_name_len);

      uint32_t num_elements;
      file.read(reinterpret_cast<char*>(&num_elements), 4);

      read_buf.resize(num_elements);
      file.read(reinterpret_cast<char*>(read_buf.data()), num_elements * 2);

      for (uint32_t i = 0; i < num_elements; ++i)
        g_weight_storage[offset + i] = half_to_float(read_buf[i]);

      assign_ptr(g_float_ptrs[layout_idx], arr_name, &g_weight_storage[offset]);
      offset += num_elements;
    }
  }

  return true;
}

void build_layout_table(int idx,
                        std::size_t num_clusters,
                        std::size_t n_centroids,
                        std::size_t total_tiles,
                        std::size_t inter_hidden,
                        float w_b,
                        const int* centroid_to_cluster,
                        const int* tile_offsets,
                        const int* tile_counts,
                        const int* tile_mt_m,
                        const int* tile_mt_n,
                        const int* tile_mt_k) {
  auto& p             = g_float_ptrs[idx];
  g_layout_table[idx] = {
      num_clusters,
      n_centroids,
      total_tiles,
      inter_hidden,
      w_b,
      p.centroids,
      centroid_to_cluster,
      tile_offsets,
      tile_counts,
      tile_mt_m,
      tile_mt_n,
      tile_mt_k,
      p.tile_embeddings,
      p.gemm_mean,
      p.gemm_std,
      p.inter_mean,
      p.inter_std,
      p.qw1,
      p.qb1,
      p.qw2,
      p.qb2,
      p.qw3,
      p.qb3,
      p.qw4,
      p.qb4,
      p.iw1,
      p.ib1,
      p.iw2,
      p.ib2,
  };
}

void init_weights() {
  const char* env_path = std::getenv("ML_RECOMMENDER_WEIGHTS");
  bool ok              = false;
  if (env_path && env_path[0]) ok = load_binary_weights(env_path);
  if (!ok) ok = load_binary_weights("ml_recommender_weights.bin");
  if (!ok) ok = load_binary_weights("/opt/rocm/share/origami/ml_recommender_weights.bin");

  if (ok) {
    build_layout_table(0,
                       bbs_tn::NUM_CLUSTERS,
                       bbs_tn::N_CENTROIDS,
                       bbs_tn::TOTAL_TILES,
                       bbs_tn::INTER_HIDDEN,
                       bbs_tn::W_B,
                       bbs_tn::CENTROID_TO_CLUSTER,
                       bbs_tn::TILE_OFFSETS,
                       bbs_tn::TILE_COUNTS,
                       bbs_tn::ALL_TILE_MT_M,
                       bbs_tn::ALL_TILE_MT_N,
                       bbs_tn::ALL_TILE_MT_K);
    build_layout_table(1,
                       bbs_nn::NUM_CLUSTERS,
                       bbs_nn::N_CENTROIDS,
                       bbs_nn::TOTAL_TILES,
                       bbs_nn::INTER_HIDDEN,
                       bbs_nn::W_B,
                       bbs_nn::CENTROID_TO_CLUSTER,
                       bbs_nn::TILE_OFFSETS,
                       bbs_nn::TILE_COUNTS,
                       bbs_nn::ALL_TILE_MT_M,
                       bbs_nn::ALL_TILE_MT_N,
                       bbs_nn::ALL_TILE_MT_K);
    build_layout_table(2,
                       sssmx_nt::NUM_CLUSTERS,
                       sssmx_nt::N_CENTROIDS,
                       sssmx_nt::TOTAL_TILES,
                       sssmx_nt::INTER_HIDDEN,
                       sssmx_nt::W_B,
                       sssmx_nt::CENTROID_TO_CLUSTER,
                       sssmx_nt::TILE_OFFSETS,
                       sssmx_nt::TILE_COUNTS,
                       sssmx_nt::ALL_TILE_MT_M,
                       sssmx_nt::ALL_TILE_MT_N,
                       sssmx_nt::ALL_TILE_MT_K);
    g_weights_loaded = true;
  }
}

void ensure_weights() { std::call_once(g_load_once, init_weights); }

// ---------------------------------------------------------------------------
// Feature computation
// ---------------------------------------------------------------------------

float safe_log2(float x) { return std::log2(std::max(x, 1.0f)); }

int find_nearest_centroid(int m, int n, int k, int batch, const layout_weights_t& w) {
  float log_m = safe_log2(static_cast<float>(m));
  float log_n = safe_log2(static_cast<float>(n));
  float log_k = safe_log2(static_cast<float>(k));
  float log_b = w.w_b * safe_log2(static_cast<float>(std::max(batch, 1)));

  float best_dist   = std::numeric_limits<float>::max();
  int best_centroid = 0;

  for (std::size_t i = 0; i < w.n_centroids; ++i) {
    float dm   = log_m - w.centroids[i * 4 + 0];
    float dn   = log_n - w.centroids[i * 4 + 1];
    float dk   = log_k - w.centroids[i * 4 + 2];
    float db   = log_b - w.centroids[i * 4 + 3];
    float dist = dm * dm + dn * dn + dk * dk + db * db;
    if (dist < best_dist) {
      best_dist     = dist;
      best_centroid = static_cast<int>(i);
    }
  }

  int cluster_idx = w.centroid_to_cluster[best_centroid];
  return (cluster_idx >= 0) ? cluster_idx : 0;
}

void compute_gemm_features(int m, int n, int k, int batch, float* out) {
  float fm = static_cast<float>(m), fn = static_cast<float>(n);
  float fk = static_cast<float>(k), fb = static_cast<float>(batch);
  float mn = fm * fn, mk = fm * fk, nk = fn * fk;
  float mem  = (mk + nk + mn) * 2.0f;
  float ai   = (mem > 0) ? (2.0f * fm * fn * fk * fb / mem) : 0;
  float ip2m = ((m & (m - 1)) == 0 && m > 0) ? 1.0f : 0.0f;
  float ip2n = ((n & (n - 1)) == 0 && n > 0) ? 1.0f : 0.0f;

  int idx    = 0;
  out[idx++] = safe_log2(fm);
  out[idx++] = safe_log2(fn);
  out[idx++] = safe_log2(fk);
  out[idx++] = safe_log2(fb);
  out[idx++] = safe_log2(mn);
  out[idx++] = ai;
  out[idx++] = static_cast<float>(m % 256);
  out[idx++] = static_cast<float>(m % 128);
  out[idx++] = static_cast<float>(m % 64);
  out[idx++] = static_cast<float>(m % 32);
  out[idx++] = static_cast<float>(m % 16);
  out[idx++] = static_cast<float>(m % 8);
  out[idx++] = static_cast<float>(n % 256);
  out[idx++] = static_cast<float>(n % 128);
  out[idx++] = static_cast<float>(n % 64);
  out[idx++] = static_cast<float>(n % 32);
  out[idx++] = static_cast<float>(n % 16);
  out[idx++] = static_cast<float>(n % 8);
  out[idx++] = std::min(std::max(fm / std::max(fn, 1.0f), 0.001f), 10000.0f);
  out[idx++] = ip2m;
  out[idx++] = ip2n;

  int mt_sizes[] = {32, 64, 128, 256};
  for (int mt : mt_sizes) {
    int nt_m = (m + mt - 1) / mt, nt_n = (n + mt - 1) / mt;
    out[idx++] = safe_log2(static_cast<float>(nt_m * nt_n));
    out[idx++] = (fm * fn) / (static_cast<float>(nt_m) * mt * nt_n * mt);
  }
  int k_divs[] = {32, 64, 128, 256};
  for (int kd : k_divs) out[idx++] = safe_log2(static_cast<float>((k + kd - 1) / kd));
  out[idx++]     = std::log2(std::max(mk * 2.0f / (1024.0f * 1024.0f), 0.001f));
  out[idx++]     = std::log2(std::max(nk * 2.0f / (1024.0f * 1024.0f), 0.001f));
  int bd_sizes[] = {64, 128, 256};
  for (int bd : bd_sizes) {
    out[idx++] = std::min(static_cast<float>(m % bd), static_cast<float>(bd - m % bd)) / bd;
    out[idx++] = std::min(static_cast<float>(n % bd), static_cast<float>(bd - n % bd)) / bd;
  }
  int wave_mts[] = {128, 256};
  for (int mt : wave_mts) {
    int nt     = ((m + mt - 1) / mt) * ((n + mt - 1) / mt);
    int ncu    = static_cast<int>(N_CU);
    int w      = (nt + ncu - 1) / ncu;
    out[idx++] = safe_log2(static_cast<float>(w));
    out[idx++] = (w > 0) ? static_cast<float>(nt) / (w * static_cast<float>(N_CU)) : 1.0f;
  }
}

void compute_inter_features(int m, int n, int k, int mt_m, int mt_n, int mt_k, float* out) {
  int nt_m      = (mt_m > 0) ? (m + mt_m - 1) / mt_m : 1;
  int nt_n      = (mt_n > 0) ? (n + mt_n - 1) / mt_n : 1;
  int num_tiles = nt_m * nt_n;
  float util    = (mt_m > 0 && mt_n > 0)
                      ? (static_cast<float>(m) * n) / (static_cast<float>(nt_m) * mt_m * nt_n * mt_n)
                      : 1.0f;
  int ncu       = static_cast<int>(N_CU);
  int waves     = (num_tiles + ncu - 1) / ncu;
  float wave_eff =
      (waves > 0) ? static_cast<float>(num_tiles) / (waves * static_cast<float>(N_CU)) : 1.0f;
  int k_iters = (mt_k > 0) ? (k + mt_k - 1) / mt_k : 1;
  int lds     = (mt_m * mt_k + mt_k * mt_n) * 2;

  out[0]  = safe_log2(static_cast<float>(num_tiles));
  out[1]  = util;
  out[2]  = safe_log2(static_cast<float>(waves));
  out[3]  = wave_eff;
  out[4]  = safe_log2(static_cast<float>(k_iters));
  out[5]  = safe_log2(static_cast<float>(lds));
  out[6]  = wave_eff;
  out[7]  = 0.0f;
  out[8]  = 0.0f;
  out[9]  = 0.0f;
  out[10] = 0.0f;
  out[11] = 0.0f;
}

// ---------------------------------------------------------------------------
// MLP forward passes
// ---------------------------------------------------------------------------

void mlp_forward_4layer(const float* input,
                        float* output,
                        int cluster_idx,
                        const layout_weights_t& w) {
  float h1[HIDDEN_DIM], h2[HIDDEN_DIM], h3[HIDDEN_DIM];

  std::size_t off  = cluster_idx * HIDDEN_DIM * GEMM_DIM;
  std::size_t boff = cluster_idx * HIDDEN_DIM;
  for (std::size_t i = 0; i < HIDDEN_DIM; ++i) {
    float s = w.qb1[boff + i] + dot(&w.qw1[off + i * GEMM_DIM], input, GEMM_DIM);
    h1[i]   = std::max(s, 0.0f);
  }

  off  = cluster_idx * HIDDEN_DIM * HIDDEN_DIM;
  boff = cluster_idx * HIDDEN_DIM;
  for (std::size_t i = 0; i < HIDDEN_DIM; ++i) {
    float s = w.qb2[boff + i] + dot(&w.qw2[off + i * HIDDEN_DIM], h1, HIDDEN_DIM);
    h2[i]   = std::max(s, 0.0f);
  }

  off  = cluster_idx * HIDDEN_DIM * HIDDEN_DIM;
  boff = cluster_idx * HIDDEN_DIM;
  for (std::size_t i = 0; i < HIDDEN_DIM; ++i) {
    float s = w.qb3[boff + i] + dot(&w.qw3[off + i * HIDDEN_DIM], h2, HIDDEN_DIM);
    h3[i]   = std::max(s, 0.0f);
  }

  off  = cluster_idx * EMBED_DIM * HIDDEN_DIM;
  boff = cluster_idx * EMBED_DIM;
  for (std::size_t i = 0; i < EMBED_DIM; ++i) {
    output[i] = w.qb4[boff + i] + dot(&w.qw4[off + i * HIDDEN_DIM], h3, HIDDEN_DIM);
  }
}

float inter_mlp_forward(const float* inter_feats_norm, int cluster_idx, const layout_weights_t& w) {
  std::size_t ih      = w.inter_hidden;
  std::size_t iw1_off = cluster_idx * ih * INTER_DIM;
  std::size_t ib1_off = cluster_idx * ih;

  float h[256];
  for (std::size_t i = 0; i < ih; ++i) {
    float s =
        w.ib1[ib1_off + i] + dot(&w.iw1[iw1_off + i * INTER_DIM], inter_feats_norm, INTER_DIM);
    h[i] = std::max(s, 0.0f);
  }
  std::size_t iw2_off = cluster_idx * ih;
  std::size_t ib2_off = cluster_idx;
  return w.ib2[ib2_off] + dot(&w.iw2[iw2_off], h, ih);
}

// ---------------------------------------------------------------------------
// Core prediction
// ---------------------------------------------------------------------------

predicted_tile_t predict_tile_impl(const problem_t& problem, const layout_weights_t& w) {
  int m     = static_cast<int>(problem.size.m);
  int n     = static_cast<int>(problem.size.n);
  int k     = static_cast<int>(problem.size.k);
  int batch = static_cast<int>(problem.batch);

  int cluster_idx = find_nearest_centroid(m, n, k, batch, w);

  float gemm_feats[GEMM_DIM];
  compute_gemm_features(m, n, k, batch, gemm_feats);

  std::size_t g_off = cluster_idx * GEMM_DIM;
  float gemm_norm[GEMM_DIM];
  for (std::size_t i = 0; i < GEMM_DIM; ++i) {
    float std_val = w.gemm_std[g_off + i];
    gemm_norm[i]  = (std_val > 1e-6f) ? (gemm_feats[i] - w.gemm_mean[g_off + i]) / std_val : 0.0f;
  }

  float embedding[EMBED_DIM];
  mlp_forward_4layer(gemm_norm, embedding, cluster_idx, w);

  int t_off         = w.tile_offsets[cluster_idx];
  int t_cnt         = w.tile_counts[cluster_idx];
  std::size_t i_off = cluster_idx * INTER_DIM;

  float best_score = -std::numeric_limits<float>::max();
  int best_tile    = 0;

  for (int t = 0; t < t_cnt; ++t) {
    std::size_t emb_off = (t_off + t) * EMBED_DIM;
    float d             = dot(embedding, &w.tile_embeddings[emb_off], EMBED_DIM);

    float inter_raw[INTER_DIM];
    compute_inter_features(
        m, n, k, w.tile_mt_m[t_off + t], w.tile_mt_n[t_off + t], w.tile_mt_k[t_off + t], inter_raw);
    float inter_norm[INTER_DIM];
    for (std::size_t j = 0; j < INTER_DIM; ++j) {
      float std_val = w.inter_std[i_off + j];
      inter_norm[j] = (std_val > 1e-6f) ? (inter_raw[j] - w.inter_mean[i_off + j]) / std_val : 0.0f;
    }
    float inter_score = inter_mlp_forward(inter_norm, cluster_idx, w);

    float score = d + inter_score;
    if (score > best_score) {
      best_score = score;
      best_tile  = t;
    }
  }

  return {static_cast<std::size_t>(w.tile_mt_m[t_off + best_tile]),
          static_cast<std::size_t>(w.tile_mt_n[t_off + best_tile]),
          static_cast<std::size_t>(w.tile_mt_k[t_off + best_tile]),
          best_score};
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

layout_key_t classify_layout(const problem_t& p) {
  bool tn = (p.a_transpose == transpose_t::T && p.b_transpose == transpose_t::N);
  bool nn = (p.a_transpose == transpose_t::N && p.b_transpose == transpose_t::N);
  bool nt = (p.a_transpose == transpose_t::N && p.b_transpose == transpose_t::T);

  bool is_bf16 = (p.a_dtype == data_type_t::BFloat16 || p.a_dtype == data_type_t::Half);
  bool is_tf32 = (p.mi_dtype == data_type_t::XFloat32 ||
                  (p.a_dtype == data_type_t::Float && p.mi_dtype != data_type_t::Float));

  if (is_bf16 && tn) return layout_key_t::BBS_TN;
  if (is_bf16 && nn) return layout_key_t::BBS_NN;
  if (is_tf32 && nt) return layout_key_t::SSS_MX_NT;

  return layout_key_t::BBS_NN;
}

const layout_weights_t& get_layout_weights(layout_key_t key) {
  ensure_weights();
  int idx = static_cast<int>(key);
  if (idx < 0 || idx >= static_cast<int>(layout_key_t::NUM_LAYOUTS))
    idx = static_cast<int>(layout_key_t::BBS_NN);
  return g_layout_table[idx];
}

predicted_tile_t predict_tile(const problem_t& problem) {
  ensure_weights();
  if (!g_weights_loaded) return {256, 256, 64, -1.0f};
  auto key      = classify_layout(problem);
  const auto& w = get_layout_weights(key);
  return predict_tile_impl(problem, w);
}

std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const std::vector<config_t>& configs) {
  ensure_weights();
  if (!g_weights_loaded) {
    std::vector<prediction_result_t> results;
    results.reserve(configs.size());
    for (const auto& c : configs) results.push_back({1.0, c});
    return results;
  }

  auto predicted = predict_tile(problem);

  struct scored_config_t {
    double score;
    std::size_t original_idx;
  };
  std::vector<scored_config_t> scored;
  scored.reserve(configs.size());

  for (std::size_t i = 0; i < configs.size(); ++i) {
    const auto& c = configs[i];
    if (!check_lds_capacity(hardware, c.mt, problem.a_dtype, problem.b_dtype)) continue;
    double s = 0.0;
    if (c.mt.m == predicted.mt_m && c.mt.n == predicted.mt_n && c.mt.k == predicted.mt_k)
      s = 1e9;
    else if (c.mt.m == predicted.mt_m && c.mt.n == predicted.mt_n)
      s = 1e6 +
          1e3 / (1.0 + std::abs(static_cast<double>(c.mt.k) - static_cast<double>(predicted.mt_k)));
    else if (c.mt.m == predicted.mt_m || c.mt.n == predicted.mt_n)
      s = 1e3;
    scored.push_back({s, i});
  }

  if (scored.empty()) {
    for (std::size_t i = 0; i < configs.size(); ++i) {
      const auto& c = configs[i];
      double s      = 0.0;
      if (c.mt.m == predicted.mt_m && c.mt.n == predicted.mt_n && c.mt.k == predicted.mt_k)
        s = 1e9;
      else if (c.mt.m == predicted.mt_m && c.mt.n == predicted.mt_n)
        s = 1e6;
      else if (c.mt.m == predicted.mt_m || c.mt.n == predicted.mt_n)
        s = 1e3;
      scored.push_back({s, i});
    }
  }

  std::stable_sort(
      scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.score > b.score; });

  std::vector<prediction_result_t> results;
  results.reserve(scored.size());
  for (const auto& sc : scored)
    results.push_back({1.0 / (1.0 + sc.score), configs[sc.original_idx]});
  return results;
}

bool weights_loaded() {
  ensure_weights();
  return g_weights_loaded;
}

bool load_weights(const std::string& bin_path) {
  if (load_binary_weights(bin_path.c_str())) {
    build_layout_table(0,
                       bbs_tn::NUM_CLUSTERS,
                       bbs_tn::N_CENTROIDS,
                       bbs_tn::TOTAL_TILES,
                       bbs_tn::INTER_HIDDEN,
                       bbs_tn::W_B,
                       bbs_tn::CENTROID_TO_CLUSTER,
                       bbs_tn::TILE_OFFSETS,
                       bbs_tn::TILE_COUNTS,
                       bbs_tn::ALL_TILE_MT_M,
                       bbs_tn::ALL_TILE_MT_N,
                       bbs_tn::ALL_TILE_MT_K);
    build_layout_table(1,
                       bbs_nn::NUM_CLUSTERS,
                       bbs_nn::N_CENTROIDS,
                       bbs_nn::TOTAL_TILES,
                       bbs_nn::INTER_HIDDEN,
                       bbs_nn::W_B,
                       bbs_nn::CENTROID_TO_CLUSTER,
                       bbs_nn::TILE_OFFSETS,
                       bbs_nn::TILE_COUNTS,
                       bbs_nn::ALL_TILE_MT_M,
                       bbs_nn::ALL_TILE_MT_N,
                       bbs_nn::ALL_TILE_MT_K);
    build_layout_table(2,
                       sssmx_nt::NUM_CLUSTERS,
                       sssmx_nt::N_CENTROIDS,
                       sssmx_nt::TOTAL_TILES,
                       sssmx_nt::INTER_HIDDEN,
                       sssmx_nt::W_B,
                       sssmx_nt::CENTROID_TO_CLUSTER,
                       sssmx_nt::TILE_OFFSETS,
                       sssmx_nt::TILE_COUNTS,
                       sssmx_nt::ALL_TILE_MT_M,
                       sssmx_nt::ALL_TILE_MT_N,
                       sssmx_nt::ALL_TILE_MT_K);
    g_weights_loaded = true;
    return true;
  }
  return false;
}

}  // namespace ml_recommender
}  // namespace origami
