#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <vector>

// Orthogonal PopKV constants: only representative G_DIM is ablated.
constexpr int kTokenGroupSize = 32;
constexpr int kPackingWordBits = 32;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")
#define CHECK_FLOATISH(x) TORCH_CHECK((x).scalar_type() == at::kFloat || (x).scalar_type() == at::kHalf || (x).scalar_type() == at::kBFloat16, #x " must be fp32/fp16/bf16")

template <typename scalar_t>
__device__ inline float load_as_float(const scalar_t* ptr, int64_t idx) {
    return static_cast<float>(ptr[idx]);
}

template <>
__device__ inline float load_as_float<c10::Half>(const c10::Half* ptr, int64_t idx) {
    return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
}

template <>
__device__ inline float load_as_float<c10::BFloat16>(const c10::BFloat16* ptr, int64_t idx) {
    return static_cast<float>(ptr[idx]);
}

template <typename scalar_t>
__global__ void quest_update_kernel(
    const scalar_t* __restrict__ keys, float* __restrict__ page_min,
    float* __restrict__ page_max, int64_t tokens, int64_t head_dim,
    int64_t capacity_pages, int64_t page) {
    int64_t kvh = blockIdx.x;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float lo = load_as_float(keys, (kvh * tokens) * head_dim + d);
        float hi = lo;
        for (int64_t t = 1; t < tokens; ++t) {
            float value = load_as_float(keys, (kvh * tokens + t) * head_dim + d);
            lo = fminf(lo, value); hi = fmaxf(hi, value);
        }
        int64_t output = (kvh * capacity_pages + page) * head_dim + d;
        page_min[output] = lo; page_max[output] = hi;
    }
}

template <typename scalar_t>
__global__ void fier_update_kernel(
    const scalar_t* __restrict__ keys, int32_t* __restrict__ packed,
    scalar_t* __restrict__ group_min, scalar_t* __restrict__ group_max,
    int64_t tokens, int64_t head_dim, int64_t storage_words,
    int64_t storage_groups, int64_t word, int64_t group) {
    int64_t kvh = blockIdx.x;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float lo = load_as_float(keys, (kvh * tokens) * head_dim + d);
        float hi = lo;
        for (int64_t t = 1; t < tokens; ++t) {
            float value = load_as_float(keys, (kvh * tokens + t) * head_dim + d);
            lo = fminf(lo, value); hi = fmaxf(hi, value);
        }
        uint32_t bits = 0;
        scalar_t rounded_sum = static_cast<scalar_t>(lo + hi);
        float threshold = static_cast<float>(rounded_sum) * 0.5f;
        for (int64_t t = 0; t < tokens; ++t) {
            float value = load_as_float(keys, (kvh * tokens + t) * head_dim + d);
            if (value >= threshold) bits |= (uint32_t(1) << t);
        }
        packed[(kvh * head_dim + d) * storage_words + word] = static_cast<int32_t>(bits);
        int64_t meta = (kvh * storage_groups + group) * head_dim + d;
        group_min[meta] = static_cast<scalar_t>(lo);
        group_max[meta] = static_cast<scalar_t>(hi);
    }
}

template <typename scalar_t>
__global__ void popkv_pack_g32_kernel(
    const scalar_t* __restrict__ keys,
    int32_t* __restrict__ k_sign,
    int32_t* __restrict__ k_mag,
    __half* __restrict__ low_mean,
    __half* __restrict__ delta_mean,
    int64_t tokens,
    int64_t head_dim,
    int64_t words,
    int64_t num_groups,
    int64_t output_capacity,
    int64_t token_offset) {
    int lane = threadIdx.x;
    int64_t linear = blockIdx.x;
    int64_t word = linear % words;
    int64_t group = (linear / words) % num_groups;
    int64_t kvh = linear / (words * num_groups);
    int64_t d = word * 32 + lane;
    int64_t group_start = group * 32;
    int64_t group_end = min(group_start + 32, tokens);
    bool valid_dim = d < head_dim;
    float min_v = 0.0f, max_v = 0.0f;
    if (valid_dim && group_start < group_end) {
        min_v = max_v = load_as_float(keys, (kvh * tokens + group_start) * head_dim + d);
        for (int64_t t = group_start + 1; t < group_end; ++t) {
            float v = load_as_float(keys, (kvh * tokens + t) * head_dim + d);
            min_v = fminf(min_v, v); max_v = fmaxf(max_v, v);
        }
    }
    for (int64_t t = group_start; t < group_end; ++t) {
        float v = valid_dim ? load_as_float(keys, (kvh * tokens + t) * head_dim + d) : 0.0f;
        bool sign = valid_dim && v >= 0.0f;
        bool mag = valid_dim && (sign ? (v > max_v * 0.5f) : (v < min_v * 0.5f));
        uint32_t sign_word = __ballot_sync(0xffffffffu, sign);
        uint32_t mag_word = __ballot_sync(0xffffffffu, mag);
        float low_sum = (!mag && valid_dim) ? fabsf(v) : 0.0f;
        float high_sum = (mag && valid_dim) ? fabsf(v) : 0.0f;
        int low_count = (!mag && valid_dim) ? 1 : 0;
        int high_count = (mag && valid_dim) ? 1 : 0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            low_sum += __shfl_down_sync(0xffffffffu, low_sum, offset);
            high_sum += __shfl_down_sync(0xffffffffu, high_sum, offset);
            low_count += __shfl_down_sync(0xffffffffu, low_count, offset);
            high_count += __shfl_down_sync(0xffffffffu, high_count, offset);
        }
        if (lane == 0) {
            float low, high;
            if (low_count == 0) {
                low = high_count ? high_sum / high_count : 0.0f; high = low;
            } else {
                low = low_sum / low_count;
                high = high_count ? high_sum / high_count : low;
            }
            int64_t out = (kvh * output_capacity + token_offset + t) * words + word;
            k_sign[out] = static_cast<int32_t>(sign_word);
            k_mag[out] = static_cast<int32_t>(mag_word);
            low_mean[out] = __float2half_rn(low);
            delta_mean[out] = __float2half_rn(high - low);
        }
    }
}

template <typename scalar_t>
__global__ void popkv_score_kernel(
    const scalar_t* __restrict__ queries,
    const int32_t* __restrict__ k_sign,
    const int32_t* __restrict__ k_mag,
    const __half* __restrict__ k_low_mean,
    const __half* __restrict__ k_delta_mean,
    const int64_t* __restrict__ head_to_kv,
    float* __restrict__ scores,
    int64_t batch,
    int64_t q_heads,
    int64_t token_capacity,
    int64_t output_tokens,
    int64_t words,
    int64_t head_dim) {
    int64_t row = blockIdx.x;
    if (row >= batch * q_heads) return;
    int64_t qh = row % q_heads;
    int64_t kvh = head_to_kv[qh];
    extern __shared__ unsigned char shared_raw[];
    uint32_t* q_sign = reinterpret_cast<uint32_t*>(shared_raw);
    uint32_t* q_mag = q_sign + words;
    float* q_low = reinterpret_cast<float*>(q_mag + words);
    float* q_delta = q_low + words;
    if (threadIdx.x == 0) {
        int64_t q_base = row * head_dim;
        float min_v = load_as_float(queries, q_base), max_v = min_v;
        for (int64_t d = 1; d < head_dim; ++d) {
            float v = load_as_float(queries, q_base + d);
            min_v = fminf(min_v, v); max_v = fmaxf(max_v, v);
        }
        float neg = min_v * 0.5f, pos = max_v * 0.5f;
        for (int64_t w = 0; w < words; ++w) {
            uint32_t sw = 0, mw = 0;
            float low_sum = 0.0f, high_sum = 0.0f;
            int low_count = 0, high_count = 0;
            int64_t start = w * 32, end = min(start + 32, head_dim);
            for (int64_t d = start; d < end; ++d) {
                float v = load_as_float(queries, q_base + d);
                bool sign = v >= 0.0f;
                bool mag = sign ? (v > pos) : (v < neg);
                uint32_t bit = uint32_t(1) << uint32_t(d - start);
                if (sign) sw |= bit;
                if (mag) { mw |= bit; high_sum += fabsf(v); ++high_count; }
                else { low_sum += fabsf(v); ++low_count; }
            }
            float low, high;
            if (low_count == 0) {
                low = high_count ? high_sum / high_count : 0.0f; high = low;
            } else {
                low = low_sum / low_count;
                high = high_count ? high_sum / high_count : low;
            }
            q_sign[w] = sw; q_mag[w] = mw;
            q_low[w] = low; q_delta[w] = high - low;
        }
    }
    __syncthreads();
    float scale = rsqrtf(static_cast<float>(head_dim));
    for (int64_t token = threadIdx.x; token < output_tokens; token += blockDim.x) {
        int64_t base = (kvh * token_capacity + token) * words;
        float score = 0.0f;
        for (int64_t w = 0; w < words; ++w) {
            uint32_t valid_mask = 0xffffffffu;
            if (w == words - 1 && (head_dim & 31))
                valid_mask = (uint32_t(1) << uint32_t(head_dim & 31)) - 1u;
            uint32_t qm = q_mag[w] & valid_mask;
            uint32_t km = static_cast<uint32_t>(k_mag[base + w]) & valid_mask;
            uint32_t match = ~(q_sign[w] ^ static_cast<uint32_t>(k_sign[base + w])) & valid_mask;
            uint32_t both = qm & km;
            int c_sign = 2 * __popc(match) - __popc(valid_mask);
            int c_q = 2 * __popc(match & qm) - __popc(qm);
            int c_k = 2 * __popc(match & km) - __popc(km);
            int c_qk = 2 * __popc(match & both) - __popc(both);
            float ql = q_low[w], dq = q_delta[w];
            float kl = __half2float(k_low_mean[base + w]);
            float dk = __half2float(k_delta_mean[base + w]);
            float group_score = fmaf(ql * kl, static_cast<float>(c_sign),
                fmaf(dq * kl, static_cast<float>(c_q),
                fmaf(ql * dk, static_cast<float>(c_k),
                     dq * dk * static_cast<float>(c_qk))));
            score += group_score;
        }
        scores[row * output_tokens + token] = score * scale;
    }
}

// Benchmark/debug-only mirror of the query prelude in popkv_score_kernel.
// Keep the rules and empty-bucket behavior synchronized with that production
// kernel.  Production inference never calls this entry point.
template <typename scalar_t>
__global__ void popkv_query_encode_g32_kernel(
    const scalar_t* __restrict__ queries,
    int32_t* __restrict__ q_sign,
    int32_t* __restrict__ q_mag,
    float* __restrict__ q_low,
    float* __restrict__ q_delta,
    int64_t rows,
    int64_t words,
    int64_t head_dim) {
    int64_t row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    int64_t q_base = row * head_dim;
    float min_v = load_as_float(queries, q_base), max_v = min_v;
    for (int64_t d = 1; d < head_dim; ++d) {
        float v = load_as_float(queries, q_base + d);
        min_v = fminf(min_v, v); max_v = fmaxf(max_v, v);
    }
    float neg = min_v * 0.5f, pos = max_v * 0.5f;
    for (int64_t w = 0; w < words; ++w) {
        uint32_t sw = 0, mw = 0;
        float low_sum = 0.0f, high_sum = 0.0f;
        int low_count = 0, high_count = 0;
        int64_t start = w * 32, end = min(start + 32, head_dim);
        for (int64_t d = start; d < end; ++d) {
            float v = load_as_float(queries, q_base + d);
            bool sign = v >= 0.0f;
            bool mag = sign ? (v > pos) : (v < neg);
            uint32_t bit = uint32_t(1) << uint32_t(d - start);
            if (sign) sw |= bit;
            if (mag) { mw |= bit; high_sum += fabsf(v); ++high_count; }
            else { low_sum += fabsf(v); ++low_count; }
        }
        float low, high;
        if (low_count == 0) {
            low = high_count ? high_sum / high_count : 0.0f; high = low;
        } else {
            low = low_sum / low_count;
            high = high_count ? high_sum / high_count : low;
        }
        int64_t out = row * words + w;
        q_sign[out] = static_cast<int32_t>(sw);
        q_mag[out] = static_cast<int32_t>(mw);
        q_low[out] = low;
        q_delta[out] = high - low;
    }
}

// B_token and W_pack remain 32. Only representative grouping G_DIM varies.
template <typename scalar_t>
__global__ void popkv_pack_bits_kernel(
    const scalar_t* __restrict__ keys, int32_t* __restrict__ k_sign,
    int32_t* __restrict__ k_mag, int64_t tokens, int64_t head_dim,
    int64_t words, int64_t token_groups, int64_t output_capacity,
    int64_t token_offset) {
    int lane = threadIdx.x;
    int64_t linear = blockIdx.x;
    int64_t word = linear % words;
    int64_t token_group = (linear / words) % token_groups;
    int64_t kvh = linear / (words * token_groups);
    int64_t d = word * 32 + lane;
    int64_t group_start = token_group * 32;
    int64_t group_end = min(group_start + 32, tokens);
    bool valid_dim = d < head_dim;
    float min_v = 0.0f, max_v = 0.0f;
    if (valid_dim && group_start < group_end) {
        min_v = max_v = load_as_float(keys, (kvh * tokens + group_start) * head_dim + d);
        for (int64_t t = group_start + 1; t < group_end; ++t) {
            float v = load_as_float(keys, (kvh * tokens + t) * head_dim + d);
            min_v = fminf(min_v, v); max_v = fmaxf(max_v, v);
        }
    }
    for (int64_t t = group_start; t < group_end; ++t) {
        float v = valid_dim ? load_as_float(keys, (kvh * tokens + t) * head_dim + d) : 0.0f;
        bool sign = valid_dim && v >= 0.0f;
        bool mag = valid_dim && (sign ? (v > max_v * 0.5f) : (v < min_v * 0.5f));
        uint32_t sign_word = __ballot_sync(0xffffffffu, sign);
        uint32_t mag_word = __ballot_sync(0xffffffffu, mag);
        if (lane == 0) {
            int64_t out = (kvh * output_capacity + token_offset + t) * words + word;
            k_sign[out] = static_cast<int32_t>(sign_word);
            k_mag[out] = static_cast<int32_t>(mag_word);
        }
    }
}

template <typename scalar_t, int G_DIM>
__global__ void popkv_pack_representatives_kernel(
    const scalar_t* __restrict__ keys, __half* __restrict__ low_mean,
    __half* __restrict__ delta_mean, int64_t tokens, int64_t head_dim,
    int64_t rep_groups, int64_t token_groups, int64_t output_capacity,
    int64_t token_offset) {
    int lane = threadIdx.x;
    int64_t linear = blockIdx.x;
    int64_t rep = linear % rep_groups;
    int64_t token_group = (linear / rep_groups) % token_groups;
    int64_t kvh = linear / (rep_groups * token_groups);
    int64_t group_start = token_group * 32;
    int64_t group_end = min(group_start + 32, tokens);
    for (int64_t t = group_start; t < group_end; ++t) {
        float low_sum = 0.0f, high_sum = 0.0f;
        int low_count = 0, high_count = 0;
#pragma unroll
        for (int local = lane; local < G_DIM; local += 32) {
            int64_t d = rep * G_DIM + local;
            if (d < head_dim) {
                float min_v = load_as_float(keys, (kvh * tokens + group_start) * head_dim + d);
                float max_v = min_v;
                for (int64_t scan = group_start + 1; scan < group_end; ++scan) {
                    float candidate = load_as_float(keys, (kvh * tokens + scan) * head_dim + d);
                    min_v = fminf(min_v, candidate); max_v = fmaxf(max_v, candidate);
                }
                float v = load_as_float(keys, (kvh * tokens + t) * head_dim + d);
                bool sign = v >= 0.0f;
                bool mag = sign ? (v > max_v * 0.5f) : (v < min_v * 0.5f);
                if (mag) { high_sum += fabsf(v); ++high_count; }
                else { low_sum += fabsf(v); ++low_count; }
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            low_sum += __shfl_down_sync(0xffffffffu, low_sum, offset);
            high_sum += __shfl_down_sync(0xffffffffu, high_sum, offset);
            low_count += __shfl_down_sync(0xffffffffu, low_count, offset);
            high_count += __shfl_down_sync(0xffffffffu, high_count, offset);
        }
        if (lane == 0) {
            float low, high;
            if (low_count == 0) {
                low = high_count ? high_sum / high_count : 0.0f; high = low;
            } else {
                low = low_sum / low_count;
                high = high_count ? high_sum / high_count : low;
            }
            int64_t out = (kvh * output_capacity + token_offset + t) * rep_groups + rep;
            low_mean[out] = __float2half_rn(low);
            delta_mean[out] = __float2half_rn(high - low);
        }
    }
}

template <typename scalar_t, int G_DIM>
__global__ void popkv_score_dim_kernel(
    const scalar_t* __restrict__ queries, const int32_t* __restrict__ k_sign,
    const int32_t* __restrict__ k_mag, const __half* __restrict__ k_low,
    const __half* __restrict__ k_delta, const int64_t* __restrict__ head_to_kv,
    float* __restrict__ scores, int64_t batch, int64_t q_heads,
    int64_t token_capacity, int64_t output_tokens, int64_t words,
    int64_t rep_groups, int64_t head_dim) {
    int64_t row = blockIdx.x;
    if (row >= batch * q_heads) return;
    int64_t qh = row % q_heads, kvh = head_to_kv[qh];
    extern __shared__ unsigned char shared_raw[];
    uint32_t* q_sign = reinterpret_cast<uint32_t*>(shared_raw);
    uint32_t* q_mag = q_sign + words;
    float* q_low = reinterpret_cast<float*>(q_mag + words);
    float* q_delta = q_low + rep_groups;
    if (threadIdx.x == 0) {
        int64_t q_base = row * head_dim;
        float min_v = load_as_float(queries, q_base), max_v = min_v;
        for (int64_t d = 1; d < head_dim; ++d) {
            float v = load_as_float(queries, q_base + d);
            min_v = fminf(min_v, v); max_v = fmaxf(max_v, v);
        }
        float neg = min_v * 0.5f, pos = max_v * 0.5f;
        for (int64_t w = 0; w < words; ++w) {
            uint32_t sw = 0, mw = 0;
            int64_t start = w * 32, end = min(start + 32, head_dim);
            for (int64_t d = start; d < end; ++d) {
                float v = load_as_float(queries, q_base + d);
                bool sign = v >= 0.0f, mag = sign ? (v > pos) : (v < neg);
                uint32_t bit = uint32_t(1) << uint32_t(d - start);
                if (sign) sw |= bit; if (mag) mw |= bit;
            }
            q_sign[w] = sw; q_mag[w] = mw;
        }
        for (int64_t rep = 0; rep < rep_groups; ++rep) {
            float low_sum = 0.0f, high_sum = 0.0f;
            int low_count = 0, high_count = 0;
            int64_t start = rep * G_DIM, end = min(start + G_DIM, head_dim);
            for (int64_t d = start; d < end; ++d) {
                float v = load_as_float(queries, q_base + d);
                bool sign = v >= 0.0f, mag = sign ? (v > pos) : (v < neg);
                if (mag) { high_sum += fabsf(v); ++high_count; }
                else { low_sum += fabsf(v); ++low_count; }
            }
            float low, high;
            if (low_count == 0) {
                low = high_count ? high_sum / high_count : 0.0f; high = low;
            } else {
                low = low_sum / low_count; high = high_count ? high_sum / high_count : low;
            }
            q_low[rep] = low; q_delta[rep] = high - low;
        }
    }
    __syncthreads();
    float scale = rsqrtf(static_cast<float>(head_dim));
    for (int64_t token = threadIdx.x; token < output_tokens; token += blockDim.x) {
        int64_t bit_base = (kvh * token_capacity + token) * words;
        int64_t rep_base = (kvh * token_capacity + token) * rep_groups;
        float score = 0.0f;
        for (int64_t rep = 0; rep < rep_groups; ++rep) {
            int c_sign = 0, c_q = 0, c_k = 0, c_qk = 0;
            int64_t rep_start = rep * G_DIM, rep_end = min(rep_start + G_DIM, head_dim);
            int64_t first_word = rep_start / 32, last_word = (rep_end - 1) / 32;
            for (int64_t w = first_word; w <= last_word; ++w) {
                int64_t word_start = w * 32;
                int begin = static_cast<int>(max(rep_start, word_start) - word_start);
                int end = static_cast<int>(min(rep_end, word_start + 32) - word_start);
                int count = end - begin;
                uint32_t mask = count == 32 ? 0xffffffffu : ((uint32_t(1) << count) - 1u) << begin;
                uint32_t qm = q_mag[w] & mask;
                uint32_t km = static_cast<uint32_t>(k_mag[bit_base + w]) & mask;
                uint32_t match = ~(q_sign[w] ^ static_cast<uint32_t>(k_sign[bit_base + w])) & mask;
                uint32_t both = qm & km;
                c_sign += 2 * __popc(match) - __popc(mask);
                c_q += 2 * __popc(match & qm) - __popc(qm);
                c_k += 2 * __popc(match & km) - __popc(km);
                c_qk += 2 * __popc(match & both) - __popc(both);
            }
            float ql = q_low[rep], dq = q_delta[rep];
            float kl = __half2float(k_low[rep_base + rep]);
            float dk = __half2float(k_delta[rep_base + rep]);
            score += fmaf(ql * kl, static_cast<float>(c_sign),
                fmaf(dq * kl, static_cast<float>(c_q),
                fmaf(ql * dk, static_cast<float>(c_k), dq * dk * static_cast<float>(c_qk))));
        }
        scores[row * output_tokens + token] = score * scale;
    }
}

template <typename scalar_t, int G_DIM>
__global__ void popkv_query_encode_dim_kernel(
    const scalar_t* __restrict__ queries, int32_t* __restrict__ q_sign,
    int32_t* __restrict__ q_mag, float* __restrict__ q_low,
    float* __restrict__ q_delta, int64_t rows, int64_t words,
    int64_t rep_groups, int64_t head_dim) {
    int64_t row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    int64_t q_base = row * head_dim;
    float min_v = load_as_float(queries, q_base), max_v = min_v;
    for (int64_t d = 1; d < head_dim; ++d) {
        float v = load_as_float(queries, q_base + d);
        min_v = fminf(min_v, v); max_v = fmaxf(max_v, v);
    }
    float neg = min_v * 0.5f, pos = max_v * 0.5f;
    for (int64_t w = 0; w < words; ++w) {
        uint32_t sw = 0, mw = 0;
        int64_t start = w * 32, end = min(start + 32, head_dim);
        for (int64_t d = start; d < end; ++d) {
            float v = load_as_float(queries, q_base + d);
            bool sign = v >= 0.0f, mag = sign ? (v > pos) : (v < neg);
            uint32_t bit = uint32_t(1) << uint32_t(d - start);
            if (sign) sw |= bit; if (mag) mw |= bit;
        }
        q_sign[row * words + w] = static_cast<int32_t>(sw);
        q_mag[row * words + w] = static_cast<int32_t>(mw);
    }
    for (int64_t rep = 0; rep < rep_groups; ++rep) {
        float low_sum = 0.0f, high_sum = 0.0f;
        int low_count = 0, high_count = 0;
        int64_t start = rep * G_DIM, end = min(start + G_DIM, head_dim);
        for (int64_t d = start; d < end; ++d) {
            float v = load_as_float(queries, q_base + d);
            bool sign = v >= 0.0f, mag = sign ? (v > pos) : (v < neg);
            if (mag) { high_sum += fabsf(v); ++high_count; }
            else { low_sum += fabsf(v); ++low_count; }
        }
        float low, high;
        if (low_count == 0) {
            low = high_count ? high_sum / high_count : 0.0f; high = low;
        } else {
            low = low_sum / low_count; high = high_count ? high_sum / high_count : low;
        }
        q_low[row * rep_groups + rep] = low;
        q_delta[row * rep_groups + rep] = high - low;
    }
}

void quest_update_cuda(
    torch::Tensor keys, torch::Tensor page_min, torch::Tensor page_max, int64_t page) {
    CHECK_CUDA(keys); CHECK_CONTIGUOUS(keys); CHECK_FLOATISH(keys);
    CHECK_CUDA(page_min); CHECK_CUDA(page_max); CHECK_CONTIGUOUS(page_min); CHECK_CONTIGUOUS(page_max);
    TORCH_CHECK(keys.dim() == 3 && keys.size(1) > 0, "active Quest keys must be non-empty");
    TORCH_CHECK(page_min.scalar_type() == at::kFloat && page_max.scalar_type() == at::kFloat,
                "Quest page metadata must be float32");
    TORCH_CHECK(page_min.sizes() == page_max.sizes() && page_min.dim() == 3 &&
                page_min.size(0) == keys.size(0) && page_min.size(2) == keys.size(2),
                "Quest metadata shape mismatch");
    TORCH_CHECK(page >= 0 && page < page_min.size(1), "Quest page offset out of bounds");
    c10::cuda::CUDAGuard guard(keys.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, keys.scalar_type(), "quest_update_cuda", [&] {
        quest_update_kernel<scalar_t><<<keys.size(0), 256, 0, stream>>>(
            keys.data_ptr<scalar_t>(), page_min.data_ptr<float>(), page_max.data_ptr<float>(),
            keys.size(1), keys.size(2), page_min.size(1), page);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fier_update_cuda(
    torch::Tensor keys, torch::Tensor packed, torch::Tensor group_min,
    torch::Tensor group_max, int64_t token_offset) {
    CHECK_CUDA(keys); CHECK_CONTIGUOUS(keys); CHECK_FLOATISH(keys);
    CHECK_CUDA(packed); CHECK_CONTIGUOUS(packed); CHECK_INT32(packed);
    CHECK_CUDA(group_min); CHECK_CUDA(group_max); CHECK_CONTIGUOUS(group_min); CHECK_CONTIGUOUS(group_max);
    TORCH_CHECK(keys.dim() == 3 && keys.size(1) > 0 && keys.size(1) <= 32,
                "active FIER keys must contain 1..32 tokens");
    TORCH_CHECK(group_min.scalar_type() == keys.scalar_type() && group_max.scalar_type() == keys.scalar_type(),
                "FIER min/max dtype must match keys");
    TORCH_CHECK(packed.dim() == 3 && packed.size(0) == keys.size(0) && packed.size(1) == keys.size(2),
                "FIER packed shape mismatch");
    TORCH_CHECK(group_min.sizes() == group_max.sizes() && group_min.dim() == 3 &&
                group_min.size(0) == keys.size(0) && group_min.size(2) == keys.size(2),
                "FIER metadata shape mismatch");
    int64_t word = token_offset / 32, group = token_offset / 32;
    TORCH_CHECK(token_offset >= 0 && token_offset % 32 == 0 && word < packed.size(2) &&
                group < group_min.size(1), "FIER token offset out of bounds");
    c10::cuda::CUDAGuard guard(keys.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, keys.scalar_type(), "fier_update_cuda", [&] {
        fier_update_kernel<scalar_t><<<keys.size(0), 256, 0, stream>>>(
            keys.data_ptr<scalar_t>(), packed.data_ptr<int32_t>(),
            group_min.data_ptr<scalar_t>(), group_max.data_ptr<scalar_t>(),
            keys.size(1), keys.size(2), packed.size(2), group_min.size(1), word, group);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void launch_popkv_pack(
    torch::Tensor keys, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean, int64_t token_offset) {
    int64_t kv_heads = keys.size(0), tokens = keys.size(1), head_dim = keys.size(2);
    int64_t words = (head_dim + 31) / 32;
    int64_t capacity = k_sign.size(1), groups = (tokens + 31) / 32;
    int blocks = static_cast<int>(kv_heads * groups * words);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, keys.scalar_type(), "popkv_pack_cuda", [&] {
        popkv_pack_g32_kernel<scalar_t><<<blocks, 32, 0, stream>>>(
            keys.data_ptr<scalar_t>(), k_sign.data_ptr<int32_t>(), k_mag.data_ptr<int32_t>(),
            reinterpret_cast<__half*>(low_mean.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(delta_mean.data_ptr<at::Half>()),
            tokens, head_dim, words, groups, capacity, token_offset);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> popkv_pack_cuda(torch::Tensor keys, int64_t token_capacity) {
    CHECK_CUDA(keys); CHECK_CONTIGUOUS(keys); CHECK_FLOATISH(keys);
    TORCH_CHECK(keys.dim() == 3, "keys must be [KVH, T, D]");
    TORCH_CHECK(token_capacity >= keys.size(1), "token capacity is too small");
    c10::cuda::CUDAGuard guard(keys.device());
    int64_t words = (keys.size(2) + 31) / 32;
    auto shape = std::vector<int64_t>{keys.size(0), token_capacity, words};
    auto k_sign = torch::empty(shape, keys.options().dtype(torch::kInt32));
    auto k_mag = torch::empty(shape, keys.options().dtype(torch::kInt32));
    auto low_mean = torch::empty(shape, keys.options().dtype(torch::kFloat16));
    auto delta_mean = torch::empty(shape, keys.options().dtype(torch::kFloat16));
    launch_popkv_pack(keys, k_sign, k_mag, low_mean, delta_mean, 0);
    return {k_sign, k_mag, low_mean, delta_mean};
}

void popkv_pack_into_cuda(
    torch::Tensor keys, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean, int64_t token_offset) {
    CHECK_CUDA(keys); CHECK_CONTIGUOUS(keys); CHECK_FLOATISH(keys);
    CHECK_CUDA(k_sign); CHECK_CUDA(k_mag); CHECK_CUDA(low_mean); CHECK_CUDA(delta_mean);
    CHECK_CONTIGUOUS(k_sign); CHECK_CONTIGUOUS(k_mag);
    CHECK_CONTIGUOUS(low_mean); CHECK_CONTIGUOUS(delta_mean);
    CHECK_INT32(k_sign); CHECK_INT32(k_mag);
    TORCH_CHECK(low_mean.scalar_type() == at::kHalf && delta_mean.scalar_type() == at::kHalf,
                "Pop-KV representative metadata must be float16");
    TORCH_CHECK(token_offset >= 0 && token_offset + keys.size(1) <= k_sign.size(1),
                "Pop-KV cache capacity exceeded");
    c10::cuda::CUDAGuard guard(keys.device());
    launch_popkv_pack(keys, k_sign, k_mag, low_mean, delta_mean, token_offset);
}

static void launch_popkv_pack_dim(
    torch::Tensor keys, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean, int64_t token_offset,
    int64_t dim_group_size) {
    int64_t kv_heads = keys.size(0), tokens = keys.size(1), head_dim = keys.size(2);
    int64_t words = (head_dim + 31) / 32;
    int64_t rep_groups = (head_dim + dim_group_size - 1) / dim_group_size;
    int64_t token_groups = (tokens + 31) / 32, capacity = k_sign.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, keys.scalar_type(), "popkv_pack_dim_cuda", [&] {
        popkv_pack_bits_kernel<scalar_t><<<kv_heads * token_groups * words, 32, 0, stream>>>(
            keys.data_ptr<scalar_t>(), k_sign.data_ptr<int32_t>(), k_mag.data_ptr<int32_t>(),
            tokens, head_dim, words, token_groups, capacity, token_offset);
        if (dim_group_size == 16) {
            popkv_pack_representatives_kernel<scalar_t, 16><<<kv_heads * token_groups * rep_groups, 32, 0, stream>>>(
                keys.data_ptr<scalar_t>(), reinterpret_cast<__half*>(low_mean.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(delta_mean.data_ptr<at::Half>()), tokens, head_dim,
                rep_groups, token_groups, capacity, token_offset);
        } else {
            popkv_pack_representatives_kernel<scalar_t, 64><<<kv_heads * token_groups * rep_groups, 32, 0, stream>>>(
                keys.data_ptr<scalar_t>(), reinterpret_cast<__half*>(low_mean.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(delta_mean.data_ptr<at::Half>()), tokens, head_dim,
                rep_groups, token_groups, capacity, token_offset);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> popkv_pack_dim_cuda(
    torch::Tensor keys, int64_t token_capacity, int64_t dim_group_size) {
    CHECK_CUDA(keys); CHECK_CONTIGUOUS(keys); CHECK_FLOATISH(keys);
    TORCH_CHECK(keys.dim() == 3 && keys.size(1) > 0 && keys.size(2) > 0,
                "keys must be [KVH, non-empty tokens, non-empty head dim]");
    TORCH_CHECK(token_capacity >= keys.size(1), "token capacity is too small");
    TORCH_CHECK(dim_group_size == 16 || dim_group_size == 64,
                "specialized dim group must be 16 or 64");
    c10::cuda::CUDAGuard guard(keys.device());
    int64_t words = (keys.size(2) + 31) / 32;
    int64_t reps = (keys.size(2) + dim_group_size - 1) / dim_group_size;
    auto bit_shape = std::vector<int64_t>{keys.size(0), token_capacity, words};
    auto rep_shape = std::vector<int64_t>{keys.size(0), token_capacity, reps};
    auto k_sign = torch::empty(bit_shape, keys.options().dtype(torch::kInt32));
    auto k_mag = torch::empty(bit_shape, keys.options().dtype(torch::kInt32));
    auto low_mean = torch::empty(rep_shape, keys.options().dtype(torch::kFloat16));
    auto delta_mean = torch::empty(rep_shape, keys.options().dtype(torch::kFloat16));
    launch_popkv_pack_dim(keys, k_sign, k_mag, low_mean, delta_mean, 0, dim_group_size);
    return {k_sign, k_mag, low_mean, delta_mean};
}

void popkv_pack_into_dim_cuda(
    torch::Tensor keys, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean, int64_t token_offset,
    int64_t dim_group_size) {
    CHECK_CUDA(keys); CHECK_CONTIGUOUS(keys); CHECK_FLOATISH(keys);
    CHECK_CUDA(k_sign); CHECK_CUDA(k_mag); CHECK_CUDA(low_mean); CHECK_CUDA(delta_mean);
    CHECK_CONTIGUOUS(k_sign); CHECK_CONTIGUOUS(k_mag);
    CHECK_CONTIGUOUS(low_mean); CHECK_CONTIGUOUS(delta_mean);
    CHECK_INT32(k_sign); CHECK_INT32(k_mag);
    TORCH_CHECK(low_mean.scalar_type() == at::kHalf && delta_mean.scalar_type() == at::kHalf,
                "Pop-KV representative metadata must be float16");
    TORCH_CHECK(dim_group_size == 16 || dim_group_size == 64,
                "specialized dim group must be 16 or 64");
    TORCH_CHECK(keys.dim() == 3 && token_offset >= 0 &&
                token_offset + keys.size(1) <= k_sign.size(1), "Pop-KV cache capacity exceeded");
    int64_t words = (keys.size(2) + 31) / 32;
    int64_t reps = (keys.size(2) + dim_group_size - 1) / dim_group_size;
    TORCH_CHECK(k_sign.size(0) == keys.size(0) && k_sign.size(2) == words &&
                k_mag.sizes().vec() == k_sign.sizes().vec(), "packed bitplane shape mismatch");
    TORCH_CHECK(low_mean.size(0) == keys.size(0) && low_mean.size(1) == k_sign.size(1) &&
                low_mean.size(2) == reps && delta_mean.sizes().vec() == low_mean.sizes().vec(),
                "representative metadata shape mismatch");
    TORCH_CHECK(k_sign.device() == keys.device() && k_mag.device() == keys.device() &&
                low_mean.device() == keys.device() && delta_mean.device() == keys.device(),
                "Pop-KV packing tensors must share a CUDA device");
    c10::cuda::CUDAGuard guard(keys.device());
    launch_popkv_pack_dim(keys, k_sign, k_mag, low_mean, delta_mean, token_offset, dim_group_size);
}

torch::Tensor popkv_score_cuda(
    torch::Tensor queries, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean,
    torch::Tensor head_to_kv, int64_t tokens) {
    CHECK_CUDA(queries); CHECK_CONTIGUOUS(queries); CHECK_FLOATISH(queries);
    CHECK_CUDA(k_sign); CHECK_CUDA(k_mag); CHECK_CUDA(low_mean); CHECK_CUDA(delta_mean);
    CHECK_CUDA(head_to_kv); CHECK_CONTIGUOUS(head_to_kv);
    CHECK_INT32(k_sign); CHECK_INT32(k_mag);
    TORCH_CHECK(head_to_kv.scalar_type() == at::kLong, "head mapping must be int64");
    c10::cuda::CUDAGuard guard(queries.device());
    int64_t batch = queries.size(0), q_heads = queries.size(1), head_dim = queries.size(2);
    int64_t capacity = k_sign.size(1), words = k_sign.size(2);
    TORCH_CHECK(tokens > 0 && tokens <= capacity, "invalid score token count");
    auto scores = torch::empty({batch, q_heads, tokens}, queries.options().dtype(torch::kFloat32));
    size_t shared = static_cast<size_t>(4 * words) * sizeof(uint32_t);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, queries.scalar_type(), "popkv_score_cuda", [&] {
        popkv_score_kernel<scalar_t><<<batch * q_heads, 256, shared, at::cuda::getCurrentCUDAStream()>>>(
            queries.data_ptr<scalar_t>(), k_sign.data_ptr<int32_t>(), k_mag.data_ptr<int32_t>(),
            reinterpret_cast<const __half*>(low_mean.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(delta_mean.data_ptr<at::Half>()),
            head_to_kv.data_ptr<int64_t>(), scores.data_ptr<float>(),
            batch, q_heads, capacity, tokens, words, head_dim);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return scores;
}

torch::Tensor popkv_score_dim_cuda(
    torch::Tensor queries, torch::Tensor k_sign, torch::Tensor k_mag,
    torch::Tensor low_mean, torch::Tensor delta_mean,
    torch::Tensor head_to_kv, int64_t tokens, int64_t dim_group_size) {
    CHECK_CUDA(queries); CHECK_CONTIGUOUS(queries); CHECK_FLOATISH(queries);
    CHECK_CUDA(k_sign); CHECK_CUDA(k_mag); CHECK_CUDA(low_mean); CHECK_CUDA(delta_mean);
    CHECK_CUDA(head_to_kv); CHECK_CONTIGUOUS(head_to_kv);
    CHECK_INT32(k_sign); CHECK_INT32(k_mag);
    TORCH_CHECK(low_mean.scalar_type() == at::kHalf && delta_mean.scalar_type() == at::kHalf,
                "Pop-KV representative metadata must be float16");
    TORCH_CHECK(head_to_kv.scalar_type() == at::kLong, "head mapping must be int64");
    TORCH_CHECK(dim_group_size == 16 || dim_group_size == 64,
                "specialized dim group must be 16 or 64");
    c10::cuda::CUDAGuard guard(queries.device());
    int64_t batch = queries.size(0), q_heads = queries.size(1), head_dim = queries.size(2);
    int64_t capacity = k_sign.size(1), words = (head_dim + 31) / 32;
    int64_t reps = (head_dim + dim_group_size - 1) / dim_group_size;
    TORCH_CHECK(tokens > 0 && tokens <= capacity, "invalid score token count");
    TORCH_CHECK(k_sign.size(2) == words && k_mag.sizes().vec() == k_sign.sizes().vec(),
                "packed bitplane shape mismatch");
    TORCH_CHECK(low_mean.size(0) == k_sign.size(0) && low_mean.size(1) == capacity &&
                low_mean.size(2) == reps && delta_mean.sizes().vec() == low_mean.sizes().vec(),
                "representative metadata shape mismatch");
    TORCH_CHECK(head_to_kv.numel() == q_heads, "head mapping size mismatch");
    TORCH_CHECK(k_sign.device() == queries.device() && k_mag.device() == queries.device() &&
                low_mean.device() == queries.device() && delta_mean.device() == queries.device() &&
                head_to_kv.device() == queries.device(), "score tensors must share a CUDA device");
    auto scores = torch::empty({batch, q_heads, tokens}, queries.options().dtype(torch::kFloat32));
    size_t shared = static_cast<size_t>(2 * words + 2 * reps) * sizeof(uint32_t);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, queries.scalar_type(), "popkv_score_dim_cuda", [&] {
        if (dim_group_size == 16) {
            popkv_score_dim_kernel<scalar_t, 16><<<batch * q_heads, 256, shared, at::cuda::getCurrentCUDAStream()>>>(
                queries.data_ptr<scalar_t>(), k_sign.data_ptr<int32_t>(), k_mag.data_ptr<int32_t>(),
                reinterpret_cast<const __half*>(low_mean.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(delta_mean.data_ptr<at::Half>()), head_to_kv.data_ptr<int64_t>(),
                scores.data_ptr<float>(), batch, q_heads, capacity, tokens, words, reps, head_dim);
        } else {
            popkv_score_dim_kernel<scalar_t, 64><<<batch * q_heads, 256, shared, at::cuda::getCurrentCUDAStream()>>>(
                queries.data_ptr<scalar_t>(), k_sign.data_ptr<int32_t>(), k_mag.data_ptr<int32_t>(),
                reinterpret_cast<const __half*>(low_mean.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(delta_mean.data_ptr<at::Half>()), head_to_kv.data_ptr<int64_t>(),
                scores.data_ptr<float>(), batch, q_heads, capacity, tokens, words, reps, head_dim);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return scores;
}

void popkv_query_encode_out_cuda(
    torch::Tensor queries, torch::Tensor q_sign, torch::Tensor q_mag,
    torch::Tensor q_low, torch::Tensor q_delta, int64_t dim_group_size) {
    CHECK_CUDA(queries); CHECK_CONTIGUOUS(queries); CHECK_FLOATISH(queries);
    CHECK_CUDA(q_sign); CHECK_CUDA(q_mag); CHECK_CUDA(q_low); CHECK_CUDA(q_delta);
    CHECK_CONTIGUOUS(q_sign); CHECK_CONTIGUOUS(q_mag);
    CHECK_CONTIGUOUS(q_low); CHECK_CONTIGUOUS(q_delta);
    CHECK_INT32(q_sign); CHECK_INT32(q_mag);
    TORCH_CHECK(q_low.scalar_type() == at::kFloat && q_delta.scalar_type() == at::kFloat,
                "query low/delta outputs must be float32");
    TORCH_CHECK(queries.dim() == 3 && queries.size(2) > 0,
                "queries must be [batch, query heads, non-empty head dim]");
    c10::cuda::CUDAGuard guard(queries.device());
    int64_t batch = queries.size(0), q_heads = queries.size(1), head_dim = queries.size(2);
    TORCH_CHECK(dim_group_size == 16 || dim_group_size == 32 || dim_group_size == 64,
                "dim_group_size must be 16, 32, or 64");
    int64_t words = (head_dim + 31) / 32;
    int64_t reps = (head_dim + dim_group_size - 1) / dim_group_size;
    auto bit_expected = std::vector<int64_t>{batch, q_heads, words};
    auto rep_expected = std::vector<int64_t>{batch, q_heads, reps};
    TORCH_CHECK(q_sign.sizes().vec() == bit_expected && q_mag.sizes().vec() == bit_expected &&
                q_low.sizes().vec() == rep_expected && q_delta.sizes().vec() == rep_expected,
                "query encoding outputs have incorrect shape");
    TORCH_CHECK(q_sign.device() == queries.device() && q_mag.device() == queries.device() &&
                q_low.device() == queries.device() && q_delta.device() == queries.device(),
                "query encoding tensors must share a CUDA device");
    int64_t rows = batch * q_heads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, queries.scalar_type(), "popkv_query_encode_cuda", [&] {
        if (dim_group_size == 32) {
            popkv_query_encode_g32_kernel<scalar_t><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                queries.data_ptr<scalar_t>(), q_sign.data_ptr<int32_t>(), q_mag.data_ptr<int32_t>(),
                q_low.data_ptr<float>(), q_delta.data_ptr<float>(), rows, words, head_dim);
        } else if (dim_group_size == 16) {
            popkv_query_encode_dim_kernel<scalar_t, 16><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                queries.data_ptr<scalar_t>(), q_sign.data_ptr<int32_t>(), q_mag.data_ptr<int32_t>(),
                q_low.data_ptr<float>(), q_delta.data_ptr<float>(), rows, words, reps, head_dim);
        } else {
            popkv_query_encode_dim_kernel<scalar_t, 64><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                queries.data_ptr<scalar_t>(), q_sign.data_ptr<int32_t>(), q_mag.data_ptr<int32_t>(),
                q_low.data_ptr<float>(), q_delta.data_ptr<float>(), rows, words, reps, head_dim);
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
