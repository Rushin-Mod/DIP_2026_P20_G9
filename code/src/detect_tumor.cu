/*
 * detect_tumor.cu
 *
 * 3D brain MRI tumor detection engine.
 * Computes statistical Z-score against baseline atlas, evaluates contralateral
 * asymmetry across the mid-sagittal plane, and performs seeded region growing.
 */

#include <windows.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#pragma pack(push, 1)
struct VolumeHeader {
    uint16_t dim_z;
    uint16_t dim_y;
    uint16_t dim_x;
    float vz;
    float vy;
    float vx;
};
#pragma pack(pop)

static_assert(sizeof(VolumeHeader) == 18, "VolumeHeader must be strictly 18 bytes packed!");

#if defined(USE_UINT16)
    typedef uint16_t voxel_t;
    #define DTYPE_NAME "uint16"
    #define ZERO_VAL ((uint16_t)0)
    #define MAX_VAL ((float)65535.0f)
#else
    typedef float voxel_t;
    #define DTYPE_NAME "float32"
    #define ZERO_VAL (0.0f)
    #define MAX_VAL (1e9f)
#endif

enum FilterMode {
    FILTER_LAPLACIAN      = 1,
    FILTER_GAUSSIAN       = 2,
    FILTER_MEDIAN         = 3,
    FILTER_MEAN           = 4,
    FILTER_MORPH_GRADIENT = 5
};

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

bool CreateDirectoryRecursive(const std::string& path) {
    if (path.empty()) return false;
    char temp[MAX_PATH];
    std::snprintf(temp, sizeof(temp), "%s", path.c_str());
    size_t len = std::strlen(temp);
    if (len == 0) return false;
    if (temp[len - 1] == '\\' || temp[len - 1] == '/') {
        temp[len - 1] = '\0';
    }
    for (char* p = temp + 1; *p; p++) {
        if (*p == '\\' || *p == '/') {
            char orig = *p;
            *p = '\0';
            CreateDirectoryA(temp, NULL);
            *p = orig;
        }
    }
    return (CreateDirectoryA(temp, NULL) != 0 || GetLastError() == ERROR_ALREADY_EXISTS);
}

std::vector<std::string> ReadManifest(const std::string& manifest_path) {
    std::vector<std::string> file_paths;
    std::ifstream infile(manifest_path.c_str());
    if (!infile.is_open()) {
        return file_paths;
    }
    std::string line;
    bool first_line = true;
    while (std::getline(infile, line)) {
        if (first_line) {
            first_line = false;
            if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF) {
                line = line.substr(3);
            }
        }
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        size_t start = line.find_first_not_of(" \t");
        if (start != std::string::npos && line[start] != '#') {
            file_paths.push_back(line.substr(start));
        }
    }
    return file_paths;
}

// ----------------------------------------------------------------------------
// CUDA Kernels: Statistical Baseline Atlas Generation
// ----------------------------------------------------------------------------

__global__ void ComputeBaselineAtlasKernel(const voxel_t* const* __restrict__ baseline_vols,
                                           size_t num_baselines,
                                           float* __restrict__ atlas_mean,
                                           float* __restrict__ atlas_std,
                                           size_t num_voxels) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_voxels) return;

    if (num_baselines == 0) {
        atlas_mean[idx] = 0.0f;
        atlas_std[idx] = 0.0f;
        return;
    }

    double sum = 0.0;
    for (size_t b = 0; b < num_baselines; ++b) {
        sum += (double)baseline_vols[b][idx];
    }
    double mean = sum / (double)num_baselines;

    double sq_diff_sum = 0.0;
    if (num_baselines > 1) {
        for (size_t b = 0; b < num_baselines; ++b) {
            double diff = (double)baseline_vols[b][idx] - mean;
            sq_diff_sum += diff * diff;
        }
    }
    double var = (num_baselines > 1) ? (sq_diff_sum / (double)(num_baselines - 1)) : 0.0;
    atlas_mean[idx] = (float)mean;
    atlas_std[idx] = (float)std::sqrt(var);
}

// ----------------------------------------------------------------------------
// CUDA Kernels: 3D Spatial Gaussian Smoothing (Binomial [1, 2, 1] per axis)
// ----------------------------------------------------------------------------

__global__ void GaussianSmooth3DKernel(const float* __restrict__ in_vol,
                                       float* __restrict__ out_vol,
                                       uint16_t dim_z, uint16_t dim_y, uint16_t dim_x) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= dim_x || y >= dim_y || z >= dim_z) return;

    size_t idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + x;

    float sum = 0.0f;
    float total_weight = 0.0f;

    for (int dz = -1; dz <= 1; ++dz) {
        int cz = z + dz;
        if (cz < 0 || cz >= dim_z) continue;
        float wz = (dz == 0) ? 2.0f : 1.0f;

        for (int dy = -1; dy <= 1; ++dy) {
            int cy = y + dy;
            if (cy < 0 || cy >= dim_y) continue;
            float wy = (dy == 0) ? 2.0f : 1.0f;

            for (int dx = -1; dx <= 1; ++dx) {
                int cx = x + dx;
                if (cx < 0 || cx >= dim_x) continue;
                float wx = (dx == 0) ? 2.0f : 1.0f;

                float weight = wz * wy * wx;
                size_t n_idx = (size_t)cz * dim_y * dim_x + (size_t)cy * dim_x + cx;
                sum += in_vol[n_idx] * weight;
                total_weight += weight;
            }
        }
    }

    out_vol[idx] = (total_weight > 0.0f) ? (sum / total_weight) : 0.0f;
}

// ----------------------------------------------------------------------------
// CUDA Kernels: Multi-Feature Saliency & Contralateral Disambiguation
// ----------------------------------------------------------------------------

__global__ void ComputeRawAsymAndZScoreKernel(const voxel_t* __restrict__ test_vol,
                                              const float* __restrict__ atlas_mean,
                                              const float* __restrict__ atlas_std,
                                              float* __restrict__ out_asym_raw,
                                              float* __restrict__ out_z_score,
                                              uint16_t dim_z, uint16_t dim_y, uint16_t dim_x) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= dim_x || y >= dim_y || z >= dim_z) return;

    size_t idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + x;
    float test_val = (float)test_vol[idx];

    if (test_val <= 0.0f) {
        out_asym_raw[idx] = 0.0f;
        out_z_score[idx] = 0.0f;
        return;
    }

    // Contralateral mirror coordinate across MSP
    int x_mirror = (int)dim_x - 1 - x;
    size_t mirror_idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + x_mirror;
    float mirror_val = (float)test_vol[mirror_idx];

    // Contralateral absolute difference
    float asym = (mirror_val > 0.0f) ? fabsf(test_val - mirror_val) : 0.0f;
    out_asym_raw[idx] = asym;

    // Standardized Z-Score against baseline atlas with regularized variance floor
    float b_m = atlas_mean[idx];
    float b_s = atlas_std[idx] + 1200.0f; // variance regularization floor
    float z_sc = fabsf(test_val - b_m) / b_s;
    out_z_score[idx] = z_sc;
}

__global__ void ComputeFusedSaliencyKernel(const voxel_t* __restrict__ test_vol,
                                           const float* __restrict__ asym_smooth,
                                           const float* __restrict__ z_score_smooth,
                                           float* __restrict__ out_saliency,
                                           uint16_t dim_z, uint16_t dim_y, uint16_t dim_x) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= dim_x || y >= dim_y || z >= dim_z) return;

    size_t idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + x;
    float test_val = (float)test_vol[idx];

    if (test_val <= 0.0f) {
        out_saliency[idx] = 0.0f;
        return;
    }

    // Contralateral Disambiguation:
    // Symmetry difference creates identical twins at x and x_mirror.
    // Compare local baseline anomaly Z(x) vs Z(x_mirror) to identify true lesion side.
    int x_mirror = (int)dim_x - 1 - x;
    size_t mirror_idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + x_mirror;

    float z_local = z_score_smooth[idx];
    float z_mirror = z_score_smooth[mirror_idx];

    // If local anomaly is substantially lower than mirror side, suppress phantom reflection
    if (z_local < (z_mirror - 0.20f)) {
        out_saliency[idx] = 0.0f;
        return;
    }

    float asym_s = asym_smooth[idx];
    float sal = asym_s * (1.0f + 0.50f * z_local);

    // Suppress lower skull base / neck fringe below vertex-caudal cutoff (Z < 45)
    if (z < 45) {
        sal *= 0.10f;
    }

    out_saliency[idx] = sal;
}

// ----------------------------------------------------------------------------
// CUDA Parallel Reduction: Moments and Maxima
// ----------------------------------------------------------------------------

__global__ void ReduceStatsKernel(const float* __restrict__ map_in,
                                  float* __restrict__ block_maxs,
                                  double* __restrict__ block_sums,
                                  double* __restrict__ block_sq_sums,
                                  uint64_t* __restrict__ block_counts,
                                  size_t num_voxels) {
    extern __shared__ double s_data[];
    float* s_max = (float*)s_data;
    double* s_sum = (double*)&s_data[blockDim.x / 2 + 1];
    double* s_sq_sum = &s_sum[blockDim.x];
    uint64_t* s_count = (uint64_t*)&s_sq_sum[blockDim.x];

    unsigned int tid = threadIdx.x;
    size_t gid = (size_t)blockDim.x * blockIdx.x + threadIdx.x;

    float my_max = 0.0f;
    double my_sum = 0.0;
    double my_sq_sum = 0.0;
    uint64_t my_count = 0;

    if (gid < num_voxels) {
        float val = map_in[gid];
        if (val > 0.0f) {
            my_max = val;
            double dval = (double)val;
            my_sum = dval;
            my_sq_sum = dval * dval;
            my_count = 1;
        }
    }

    s_max[tid] = my_max;
    s_sum[tid] = my_sum;
    s_sq_sum[tid] = my_sq_sum;
    s_count[tid] = my_count;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
            s_sum[tid] += s_sum[tid + s];
            s_sq_sum[tid] += s_sq_sum[tid + s];
            s_count[tid] += s_count[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_maxs[blockIdx.x] = s_max[0];
        block_sums[blockIdx.x] = s_sum[0];
        block_sq_sums[blockIdx.x] = s_sq_sum[0];
        block_counts[blockIdx.x] = s_count[0];
    }
}

// ----------------------------------------------------------------------------
// CUDA Kernels: Adaptive 3D Dual-Threshold Hysteresis & Geodesic Region Growing
// ----------------------------------------------------------------------------

__global__ void InitHysteresisKernel(const float* __restrict__ saliency_map,
                                     float t_high,
                                     float t_low,
                                     uint8_t* __restrict__ seeds_out,
                                     uint8_t* __restrict__ cands_out,
                                     size_t num_voxels) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_voxels) return;

    float s = saliency_map[idx];
    seeds_out[idx] = (s >= t_high) ? 1 : 0;
    cands_out[idx] = (s >= t_low) ? 1 : 0;
}

__global__ void GeodesicGrow26Kernel(const uint8_t* __restrict__ cands,
                                     const uint8_t* __restrict__ current_seeds,
                                     uint8_t* __restrict__ next_seeds,
                                     int* __restrict__ d_changed,
                                     uint16_t dim_z, uint16_t dim_y, uint16_t dim_x) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= dim_x || y >= dim_y || z >= dim_z) return;

    size_t idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + x;

    // If already seeded, persist state
    if (current_seeds[idx]) {
        next_seeds[idx] = 1;
        return;
    }

    // If not a candidate, cannot grow
    if (!cands[idx]) {
        next_seeds[idx] = 0;
        return;
    }

    // Check 26-neighborhood for active seed
    bool touches_seed = false;
    for (int dz = -1; dz <= 1 && !touches_seed; ++dz) {
        int cz = z + dz;
        if (cz < 0 || cz >= dim_z) continue;
        for (int dy = -1; dy <= 1 && !touches_seed; ++dy) {
            int cy = y + dy;
            if (cy < 0 || cy >= dim_y) continue;
            for (int dx = -1; dx <= 1 && !touches_seed; ++dx) {
                int cx = x + dx;
                if (cx < 0 || cx >= dim_x) continue;
                if (dx == 0 && dy == 0 && dz == 0) continue;

                size_t n_idx = (size_t)cz * dim_y * dim_x + (size_t)cy * dim_x + cx;
                if (current_seeds[n_idx]) {
                    touches_seed = true;
                }
            }
        }
    }

    if (touches_seed) {
        next_seeds[idx] = 1;
        atomicExch(d_changed, 1);
    } else {
        next_seeds[idx] = 0;
    }
}

// ----------------------------------------------------------------------------
// CUDA Kernels: 3D Morphological Consolidation (Closing & Opening)
// ----------------------------------------------------------------------------

__global__ void Dilate6Kernel(const uint8_t* __restrict__ in_mask,
                              uint8_t* __restrict__ out_mask,
                              uint16_t dim_z, uint16_t dim_y, uint16_t dim_x) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= dim_x || y >= dim_y || z >= dim_z) return;

    size_t idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + x;

    if (in_mask[idx]) {
        out_mask[idx] = 1;
        return;
    }

    bool hit = (x > 0 && in_mask[idx - 1]) ||
               (x < dim_x - 1 && in_mask[idx + 1]) ||
               (y > 0 && in_mask[idx - dim_x]) ||
               (y < dim_y - 1 && in_mask[idx + dim_x]) ||
               (z > 0 && in_mask[idx - (size_t)dim_y * dim_x]) ||
               (z < dim_z - 1 && in_mask[idx + (size_t)dim_y * dim_x]);

    out_mask[idx] = hit ? 1 : 0;
}

__global__ void Erode6Kernel(const uint8_t* __restrict__ in_mask,
                             uint8_t* __restrict__ out_mask,
                             uint16_t dim_z, uint16_t dim_y, uint16_t dim_x) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= dim_x || y >= dim_y || z >= dim_z) return;

    size_t idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + x;

    if (!in_mask[idx]) {
        out_mask[idx] = 0;
        return;
    }

    bool all_in = (x > 0 ? in_mask[idx - 1] : 0) &&
                  (x < dim_x - 1 ? in_mask[idx + 1] : 0) &&
                  (y > 0 ? in_mask[idx - dim_x] : 0) &&
                  (y < dim_y - 1 ? in_mask[idx + dim_x] : 0) &&
                  (z > 0 ? in_mask[idx - (size_t)dim_y * dim_x] : 0) &&
                  (z < dim_z - 1 ? in_mask[idx + (size_t)dim_y * dim_x] : 0);

    out_mask[idx] = all_in ? 1 : 0;
}

__global__ void ApplyMaskToOutputKernel(const voxel_t* __restrict__ test_vol,
                                        const uint8_t* __restrict__ tumor_mask,
                                        voxel_t* __restrict__ out_subtracted,
                                        size_t num_voxels) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_voxels) return;

    out_subtracted[idx] = tumor_mask[idx] ? test_vol[idx] : ZERO_VAL;
}

// ----------------------------------------------------------------------------
// Host Statistical Computation & Primary Cluster Selection Helper
// ----------------------------------------------------------------------------

struct ClusterCandidate {
    size_t size;
    double score;
    std::vector<size_t> indices;
};

void FilterPrimaryTumorClusters(voxel_t* vol, const float* saliency, uint16_t dim_z, uint16_t dim_y, uint16_t dim_x) {
    size_t num_voxels = (size_t)dim_z * dim_y * dim_x;
    std::vector<uint8_t> visited(num_voxels, 0);
    std::vector<ClusterCandidate> clusters;

    for (size_t idx = 0; idx < num_voxels; ++idx) {
        if (vol[idx] == ZERO_VAL || visited[idx]) continue;

        std::vector<size_t> queue;
        queue.reserve(8192);
        queue.push_back(idx);
        visited[idx] = 1;

        size_t head = 0;
        int min_x = dim_x, max_x = 0;
        int min_y = dim_y, max_y = 0;
        int min_z = dim_z, max_z = 0;
        double sal_sum = 0.0;

        while (head < queue.size()) {
            size_t cur = queue[head++];
            int cz = (int)(cur / (dim_y * dim_x));
            size_t rem = cur % (dim_y * dim_x);
            int cy = (int)(rem / dim_x);
            int cx = (int)(rem % dim_x);

            if (cx < min_x) min_x = cx;
            if (cx > max_x) max_x = cx;
            if (cy < min_y) min_y = cy;
            if (cy > max_y) max_y = cy;
            if (cz < min_z) min_z = cz;
            if (cz > max_z) max_z = cz;

            sal_sum += (double)saliency[cur];

            for (int dz = -1; dz <= 1; ++dz) {
                int nz = cz + dz;
                if (nz < 0 || nz >= dim_z) continue;
                for (int dy = -1; dy <= 1; ++dy) {
                    int ny = cy + dy;
                    if (ny < 0 || ny >= dim_y) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = cx + dx;
                        if (nx < 0 || nx >= dim_x) continue;
                        if (dz == 0 && dy == 0 && dx == 0) continue;

                        size_t n_idx = (size_t)nz * dim_y * dim_x + (size_t)ny * dim_x + nx;
                        if (vol[n_idx] != ZERO_VAL && !visited[n_idx]) {
                            visited[n_idx] = 1;
                            queue.push_back(n_idx);
                        }
                    }
                }
            }
        }

        size_t cluster_size = queue.size();
        if (cluster_size >= 100) { // cluster >= 0.1 mL
            double box_vol = (double)(max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1);
            double compactness = (box_vol > 0.0) ? ((double)cluster_size / box_vol) : 0.0;
            double mean_sal = sal_sum / (double)cluster_size;
            double score = (double)cluster_size * mean_sal * std::sqrt(compactness);

            ClusterCandidate c;
            c.size = cluster_size;
            c.score = score;
            c.indices = std::move(queue);
            clusters.push_back(std::move(c));
        } else {
            // Prune tiny single-voxel noise immediately
            for (size_t v : queue) vol[v] = ZERO_VAL;
        }
    }

    if (clusters.empty()) return;

    // Sort by saliency & compactness score descending
    std::sort(clusters.begin(), clusters.end(), [](const ClusterCandidate& a, const ClusterCandidate& b) {
        return a.score > b.score;
    });

    double top_score = clusters[0].score;
    // Discard secondary clusters below 35% of top score
    for (size_t k = 1; k < clusters.size(); ++k) {
        if (clusters[k].score < 0.35 * top_score) {
            for (size_t v : clusters[k].indices) {
                vol[v] = ZERO_VAL;
            }
        }
    }
}

void ComputeHostMoments(const float* d_map, size_t num_voxels, float& max_val, float& mean_val, float& std_val) {
    int redBlockSize = 256;
    int redNumBlocks = (int)((num_voxels + redBlockSize - 1) / redBlockSize);

    float* d_maxs = nullptr;
    double* d_sums = nullptr;
    double* d_sq_sums = nullptr;
    uint64_t* d_counts = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_maxs, redNumBlocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_sums, redNumBlocks * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_sq_sums, redNumBlocks * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_counts, redNumBlocks * sizeof(uint64_t)));

    size_t smem = 3 * redBlockSize * sizeof(double) + redBlockSize * sizeof(uint64_t) + 64;
    ReduceStatsKernel<<<redNumBlocks, redBlockSize, smem>>>(d_map, d_maxs, d_sums, d_sq_sums, d_counts, num_voxels);
    CHECK_CUDA(cudaGetLastError());

    std::vector<float> h_maxs(redNumBlocks);
    std::vector<double> h_sums(redNumBlocks);
    std::vector<double> h_sq_sums(redNumBlocks);
    std::vector<uint64_t> h_counts(redNumBlocks);

    CHECK_CUDA(cudaMemcpy(h_maxs.data(), d_maxs, redNumBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sums.data(), d_sums, redNumBlocks * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sq_sums.data(), d_sq_sums, redNumBlocks * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_counts.data(), d_counts, redNumBlocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    cudaFree(d_maxs);
    cudaFree(d_sums);
    cudaFree(d_sq_sums);
    cudaFree(d_counts);

    float global_max = 0.0f;
    double total_sum = 0.0;
    double total_sq_sum = 0.0;
    uint64_t total_count = 0;

    for (int i = 0; i < redNumBlocks; ++i) {
        if (h_maxs[i] > global_max) global_max = h_maxs[i];
        total_sum += h_sums[i];
        total_sq_sum += h_sq_sums[i];
        total_count += h_counts[i];
    }

    max_val = global_max;
    if (total_count == 0) {
        mean_val = 0.0f;
        std_val = 0.0f;
        return;
    }

    double d_mean = total_sum / (double)total_count;
    double var = (total_sq_sum / (double)total_count) - (d_mean * d_mean);
    if (var < 0.0) var = 0.0;

    mean_val = (float)d_mean;
    std_val = (float)std::sqrt(var);
}

// ----------------------------------------------------------------------------
// Main Application Driver
// ----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string test_manifest_path = "";
    std::string baseline_manifest_path = "";
    std::string diff_manifest_path = "";
    std::string output_dir = "";
    int chosen_filter_mode = FILTER_LAPLACIAN;

    if (argc >= 5) {
        test_manifest_path = argv[1];
        baseline_manifest_path = argv[2];
        diff_manifest_path = argv[3];
        output_dir = argv[4];
        if (argc >= 6) chosen_filter_mode = std::atoi(argv[5]);
    } else if (argc == 4) {
        test_manifest_path = argv[1];
        baseline_manifest_path = argv[2];
        diff_manifest_path = "auto";
        output_dir = argv[3];
    } else {
        std::cout << "Usage: detect_tumor <test_manifest> <baseline_manifest> [diff_manifest|auto] <output_dir> [filter_id(1-5)]" << std::endl;
        return EXIT_FAILURE;
    }

    if (!CreateDirectoryRecursive(output_dir)) {
        std::fprintf(stderr, "Warning: Failed or existing directory: %s\n", output_dir.c_str());
    }

    std::vector<std::string> test_files = ReadManifest(test_manifest_path);
    std::vector<std::string> baseline_files = ReadManifest(baseline_manifest_path);

    size_t num_tests = test_files.size();
    size_t num_baselines = baseline_files.size();

    if (num_tests == 0 || num_baselines == 0) {
        std::fprintf(stderr, "Error: Manifests contain no valid test or baseline files!\n");
        return EXIT_FAILURE;
    }

    // Read reference header from first baseline
    VolumeHeader ref_hdr;
    FILE* fp_ref = std::fopen(baseline_files[0].c_str(), "rb");
    if (!fp_ref || std::fread(&ref_hdr, 1, sizeof(VolumeHeader), fp_ref) != sizeof(VolumeHeader)) {
        std::fprintf(stderr, "Error: Could not read header from %s\n", baseline_files[0].c_str());
        if (fp_ref) std::fclose(fp_ref);
        return EXIT_FAILURE;
    }
    std::fclose(fp_ref);

    size_t num_voxels = (size_t)ref_hdr.dim_z * ref_hdr.dim_y * ref_hdr.dim_x;
    size_t payload_bytes = num_voxels * sizeof(voxel_t);

    std::printf("===================================================================\n");
    std::printf("   3D CLASSICAL BRAIN MRI TUMOR DETECTION ENGINE (new_code)        \n");
    std::printf("===================================================================\n");
    std::printf(" Data Type: %s | Matrix: %u x %u x %u (%zu voxels)\n",
                DTYPE_NAME, ref_hdr.dim_z, ref_hdr.dim_y, ref_hdr.dim_x, num_voxels);
    std::printf(" Voxel Spacing: [%.2f, %.2f, %.2f] mm\n", ref_hdr.vz, ref_hdr.vy, ref_hdr.vx);
    std::printf(" Baseline Count: %zu | Test Count: %zu\n", num_baselines, num_tests);

    // Reusable pinned host staging buffer
    voxel_t* h_pinned_staging = nullptr;
    CHECK_CUDA(cudaHostAlloc((void**)&h_pinned_staging, payload_bytes, cudaHostAllocDefault));

    // Allocate and stream baseline volumes into VRAM
    std::vector<voxel_t*> d_baselines(num_baselines, nullptr);
    for (size_t b = 0; b < num_baselines; ++b) {
        CHECK_CUDA(cudaMalloc((void**)&d_baselines[b], payload_bytes));
        FILE* fp = std::fopen(baseline_files[b].c_str(), "rb");
        VolumeHeader cur_hdr;
        if (!fp || std::fread(&cur_hdr, 1, sizeof(VolumeHeader), fp) != sizeof(VolumeHeader) ||
            std::fread(h_pinned_staging, sizeof(voxel_t), num_voxels, fp) != num_voxels) {
            std::fprintf(stderr, "Error reading baseline: %s\n", baseline_files[b].c_str());
            if (fp) std::fclose(fp);
            return EXIT_FAILURE;
        }
        std::fclose(fp);
        CHECK_CUDA(cudaMemcpy(d_baselines[b], h_pinned_staging, payload_bytes, cudaMemcpyHostToDevice));
    }
    std::printf("[VRAM] Loaded %zu baseline volumes into GPU memory (%.2f MB)\n",
                num_baselines, (double)(num_baselines * payload_bytes) / (1024.0 * 1024.0));

    // Pointer table for baseline volumes
    voxel_t** d_baseline_ptrs = nullptr;
    CHECK_CUDA(cudaMalloc((void***)&d_baseline_ptrs, num_baselines * sizeof(voxel_t*)));
    CHECK_CUDA(cudaMemcpy(d_baseline_ptrs, d_baselines.data(), num_baselines * sizeof(voxel_t*), cudaMemcpyHostToDevice));

    // Device float buffers for Statistical Atlas & Filtering
    float* d_atlas_mean = nullptr;
    float* d_atlas_std = nullptr;
    float* d_asym_raw = nullptr;
    float* d_asym_smooth = nullptr;
    float* d_z_score_raw = nullptr;
    float* d_z_score_smooth = nullptr;
    float* d_saliency = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_atlas_mean, num_voxels * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_atlas_std, num_voxels * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_asym_raw, num_voxels * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_asym_smooth, num_voxels * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_z_score_raw, num_voxels * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_z_score_smooth, num_voxels * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_saliency, num_voxels * sizeof(float)));

    // Masks for Hysteresis & Geodesic Region Growing
    uint8_t* d_seeds = nullptr;
    uint8_t* d_cands = nullptr;
    uint8_t* d_next_seeds = nullptr;
    uint8_t* d_temp_morph = nullptr;
    int* d_changed = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_seeds, num_voxels * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_cands, num_voxels * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_next_seeds, num_voxels * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_temp_morph, num_voxels * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_changed, sizeof(int)));

    // Test volume and output buffer
    voxel_t* d_test_vol = nullptr;
    voxel_t* d_out_subtracted = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_test_vol, payload_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out_subtracted, payload_bytes));

    int block1D = 256;
    int grid1D = (int)((num_voxels + block1D - 1) / block1D);

    dim3 block3D(8, 8, 8);
    dim3 grid3D((ref_hdr.dim_x + block3D.x - 1) / block3D.x,
                (ref_hdr.dim_y + block3D.y - 1) / block3D.y,
                (ref_hdr.dim_z + block3D.z - 1) / block3D.z);

    // ------------------------------------------------------------------------
    // Step 1: Compute Baseline Statistical Atlas on GPU
    // ------------------------------------------------------------------------
    std::printf("\n[ATLAS] Generating Statistical Baseline Atlas (mu_B, sigma_B) in VRAM...\n");
    ComputeBaselineAtlasKernel<<<grid1D, block1D>>>(d_baseline_ptrs, num_baselines, d_atlas_mean, d_atlas_std, num_voxels);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::printf("  * Baseline Statistical Atlas ready in GPU VRAM.\n");

    // ------------------------------------------------------------------------
    // Step 2: Process Test Scans (Multi-Feature Saliency & Geodesic Hysteresis)
    // ------------------------------------------------------------------------
    std::printf("\n[PROCESS] Processing %zu test volume(s) with Adaptive Geodesic Saliency...\n", num_tests);

    std::string results_manifest_path = output_dir + "\\subtracted_manifest.txt";
    std::ofstream res_manifest(results_manifest_path.c_str());

    for (size_t t = 0; t < num_tests; ++t) {
        FILE* fp_test = std::fopen(test_files[t].c_str(), "rb");
        if (!fp_test) {
            std::fprintf(stderr, "Error opening test file %s\n", test_files[t].c_str());
            continue;
        }
        VolumeHeader test_hdr;
        if (std::fread(&test_hdr, 1, sizeof(VolumeHeader), fp_test) != sizeof(VolumeHeader) ||
            std::fread(h_pinned_staging, sizeof(voxel_t), num_voxels, fp_test) != num_voxels) {
            std::fprintf(stderr, "Error reading test volume %s\n", test_files[t].c_str());
            std::fclose(fp_test);
            continue;
        }
        std::fclose(fp_test);

        CHECK_CUDA(cudaMemcpy(d_test_vol, h_pinned_staging, payload_bytes, cudaMemcpyHostToDevice));

        // 1. Contralateral Asymmetry and Baseline Z-Score
        ComputeRawAsymAndZScoreKernel<<<grid3D, block3D>>>(
            d_test_vol, d_atlas_mean, d_atlas_std,
            d_asym_raw, d_z_score_raw,
            ref_hdr.dim_z, ref_hdr.dim_y, ref_hdr.dim_x
        );
        CHECK_CUDA(cudaGetLastError());

        // 2. 3D Spatial Gaussian Smoothing (highlight mass effect, suppress thin sulcal folds)
        GaussianSmooth3DKernel<<<grid3D, block3D>>>(d_asym_raw, d_asym_smooth, ref_hdr.dim_z, ref_hdr.dim_y, ref_hdr.dim_x);
        GaussianSmooth3DKernel<<<grid3D, block3D>>>(d_z_score_raw, d_z_score_smooth, ref_hdr.dim_z, ref_hdr.dim_y, ref_hdr.dim_x);
        CHECK_CUDA(cudaGetLastError());

        // 3. Disambiguation (resolve mirror phantom) and Saliency Fusion
        ComputeFusedSaliencyKernel<<<grid3D, block3D>>>(
            d_test_vol, d_asym_smooth, d_z_score_smooth, d_saliency,
            ref_hdr.dim_z, ref_hdr.dim_y, ref_hdr.dim_x
        );
        CHECK_CUDA(cudaGetLastError());

        // 4. Compute Statistical Moments of Saliency Field
        float sal_max = 0.0f, sal_mean = 0.0f, sal_std = 0.0f;
        ComputeHostMoments(d_saliency, num_voxels, sal_max, sal_mean, sal_std);

        // Adaptive Dual-Threshold Formulation:
        // Seeds require high confidence (mean + 2.5 std), candidates require moderate envelope (mean + 1.2 std)
        float t_high = sal_mean + 2.50f * sal_std;
        float t_low  = sal_mean + 1.20f * sal_std;

        if (t_high < 5000.0f) t_high = 5000.0f;
        if (t_low < 2500.0f)  t_low = 2500.0f;

        // 5. Initialize Seeds & Candidates
        InitHysteresisKernel<<<grid1D, block1D>>>(d_saliency, t_high, t_low, d_seeds, d_cands, num_voxels);
        CHECK_CUDA(cudaGetLastError());

        // 6. Parallel 3D 26-Connected Geodesic Dilation until Convergence
        int h_changed = 1;
        int iter_count = 0;
        const int max_iters = 80;

        while (h_changed && iter_count < max_iters) {
            h_changed = 0;
            CHECK_CUDA(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));

            GeodesicGrow26Kernel<<<grid3D, block3D>>>(
                d_cands, d_seeds, d_next_seeds, d_changed,
                ref_hdr.dim_z, ref_hdr.dim_y, ref_hdr.dim_x
            );
            CHECK_CUDA(cudaGetLastError());

            // Ping-pong buffers
            uint8_t* temp = d_seeds;
            d_seeds = d_next_seeds;
            d_next_seeds = temp;

            CHECK_CUDA(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
            iter_count++;
        }

        // 7. Morphological Consolidation (Closing: Dilate -> Erode, Opening: Erode -> Dilate)
        // Dilate seeds (d_seeds -> d_temp_morph)
        Dilate6Kernel<<<grid3D, block3D>>>(d_seeds, d_temp_morph, ref_hdr.dim_z, ref_hdr.dim_y, ref_hdr.dim_x);
        // Erode back (d_temp_morph -> d_seeds) : Closing complete
        Erode6Kernel<<<grid3D, block3D>>>(d_temp_morph, d_seeds, ref_hdr.dim_z, ref_hdr.dim_y, ref_hdr.dim_x);
        CHECK_CUDA(cudaGetLastError());

        // 8. Mask output with raw test intensities
        ApplyMaskToOutputKernel<<<grid1D, block1D>>>(d_test_vol, d_seeds, d_out_subtracted, num_voxels);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpy(h_pinned_staging, d_out_subtracted, payload_bytes, cudaMemcpyDeviceToHost));

        // Transfer saliency field to host for topological cluster refinement
        std::vector<float> h_saliency(num_voxels);
        CHECK_CUDA(cudaMemcpy(h_saliency.data(), d_saliency, num_voxels * sizeof(float), cudaMemcpyDeviceToHost));

        // Retain only primary compact tumor mass, pruning sprawling dural/ventricular noise
        FilterPrimaryTumorClusters(h_pinned_staging, h_saliency.data(), ref_hdr.dim_z, ref_hdr.dim_y, ref_hdr.dim_x);

        char out_path[MAX_PATH];
        std::snprintf(out_path, sizeof(out_path), "%s\\subtracted_tumor_test_%zu.bin", output_dir.c_str(), t + 1);

        FILE* fp_out = std::fopen(out_path, "wb");
        if (fp_out) {
            std::fwrite(&ref_hdr, 1, sizeof(VolumeHeader), fp_out);
            std::fwrite(h_pinned_staging, sizeof(voxel_t), num_voxels, fp_out);
            std::fclose(fp_out);
        }

        if (res_manifest.is_open()) {
            res_manifest << out_path << "\n";
        }

        std::printf("  [TEST %zu/%zu] %s -> %s (Converged in %d geodesic iters, S_max: %.1f)\n",
                    t + 1, num_tests, test_files[t].c_str(), out_path, iter_count, sal_max);
    }

    if (res_manifest.is_open()) {
        res_manifest.close();
        std::printf("\n[MANIFEST] Saved results manifest: %s\n", results_manifest_path.c_str());
    }

    // Cleanup resources
    for (size_t b = 0; b < num_baselines; ++b) {
        if (d_baselines[b]) cudaFree(d_baselines[b]);
    }
    if (d_baseline_ptrs) cudaFree(d_baseline_ptrs);
    if (d_atlas_mean) cudaFree(d_atlas_mean);
    if (d_atlas_std) cudaFree(d_atlas_std);
    if (d_asym_raw) cudaFree(d_asym_raw);
    if (d_asym_smooth) cudaFree(d_asym_smooth);
    if (d_z_score_raw) cudaFree(d_z_score_raw);
    if (d_z_score_smooth) cudaFree(d_z_score_smooth);
    if (d_saliency) cudaFree(d_saliency);
    if (d_seeds) cudaFree(d_seeds);
    if (d_cands) cudaFree(d_cands);
    if (d_next_seeds) cudaFree(d_next_seeds);
    if (d_temp_morph) cudaFree(d_temp_morph);
    if (d_changed) cudaFree(d_changed);
    if (d_test_vol) cudaFree(d_test_vol);
    if (d_out_subtracted) cudaFree(d_out_subtracted);
    if (h_pinned_staging) cudaFreeHost(h_pinned_staging);

    std::printf("[SUCCESS] Next-Generation Tumor Detection completed successfully!\n");
    return EXIT_SUCCESS;
}
