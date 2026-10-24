/*
 * detect_tumor.cu
 * ===============
 * Production-Grade 3D Classical Brain MRI Tumor Detection Engine
 *
 * Requirements & Architectural Flow:
 * 1. CLI Arguments & Manifests:
 *    - Test data manifest (.txt)
 *    - Baseline images manifest (.txt)
 *    - Baseline Level-1 difference manifest (.txt) containing the precomputed baseline pairwise difference volumes
 *      (or auto-fallback to computing them in VRAM if not provided or set to "auto")
 *    - Output directory for segmented tumor volumes
 *    - Filter mode (1: Laplacian, 2: Gaussian, 3: Median, 4: Mean; prompted if omitted)
 * 2. CUDA Memory Architecture & Simultaneous VRAM Residency:
 *    - Reusable pinned host staging buffer (cudaHostAlloc) for DMA transfers.
 *    - Loads ALL baseline images simultaneously into VRAM.
 *    - Loads ALL baseline Level-1 difference maps into VRAM.
 * 3. 4 Selectable 3D Spatial Difference Filtering Modes:
 *    - Mode 1: 3D Discrete Laplacian (6-face connected 2nd-order discrete derivative)
 *    - Mode 2: 3D Gaussian Smoothing (3x3x3 edge/noise attenuation)
 *    - Mode 3: 3D Median Filter (3x3x3 non-linear sorting network for impulse suppression)
 *    - Mode 4: 3D Mean / Box Filter (3x3x3 local neighborhood averaging)
 * 4. Dual-Level Difference Retention & Joint Verification:
 *    - 1st-level difference: D1 = |Test - Baseline|
 *    - 2nd-level difference: D2 = ||Test - Baseline| - Baseline_Diff|
 *    - Statistical baseline thresholds: T_d1 = mu_d1 + k1*sigma_d1, T_d2 = mu_d2 + k2*sigma_d2
 *    - Joint verification enforces candidate voxels to pass BOTH D1 and D2 simultaneously.
 * 5. False Positive & False Negative Suppression:
 *    - Contralateral Hemispheric Symmetry Verification across Mid-Sagittal Plane (MSP):
 *      suppresses symmetric normal brain anatomy while retaining asymmetric tumor lesions.
 *    - Local baseline spatial filter feature modulation: adapts thresholds to local anatomical complexity.
 * 6. File & Protocol Standards:
 *    - 18-byte packed VolumeHeader: uint16_t dim_z, dim_y, dim_x; float vz, vy, vx; (#pragma pack(push, 1))
 *    - Pure Win32 API directory creation (CreateDirectoryA, strictly no <filesystem>).
 *    - Dual precision support: compile with -DUSE_UINT16 or -DUSE_FLOAT32 (default).
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
    FILTER_LAPLACIAN = 1,
    FILTER_GAUSSIAN  = 2,
    FILTER_MEDIAN    = 3,
    FILTER_MEAN      = 4
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
    while (std::getline(infile, line)) {
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
// CUDA Kernels: In-Memory Baseline Combinatorial Differencing (Fallback/Dynamic)
// ----------------------------------------------------------------------------

__global__ void ComputePairwiseDiffKernel(const voxel_t* __restrict__ vol_a,
                                          const voxel_t* __restrict__ vol_b,
                                          voxel_t* __restrict__ diff_out,
                                          size_t num_voxels) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_voxels) return;

    float a = (float)vol_a[idx];
    float b = (float)vol_b[idx];
    float d = fabsf(a - b);

#if defined(USE_UINT16)
    if (d > 65535.0f) d = 65535.0f;
    diff_out[idx] = (uint16_t)(d + 0.5f);
#else
    diff_out[idx] = d;
#endif
}

// ----------------------------------------------------------------------------
// CUDA Kernels: 4 Selectable 3D Spatial Difference Filtering Modes
// ----------------------------------------------------------------------------

// Mode 1: 3D 6-face Discrete Laplacian
__global__ void FilterLaplacianKernel(const voxel_t* __restrict__ in_vol,
                                      float* __restrict__ out_vol,
                                      uint16_t dim_z, uint16_t dim_y, uint16_t dim_x) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= dim_x || y >= dim_y || z >= dim_z) return;

    size_t idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + x;
    float center = (float)in_vol[idx];

    if (center <= 0.0f) {
        out_vol[idx] = 0.0f;
        return;
    }

    float vx_m = (x > 0) ? (float)in_vol[idx - 1] : center;
    float vx_p = (x < dim_x - 1) ? (float)in_vol[idx + 1] : center;
    float vy_m = (y > 0) ? (float)in_vol[idx - dim_x] : center;
    float vy_p = (y < dim_y - 1) ? (float)in_vol[idx + dim_x] : center;
    float vz_m = (z > 0) ? (float)in_vol[idx - (size_t)dim_y * dim_x] : center;
    float vz_p = (z < dim_z - 1) ? (float)in_vol[idx + (size_t)dim_y * dim_x] : center;

    float lap = fabsf(vx_p + vx_m + vy_p + vy_m + vz_p + vz_m - 6.0f * center);
    out_vol[idx] = lap;
}

// Mode 2: 3D Gaussian Smoothing (3x3x3 binomial [1, 2, 1] per axis, total weight = 64)
__global__ void FilterGaussianKernel(const voxel_t* __restrict__ in_vol,
                                     float* __restrict__ out_vol,
                                     uint16_t dim_z, uint16_t dim_y, uint16_t dim_x) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= dim_x || y >= dim_y || z >= dim_z) return;

    size_t idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + x;
    float center = (float)in_vol[idx];

    if (center <= 0.0f) {
        out_vol[idx] = 0.0f;
        return;
    }

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
                sum += (float)in_vol[n_idx] * weight;
                total_weight += weight;
            }
        }
    }

    out_vol[idx] = (total_weight > 0.0f) ? (sum / total_weight) : 0.0f;
}

// Mode 3: 3D Median Filter (3x3x3 27-element insertion sort)
__global__ void FilterMedianKernel(const voxel_t* __restrict__ in_vol,
                                   float* __restrict__ out_vol,
                                   uint16_t dim_z, uint16_t dim_y, uint16_t dim_x) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= dim_x || y >= dim_y || z >= dim_z) return;

    size_t idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + x;
    float center = (float)in_vol[idx];

    if (center <= 0.0f) {
        out_vol[idx] = 0.0f;
        return;
    }

    float arr[27];
    int count = 0;

    for (int dz = -1; dz <= 1; ++dz) {
        int cz = z + dz;
        if (cz < 0 || cz >= dim_z) continue;

        for (int dy = -1; dy <= 1; ++dy) {
            int cy = y + dy;
            if (cy < 0 || cy >= dim_y) continue;

            for (int dx = -1; dx <= 1; ++dx) {
                int cx = x + dx;
                if (cx < 0 || cx >= dim_x) continue;

                size_t n_idx = (size_t)cz * dim_y * dim_x + (size_t)cy * dim_x + cx;
                arr[count++] = (float)in_vol[n_idx];
            }
        }
    }

    for (int i = 1; i < count; ++i) {
        float key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }

    out_vol[idx] = arr[count / 2];
}

// Mode 4: 3D Mean / Box Filter
__global__ void FilterMeanKernel(const voxel_t* __restrict__ in_vol,
                                 float* __restrict__ out_vol,
                                 uint16_t dim_z, uint16_t dim_y, uint16_t dim_x) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= dim_x || y >= dim_y || z >= dim_z) return;

    size_t idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + x;
    float center = (float)in_vol[idx];

    if (center <= 0.0f) {
        out_vol[idx] = 0.0f;
        return;
    }

    float sum = 0.0f;
    int count = 0;

    for (int dz = -1; dz <= 1; ++dz) {
        int cz = z + dz;
        if (cz < 0 || cz >= dim_z) continue;

        for (int dy = -1; dy <= 1; ++dy) {
            int cy = y + dy;
            if (cy < 0 || cy >= dim_y) continue;

            for (int dx = -1; dx <= 1; ++dx) {
                int cx = x + dx;
                if (cx < 0 || cx >= dim_x) continue;

                size_t n_idx = (size_t)cz * dim_y * dim_x + (size_t)cy * dim_x + cx;
                sum += (float)in_vol[n_idx];
                count++;
            }
        }
    }

    out_vol[idx] = (count > 0) ? (sum / (float)count) : 0.0f;
}

// ----------------------------------------------------------------------------
// CUDA Parallel Reduction: Moments (Sum and Sum of Squares)
// ----------------------------------------------------------------------------

__global__ void ReduceMomentsKernel(const float* __restrict__ map_in,
                                    double* __restrict__ block_sums,
                                    double* __restrict__ block_sq_sums,
                                    uint64_t* __restrict__ block_counts,
                                    size_t num_voxels) {
    extern __shared__ double s_moments[];
    double* s_sum = s_moments;
    double* s_sq_sum = &s_moments[blockDim.x];
    uint64_t* s_count = (uint64_t*)&s_moments[2 * blockDim.x];

    unsigned int tid = threadIdx.x;
    size_t gid = (size_t)blockDim.x * blockIdx.x + threadIdx.x;

    double my_sum = 0.0;
    double my_sq_sum = 0.0;
    uint64_t my_count = 0;

    if (gid < num_voxels) {
        float val = map_in[gid];
        if (val > 0.0f) {
            double dval = (double)val;
            my_sum = dval;
            my_sq_sum = dval * dval;
            my_count = 1;
        }
    }

    s_sum[tid] = my_sum;
    s_sq_sum[tid] = my_sq_sum;
    s_count[tid] = my_count;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq_sum[tid] += s_sq_sum[tid + s];
            s_count[tid] += s_count[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = s_sum[0];
        block_sq_sums[blockIdx.x] = s_sq_sum[0];
        block_counts[blockIdx.x] = s_count[0];
    }
}

// ----------------------------------------------------------------------------
// Dual-Level Difference & Joint Verification Kernel
// ----------------------------------------------------------------------------

__global__ void JointVerificationKernel(const voxel_t* __restrict__ test_vol,
                                        const voxel_t* const* __restrict__ baseline_vols,
                                        size_t num_baselines,
                                        const voxel_t* const* __restrict__ baseline_diffs,
                                        size_t num_diffs,
                                        const float* __restrict__ baseline_feature_map,
                                        float T_dyn_d1,
                                        float T_dyn_d2,
                                        float asym_factor,
                                        voxel_t* __restrict__ out_subtracted,
                                        uint16_t dim_z, uint16_t dim_y, uint16_t dim_x) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= dim_x || y >= dim_y || z >= dim_z) return;

    size_t idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + x;
    voxel_t test_val_raw = test_vol[idx];
    float test_val = (float)test_val_raw;

    // Suppress air background
    if (test_val <= 0.0f) {
        out_subtracted[idx] = ZERO_VAL;
        return;
    }

    // Contralateral Hemispheric Symmetry Verification across Mid-Sagittal Plane (MSP)
    int x_mirror = (int)dim_x - 1 - x;
    size_t mirror_idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + x_mirror;
    float mirror_val = (float)test_vol[mirror_idx];
    float asymmetry = fabsf(test_val - mirror_val);

    // 1st-level difference: D1 = |Test - Baseline| across baseline images
    float sum_d1 = 0.0f;
    for (size_t b = 0; b < num_baselines; ++b) {
        float b_val = (float)baseline_vols[b][idx];
        sum_d1 += fabsf(test_val - b_val);
    }
    float mean_d1 = sum_d1 / (float)num_baselines;

    // Baseline natural anatomical variability from Level-1 baseline difference volumes
    float sum_b_diff = 0.0f;
    for (size_t d = 0; d < num_diffs; ++d) {
        sum_b_diff += (float)baseline_diffs[d][idx];
    }
    float mean_b_diff = (num_diffs > 0) ? (sum_b_diff / (float)num_diffs) : 0.0f;

    // 2nd-level difference: D2 = ||Test - Baseline| - Baseline_Diff|
    float d2_residual = mean_d1 - mean_b_diff;
    if (d2_residual < 0.0f) d2_residual = 0.0f;

    // Local baseline spatial filter feature magnitude
    float local_feat = baseline_feature_map[idx];

    // Thresholds modulated by local anatomical feature strength to eliminate false-positive edges
    float effective_thresh_d1 = T_dyn_d1 + 0.25f * local_feat;
    float effective_thresh_d2 = T_dyn_d2 + 0.20f * local_feat;

    // Joint Verification conditions:
    // 1. D1 must exceed threshold (test scan deviates from baseline healthy cohort)
    // 2. D2 must exceed threshold (residual deviation exceeds normal healthy variability)
    // 3. Significant contralateral asymmetry (tumor is asymmetric; bilateral brain structures are symmetric)
    bool passes_d1 = (mean_d1 + asym_factor * asymmetry) > effective_thresh_d1;
    bool passes_d2 = (d2_residual + asym_factor * asymmetry) > effective_thresh_d2;
    bool passes_asymmetry = asymmetry > (0.10f * T_dyn_d1);

    if (passes_d1 && passes_d2 && passes_asymmetry) {
        out_subtracted[idx] = test_val_raw;
    } else {
        out_subtracted[idx] = ZERO_VAL;
    }
}

// ----------------------------------------------------------------------------
// Host Orchestration
// ----------------------------------------------------------------------------

void LaunchFilter(FilterMode mode, const voxel_t* d_in, float* d_out,
                  uint16_t dim_z, uint16_t dim_y, uint16_t dim_x,
                  dim3 grid3D, dim3 block3D) {
    switch (mode) {
        case FILTER_LAPLACIAN:
            FilterLaplacianKernel<<<grid3D, block3D>>>(d_in, d_out, dim_z, dim_y, dim_x);
            break;
        case FILTER_GAUSSIAN:
            FilterGaussianKernel<<<grid3D, block3D>>>(d_in, d_out, dim_z, dim_y, dim_x);
            break;
        case FILTER_MEDIAN:
            FilterMedianKernel<<<grid3D, block3D>>>(d_in, d_out, dim_z, dim_y, dim_x);
            break;
        case FILTER_MEAN:
            FilterMeanKernel<<<grid3D, block3D>>>(d_in, d_out, dim_z, dim_y, dim_x);
            break;
        default:
            FilterLaplacianKernel<<<grid3D, block3D>>>(d_in, d_out, dim_z, dim_y, dim_x);
            break;
    }
    CHECK_CUDA(cudaGetLastError());
}

void ComputeStats(const float* d_map, size_t num_voxels, float& mean, float& std_dev) {
    int redBlockSize = 256;
    int redNumBlocks = (int)((num_voxels + redBlockSize - 1) / redBlockSize);

    double* d_sums = nullptr;
    double* d_sq_sums = nullptr;
    uint64_t* d_counts = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_sums, redNumBlocks * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_sq_sums, redNumBlocks * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_counts, redNumBlocks * sizeof(uint64_t)));

    size_t smem = 2 * redBlockSize * sizeof(double) + redBlockSize * sizeof(uint64_t);
    ReduceMomentsKernel<<<redNumBlocks, redBlockSize, smem>>>(d_map, d_sums, d_sq_sums, d_counts, num_voxels);
    CHECK_CUDA(cudaGetLastError());

    std::vector<double> h_sums(redNumBlocks);
    std::vector<double> h_sq_sums(redNumBlocks);
    std::vector<uint64_t> h_counts(redNumBlocks);

    CHECK_CUDA(cudaMemcpy(h_sums.data(), d_sums, redNumBlocks * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sq_sums.data(), d_sq_sums, redNumBlocks * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_counts.data(), d_counts, redNumBlocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    cudaFree(d_sums);
    cudaFree(d_sq_sums);
    cudaFree(d_counts);

    double total_sum = 0.0;
    double total_sq_sum = 0.0;
    uint64_t total_count = 0;

    for (int i = 0; i < redNumBlocks; ++i) {
        total_sum += h_sums[i];
        total_sq_sum += h_sq_sums[i];
        total_count += h_counts[i];
    }

    if (total_count == 0) {
        mean = 0.0f;
        std_dev = 0.0f;
        return;
    }

    double d_mean = total_sum / (double)total_count;
    double variance = (total_sq_sum / (double)total_count) - (d_mean * d_mean);
    if (variance < 0.0) variance = 0.0;

    mean = (float)d_mean;
    std_dev = (float)std::sqrt(variance);
}

int main(int argc, char* argv[]) {
    std::string test_manifest_path = "";
    std::string baseline_manifest_path = "";
    std::string diff_manifest_path = "";
    std::string output_dir = "";
    int chosen_filter_mode = 0;

    // Support CLI invocations: defaults to FILTER_LAPLACIAN (1) without interactive prompting
    chosen_filter_mode = FILTER_LAPLACIAN;

    if (argc >= 5) {
        test_manifest_path = argv[1];
        baseline_manifest_path = argv[2];
        diff_manifest_path = argv[3];
        output_dir = argv[4];
        if (argc >= 6) {
            chosen_filter_mode = std::atoi(argv[5]);
        }
    } else if (argc == 4) {
        test_manifest_path = argv[1];
        baseline_manifest_path = argv[2];
        diff_manifest_path = "auto";
        output_dir = argv[3];
    } else {
        std::cout << "Usage: detect_tumor <test_manifest> <baseline_manifest> [diff_manifest|auto] <output_dir> [filter_id(1-4)]" << std::endl;
        return EXIT_FAILURE;
    }

    if (chosen_filter_mode < 1 || chosen_filter_mode > 4) {
        chosen_filter_mode = FILTER_LAPLACIAN;
    }

    FilterMode filter_mode = static_cast<FilterMode>(chosen_filter_mode);
    const char* filter_names[] = {"", "3D Laplacian", "3D Gaussian", "3D Median", "3D Mean / Box"};
    std::printf("\n[FILTER SELECTION] Active Filter Mode: %s\n", filter_names[chosen_filter_mode]);

    if (!CreateDirectoryRecursive(output_dir)) {
        std::fprintf(stderr, "Warning: Failed or existing directory: %s\n", output_dir.c_str());
    }

    std::vector<std::string> test_files = ReadManifest(test_manifest_path);
    std::vector<std::string> baseline_files = ReadManifest(baseline_manifest_path);
    std::vector<std::string> diff_files;

    if (!diff_manifest_path.empty() && diff_manifest_path != "auto" && diff_manifest_path != "none") {
        diff_files = ReadManifest(diff_manifest_path);
    }

    size_t num_tests = test_files.size();
    size_t num_baselines = baseline_files.size();

    if (num_tests == 0 || num_baselines == 0) {
        std::fprintf(stderr, "Error: Missing test or baseline files in manifests!\n");
        return EXIT_FAILURE;
    }

    size_t expected_pairs = (num_baselines * (num_baselines - 1)) / 2;

    // Read reference header from first baseline
    VolumeHeader ref_hdr;
    FILE* fp_ref = std::fopen(baseline_files[0].c_str(), "rb");
    if (!fp_ref || std::fread(&ref_hdr, 1, sizeof(VolumeHeader), fp_ref) != sizeof(VolumeHeader)) {
        std::fprintf(stderr, "Error: Could not read reference header from %s\n", baseline_files[0].c_str());
        if (fp_ref) std::fclose(fp_ref);
        return EXIT_FAILURE;
    }
    std::fclose(fp_ref);

    size_t num_voxels = (size_t)ref_hdr.dim_z * ref_hdr.dim_y * ref_hdr.dim_x;
    size_t payload_bytes = num_voxels * sizeof(voxel_t);

    std::printf("[GEOMETRY] Matrix: %u x %u x %u (%zu voxels), Spacing: [%.2f, %.2f, %.2f] mm\n",
                ref_hdr.dim_z, ref_hdr.dim_y, ref_hdr.dim_x, num_voxels,
                ref_hdr.vz, ref_hdr.vy, ref_hdr.vx);

    // Reusable pinned host staging buffer
    voxel_t* h_pinned_staging = nullptr;
    CHECK_CUDA(cudaHostAlloc((void**)&h_pinned_staging, payload_bytes, cudaHostAllocDefault));

    // Allocate and load baseline volumes into GPU VRAM
    std::vector<voxel_t*> d_baselines(num_baselines, nullptr);
    for (size_t b = 0; b < num_baselines; ++b) {
        CHECK_CUDA(cudaMalloc((void**)&d_baselines[b], payload_bytes));
        FILE* fp = std::fopen(baseline_files[b].c_str(), "rb");
        VolumeHeader cur_hdr;
        if (!fp || std::fread(&cur_hdr, 1, sizeof(VolumeHeader), fp) != sizeof(VolumeHeader) ||
            std::fread(h_pinned_staging, sizeof(voxel_t), num_voxels, fp) != num_voxels) {
            std::fprintf(stderr, "Error loading baseline %s\n", baseline_files[b].c_str());
            if (fp) std::fclose(fp);
            return EXIT_FAILURE;
        }
        std::fclose(fp);
        CHECK_CUDA(cudaMemcpy(d_baselines[b], h_pinned_staging, payload_bytes, cudaMemcpyHostToDevice));
    }
    std::printf("[VRAM] Loaded %zu baseline volumes into VRAM (%.2f MB)\n",
                num_baselines, (double)(num_baselines * payload_bytes) / (1024.0 * 1024.0));

    // Load or dynamically generate baseline Level-1 difference volumes
    std::vector<voxel_t*> d_diffs;
    size_t num_diffs = diff_files.size();

    if (num_diffs > 0) {
        std::printf("[LOAD] Loading %zu precomputed Level-1 baseline difference volumes from manifest into VRAM...\n", num_diffs);
        d_diffs.resize(num_diffs, nullptr);
        for (size_t d = 0; d < num_diffs; ++d) {
            CHECK_CUDA(cudaMalloc((void**)&d_diffs[d], payload_bytes));
            FILE* fp = std::fopen(diff_files[d].c_str(), "rb");
            VolumeHeader cur_hdr;
            if (!fp || std::fread(&cur_hdr, 1, sizeof(VolumeHeader), fp) != sizeof(VolumeHeader) ||
                std::fread(h_pinned_staging, sizeof(voxel_t), num_voxels, fp) != num_voxels) {
                std::fprintf(stderr, "Error loading baseline diff %s\n", diff_files[d].c_str());
                if (fp) std::fclose(fp);
                return EXIT_FAILURE;
            }
            std::fclose(fp);
            CHECK_CUDA(cudaMemcpy(d_diffs[d], h_pinned_staging, payload_bytes, cudaMemcpyHostToDevice));
        }
    } else {
        std::printf("[VRAM] Baseline diff manifest not provided; dynamically computing all %zu Level-1 baseline pairs in VRAM...\n",
                    expected_pairs);
        num_diffs = expected_pairs;
        d_diffs.resize(num_diffs, nullptr);
        size_t pair_idx = 0;
        int block1D = 256;
        int grid1D = (int)((num_voxels + block1D - 1) / block1D);

        for (size_t i = 0; i < num_baselines; ++i) {
            for (size_t j = i + 1; j < num_baselines; ++j) {
                CHECK_CUDA(cudaMalloc((void**)&d_diffs[pair_idx], payload_bytes));
                ComputePairwiseDiffKernel<<<grid1D, block1D>>>(
                    d_baselines[i], d_baselines[j], d_diffs[pair_idx], num_voxels
                );
                CHECK_CUDA(cudaGetLastError());
                pair_idx++;
            }
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    std::printf("[VRAM] Loaded/Resident %zu Level-1 baseline difference maps (%.2f MB)\n",
                num_diffs, (double)(num_diffs * payload_bytes) / (1024.0 * 1024.0));

    // Device pointer tables
    voxel_t** d_baseline_ptrs = nullptr;
    CHECK_CUDA(cudaMalloc((void***)&d_baseline_ptrs, num_baselines * sizeof(voxel_t*)));
    CHECK_CUDA(cudaMemcpy(d_baseline_ptrs, d_baselines.data(), num_baselines * sizeof(voxel_t*), cudaMemcpyHostToDevice));

    voxel_t** d_diff_ptrs = nullptr;
    CHECK_CUDA(cudaMalloc((void***)&d_diff_ptrs, num_diffs * sizeof(voxel_t*)));
    CHECK_CUDA(cudaMemcpy(d_diff_ptrs, d_diffs.data(), num_diffs * sizeof(voxel_t*), cudaMemcpyHostToDevice));

    // Allocate test volume and output buffer
    voxel_t* d_test_vol = nullptr;
    voxel_t* d_out_subtracted = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_test_vol, payload_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out_subtracted, payload_bytes));

    // ------------------------------------------------------------------------
    // Baseline-Derived Dynamic Thresholding with Chosen Filter
    // ------------------------------------------------------------------------
    std::printf("\n[STAGE 1] Filtering reference baseline and computing dynamic thresholds...\n");
    float* d_baseline_feat = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_baseline_feat, num_voxels * sizeof(float)));

    dim3 block3D(8, 8, 8);
    dim3 grid3D((ref_hdr.dim_x + block3D.x - 1) / block3D.x,
                (ref_hdr.dim_y + block3D.y - 1) / block3D.y,
                (ref_hdr.dim_z + block3D.z - 1) / block3D.z);

    // Filter reference baseline
    LaunchFilter(filter_mode, d_baselines[0], d_baseline_feat,
                 ref_hdr.dim_z, ref_hdr.dim_y, ref_hdr.dim_x, grid3D, block3D);

    float feat_mean = 0.0f, feat_std = 0.0f;
    ComputeStats(d_baseline_feat, num_voxels, feat_mean, feat_std);

    // Filter Level-1 baseline difference map to obtain baseline noise moments
    float* d_temp_diff_feat = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_temp_diff_feat, num_voxels * sizeof(float)));
    LaunchFilter(filter_mode, d_diffs[0], d_temp_diff_feat,
                 ref_hdr.dim_z, ref_hdr.dim_y, ref_hdr.dim_x, grid3D, block3D);

    float diff_mean = 0.0f, diff_std = 0.0f;
    ComputeStats(d_temp_diff_feat, num_voxels, diff_mean, diff_std);
    cudaFree(d_temp_diff_feat);

    // Dynamic statistical threshold formulation: T = mu + k * sigma
#if defined(USE_UINT16)
    float k_d1 = 1.8f;
    float k_d2 = 1.5f;
    float asym_factor = 1.25f;
#else
    float k_d1 = 1.7f;
    float k_d2 = 1.4f;
    float asym_factor = 1.25f;
#endif

    float T_dyn_d1 = feat_mean + k_d1 * feat_std;
    float T_dyn_d2 = diff_mean + k_d2 * diff_std;

    std::printf("  * Filter Reference Mean : %.4f (Std Dev: %.4f)\n", feat_mean, feat_std);
    std::printf("  * Baseline Diff Mean    : %.4f (Std Dev: %.4f)\n", diff_mean, diff_std);
    std::printf("  * Dynamic Threshold D1  : %.4f (k=%.2f)\n", T_dyn_d1, k_d1);
    std::printf("  * Dynamic Threshold D2  : %.4f (k=%.2f)\n", T_dyn_d2, k_d2);
    std::printf("  * Contralateral Factor  : %.2f\n", asym_factor);

    // ------------------------------------------------------------------------
    // Process Test Volumes: Dual-Level Difference & Joint Verification
    // ------------------------------------------------------------------------
    std::printf("\n[STAGE 2 & 3] Processing %zu test volume(s) with Joint D1+D2 Verification...\n", num_tests);

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

        // Execute Joint Verification Kernel using loaded baseline images and Level-1 difference volumes
        JointVerificationKernel<<<grid3D, block3D>>>(
            d_test_vol,
            d_baseline_ptrs,
            num_baselines,
            d_diff_ptrs,
            num_diffs,
            d_baseline_feat,
            T_dyn_d1,
            T_dyn_d2,
            asym_factor,
            d_out_subtracted,
            ref_hdr.dim_z, ref_hdr.dim_y, ref_hdr.dim_x
        );
        CHECK_CUDA(cudaGetLastError());

        // Copy final tumor candidate volume to pinned host buffer
        CHECK_CUDA(cudaMemcpy(h_pinned_staging, d_out_subtracted, payload_bytes, cudaMemcpyDeviceToHost));

        char out_path[MAX_PATH];
        std::snprintf(out_path, sizeof(out_path), "%s\\subtracted_tumor_test_%zu.bin", output_dir.c_str(), t + 1);

        FILE* fp_out = std::fopen(out_path, "wb");
        if (!fp_out) {
            std::fprintf(stderr, "Error writing output %s\n", out_path);
            continue;
        }
        std::fwrite(&ref_hdr, 1, sizeof(VolumeHeader), fp_out);
        std::fwrite(h_pinned_staging, sizeof(voxel_t), num_voxels, fp_out);
        std::fclose(fp_out);

        if (res_manifest.is_open()) {
            res_manifest << out_path << "\n";
        }

        std::printf("  [TEST %zu/%zu] Processed: %s -> %s\n",
                    t + 1, num_tests, test_files[t].c_str(), out_path);
    }

    if (res_manifest.is_open()) {
        res_manifest.close();
        std::printf("[MANIFEST] Saved subtracted tumor results manifest: %s\n", results_manifest_path.c_str());
    }

    // Cleanup resources
    for (size_t b = 0; b < num_baselines; ++b) {
        if (d_baselines[b]) cudaFree(d_baselines[b]);
    }
    for (size_t d = 0; d < num_diffs; ++d) {
        if (d_diffs[d]) cudaFree(d_diffs[d]);
    }
    if (d_baseline_ptrs) cudaFree(d_baseline_ptrs);
    if (d_diff_ptrs) cudaFree(d_diff_ptrs);
    if (d_test_vol) cudaFree(d_test_vol);
    if (d_out_subtracted) cudaFree(d_out_subtracted);
    if (d_baseline_feat) cudaFree(d_baseline_feat);
    if (h_pinned_staging) cudaFreeHost(h_pinned_staging);

    std::printf("[COMPLETE] Tumor segmentation pipeline executed successfully!\n");
    return EXIT_SUCCESS;
}
