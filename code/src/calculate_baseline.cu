/*
 * calculate_baseline.cu
 *
 * Baseline image processing engine.
 * Computes voxel-wise mean and standard deviation from baseline scans,
 * and generates pairwise baseline difference volumes in .bin format.
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
#else
    typedef float voxel_t;
    #define DTYPE_NAME "float32"
#endif

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

// Win32 recursive directory creation (avoiding <filesystem>)
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

// ----------------------------------------------------------------------------
// CUDA Kernels: Statistical Atlas Generation
// ----------------------------------------------------------------------------

__global__ void ComputeAtlasMeanStdKernel(const voxel_t* const* __restrict__ baseline_vols,
                                          size_t num_baselines,
                                          voxel_t* __restrict__ mean_out,
                                          voxel_t* __restrict__ std_out,
                                          size_t num_voxels) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_voxels) return;

    if (num_baselines == 0) {
        mean_out[idx] = 0;
        std_out[idx] = 0;
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
    double std_dev = std::sqrt(var);

#if defined(USE_UINT16)
    if (mean > 65535.0) mean = 65535.0;
    if (std_dev > 65535.0) std_dev = 65535.0;
    mean_out[idx] = (uint16_t)(mean + 0.5);
    std_out[idx] = (uint16_t)(std_dev + 0.5);
#else
    mean_out[idx] = (float)mean;
    std_out[idx] = (float)std_dev;
#endif
}

// ----------------------------------------------------------------------------
// CUDA Kernels: Combinatorial Pairwise Absolute Difference
// ----------------------------------------------------------------------------

#if defined(USE_UINT16)
__global__ void AbsDiffKernel(const uint16_t* __restrict__ img_a,
                              const uint16_t* __restrict__ img_b,
                              uint16_t* __restrict__ diff_out,
                              size_t num_voxels) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_voxels) {
        int64_t va = (int64_t)img_a[idx];
        int64_t vb = (int64_t)img_b[idx];
        int64_t diff = va - vb;
        diff_out[idx] = (uint16_t)(diff >= 0 ? diff : -diff);
    }
}
#else
__global__ void AbsDiffKernel(const float* __restrict__ img_a,
                              const float* __restrict__ img_b,
                              float* __restrict__ diff_out,
                              size_t num_voxels) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_voxels) {
        diff_out[idx] = fabsf(img_a[idx] - img_b[idx]);
    }
}
#endif

std::vector<std::string> ReadManifest(const std::string& manifest_path) {
    std::vector<std::string> file_paths;
    std::ifstream infile(manifest_path.c_str());
    if (!infile.is_open()) {
        std::fprintf(stderr, "Error: Unable to open manifest file: %s\n", manifest_path.c_str());
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

int main(int argc, char* argv[]) {
    std::string manifest_path = "";
    std::string output_dir = "";

    if (argc >= 3) {
        manifest_path = argv[1];
        output_dir = argv[2];
    } else {
        std::cout << "=== Statistical & Combinatorial Baseline Engine (Classical DIP) ===" << std::endl;
        std::cout << "Mode: " << DTYPE_NAME << " (Size of voxel_t: " << sizeof(voxel_t) << " bytes)" << std::endl;
        std::cout << "Enter baseline binaries manifest path: ";
        std::getline(std::cin, manifest_path);
        std::cout << "Enter output directory for baseline atlas & difference maps: ";
        std::getline(std::cin, output_dir);
    }

    if (!CreateDirectoryRecursive(output_dir)) {
        std::fprintf(stderr, "Warning/Error: Failed or already existing output directory: %s\n", output_dir.c_str());
    }

    std::vector<std::string> files = ReadManifest(manifest_path);
    size_t N = files.size();
    if (N < 2) {
        std::fprintf(stderr, "Error: Need at least 2 baseline images to compute baseline statistics. Found: %zu\n", N);
        return EXIT_FAILURE;
    }

    std::printf("[INFO] Found %zu baseline volumes in manifest.\n", N);

    VolumeHeader ref_header;
    FILE* fp0 = std::fopen(files[0].c_str(), "rb");
    if (!fp0) {
        std::fprintf(stderr, "Error: Could not open reference file: %s\n", files[0].c_str());
        return EXIT_FAILURE;
    }
    if (std::fread(&ref_header, 1, sizeof(VolumeHeader), fp0) != sizeof(VolumeHeader)) {
        std::fprintf(stderr, "Error: Could not read 18-byte header from: %s\n", files[0].c_str());
        std::fclose(fp0);
        return EXIT_FAILURE;
    }
    std::fclose(fp0);

    size_t num_voxels = (size_t)ref_header.dim_z * ref_header.dim_y * ref_header.dim_x;
    size_t payload_bytes = num_voxels * sizeof(voxel_t);

    std::printf("[GEOMETRY] Dimensions: %u x %u x %u (Total voxels: %zu)\n",
                ref_header.dim_z, ref_header.dim_y, ref_header.dim_x, num_voxels);
    std::printf("[GEOMETRY] Spacing: vz=%.3f mm, vy=%.3f mm, vx=%.3f mm\n",
                ref_header.vz, ref_header.vy, ref_header.vx);
    std::printf("[MEMORY] Total VRAM for %zu volumes: %.2f MB\n", N, (double)(N * payload_bytes) / (1024.0 * 1024.0));

    // Reusable pinned host staging buffer
    voxel_t* h_pinned_staging = nullptr;
    CHECK_CUDA(cudaHostAlloc((void**)&h_pinned_staging, payload_bytes, cudaHostAllocDefault));

    // Allocate VRAM for all N baseline volumes
    std::vector<voxel_t*> d_volumes(N, nullptr);
    for (size_t i = 0; i < N; ++i) {
        CHECK_CUDA(cudaMalloc((void**)&d_volumes[i], payload_bytes));
    }

    // Allocate device buffers for statistical atlas & pairwise diff
    voxel_t* d_atlas_mean = nullptr;
    voxel_t* d_atlas_std = nullptr;
    voxel_t* d_diff_out = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_atlas_mean, payload_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_atlas_std, payload_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_diff_out, payload_bytes));

    // Load all baseline volumes into VRAM
    for (size_t i = 0; i < N; ++i) {
        FILE* fp = std::fopen(files[i].c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "Error: Failed to open %s\n", files[i].c_str());
            return EXIT_FAILURE;
        }
        VolumeHeader cur_hdr;
        if (std::fread(&cur_hdr, 1, sizeof(VolumeHeader), fp) != sizeof(VolumeHeader) ||
            std::fread(h_pinned_staging, sizeof(voxel_t), num_voxels, fp) != num_voxels) {
            std::fprintf(stderr, "Error: Failed to read volume %zu (%s)\n", i, files[i].c_str());
            std::fclose(fp);
            return EXIT_FAILURE;
        }
        std::fclose(fp);
        CHECK_CUDA(cudaMemcpy(d_volumes[i], h_pinned_staging, payload_bytes, cudaMemcpyHostToDevice));
        std::printf("[LOAD] (%zu/%zu) Streamed %s to VRAM\n", i + 1, N, files[i].c_str());
    }

    // Pointer array for device access
    voxel_t** d_vol_ptrs = nullptr;
    CHECK_CUDA(cudaMalloc((void***)&d_vol_ptrs, N * sizeof(voxel_t*)));
    CHECK_CUDA(cudaMemcpy(d_vol_ptrs, d_volumes.data(), N * sizeof(voxel_t*), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (int)((num_voxels + blockSize - 1) / blockSize);

    // ------------------------------------------------------------------------
    // Step 1: Compute Statistical Baseline Atlas (Mean & Std Dev)
    // ------------------------------------------------------------------------
    std::printf("\n[ATLAS] Computing Voxel-wise Statistical Baseline Atlas on GPU...\n");
    ComputeAtlasMeanStdKernel<<<numBlocks, blockSize>>>(d_vol_ptrs, N, d_atlas_mean, d_atlas_std, num_voxels);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Export baseline_atlas_mean.bin
    char mean_path[MAX_PATH];
    std::snprintf(mean_path, sizeof(mean_path), "%s\\baseline_atlas_mean.bin", output_dir.c_str());
    CHECK_CUDA(cudaMemcpy(h_pinned_staging, d_atlas_mean, payload_bytes, cudaMemcpyDeviceToHost));
    FILE* fp_mean = std::fopen(mean_path, "wb");
    if (fp_mean) {
        std::fwrite(&ref_header, 1, sizeof(VolumeHeader), fp_mean);
        std::fwrite(h_pinned_staging, sizeof(voxel_t), num_voxels, fp_mean);
        std::fclose(fp_mean);
        std::printf("  * Exported Baseline Atlas Mean: %s\n", mean_path);
    }

    // Export baseline_atlas_std.bin
    char std_path[MAX_PATH];
    std::snprintf(std_path, sizeof(std_path), "%s\\baseline_atlas_std.bin", output_dir.c_str());
    CHECK_CUDA(cudaMemcpy(h_pinned_staging, d_atlas_std, payload_bytes, cudaMemcpyDeviceToHost));
    FILE* fp_std = std::fopen(std_path, "wb");
    if (fp_std) {
        std::fwrite(&ref_header, 1, sizeof(VolumeHeader), fp_std);
        std::fwrite(h_pinned_staging, sizeof(voxel_t), num_voxels, fp_std);
        std::fclose(fp_std);
        std::printf("  * Exported Baseline Atlas Std Dev: %s\n", std_path);
    }

    // Write baseline atlas manifest
    std::string atlas_manifest_path = output_dir + "\\baseline_atlas_manifest.txt";
    std::ofstream atlas_manifest(atlas_manifest_path.c_str());
    if (atlas_manifest.is_open()) {
        atlas_manifest << mean_path << "\n";
        atlas_manifest << std_path << "\n";
        atlas_manifest.close();
        std::printf("  * Saved Baseline Atlas Manifest: %s\n", atlas_manifest_path.c_str());
    }

    // ------------------------------------------------------------------------
    // Step 2: Compute All Combinatorial Differences (nC2 pairs)
    // ------------------------------------------------------------------------
    size_t total_pairs = N * (N - 1) / 2;
    std::printf("\n[PROCESS] Generating all nC2 = %zu combinatorial baseline differences in VRAM...\n", total_pairs);

    std::string diff_manifest_path = output_dir + "\\baseline_diff_manifest.txt";
    std::ofstream diff_manifest(diff_manifest_path.c_str());

    size_t pair_idx = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            pair_idx++;

            AbsDiffKernel<<<numBlocks, blockSize>>>(d_volumes[i], d_volumes[j], d_diff_out, num_voxels);
            CHECK_CUDA(cudaGetLastError());

            CHECK_CUDA(cudaMemcpy(h_pinned_staging, d_diff_out, payload_bytes, cudaMemcpyDeviceToHost));

            char out_filename[MAX_PATH];
            std::snprintf(out_filename, sizeof(out_filename), "%s\\baseline_diff_pair_%zu_%zu.bin", output_dir.c_str(), i, j);

            FILE* out_fp = std::fopen(out_filename, "wb");
            if (out_fp) {
                std::fwrite(&ref_header, 1, sizeof(VolumeHeader), out_fp);
                std::fwrite(h_pinned_staging, sizeof(voxel_t), num_voxels, out_fp);
                std::fclose(out_fp);
            }

            if (diff_manifest.is_open()) {
                diff_manifest << out_filename << "\n";
            }

            std::printf("  [PAIR %zu/%zu] Generated: baseline_diff_pair_%zu_%zu.bin\n", pair_idx, total_pairs, i, j);
        }
    }

    if (diff_manifest.is_open()) {
        diff_manifest.close();
        std::printf("[MANIFEST] Saved baseline difference manifest: %s\n", diff_manifest_path.c_str());
    }

    // Free resources
    for (size_t i = 0; i < N; ++i) {
        if (d_volumes[i]) cudaFree(d_volumes[i]);
    }
    if (d_vol_ptrs) cudaFree(d_vol_ptrs);
    if (d_atlas_mean) cudaFree(d_atlas_mean);
    if (d_atlas_std) cudaFree(d_atlas_std);
    if (d_diff_out) cudaFree(d_diff_out);
    if (h_pinned_staging) cudaFreeHost(h_pinned_staging);

    std::printf("\n[COMPLETE] Statistical Baseline Atlas and all %zu difference maps generated successfully!\n", total_pairs);
    return EXIT_SUCCESS;
}
