/*
 * calculate_baseline.cu
 * =====================
 * Deliverable 2: Combinatorial Baseline Engine (Classical DIP Only)
 *
 * Requirements:
 * - Load manifest text file listing paths to exported baseline .bin files.
 * - Load all N baseline volumes into GPU VRAM simultaneously.
 * - Single, reusable pinned host memory buffer (cudaHostAlloc / cudaMallocHost) for staging DMA transfers.
 * - Compute all unique nC2 pairs (|Img_i - Img_j|) entirely inside VRAM.
 * - Dual precision modes via compile-time flag:
 *     Option A: -DUSE_UINT16  -> uses 64-bit safe arithmetic llabs((int64_t)a - (int64_t)b) to prevent unsigned underflow.
 *     Option B: -DUSE_FLOAT32 -> uses single-precision fabsf(a - b). Default if neither specified.
 * - Output baseline_diff_pair_i_j.bin with 18-byte packed header:
 *     struct VolumeHeader {
 *         uint16_t dim_z;
 *         uint16_t dim_y;
 *         uint16_t dim_x;
 *         float vz;
 *         float vy;
 *         float vx;
 *     }; (#pragma pack(push, 1))
 * - Pure Win32 API directory creation (avoid <filesystem> or std::filesystem).
 * - Generates baseline_diff_manifest.txt in the output directory.
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

// Win32 recursive directory creation avoiding <filesystem>
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

// CUDA Kernels
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

int main(int argc, char* argv[]) {
    std::string manifest_path = "";
    std::string output_dir = "";

    if (argc >= 3) {
        manifest_path = argv[1];
        output_dir = argv[2];
    } else {
        std::cout << "=== Combinatorial Baseline Engine (Classical DIP) ===" << std::endl;
        std::cout << "Mode: " << DTYPE_NAME << " (Size of voxel_t: " << sizeof(voxel_t) << " bytes)" << std::endl;
        std::cout << "Enter baseline binaries manifest path: ";
        std::getline(std::cin, manifest_path);
        std::cout << "Enter output directory for baseline difference maps: ";
        std::getline(std::cin, output_dir);
    }

    if (!CreateDirectoryRecursive(output_dir)) {
        std::fprintf(stderr, "Warning/Error: Failed or already existing output directory: %s\n", output_dir.c_str());
    }

    std::vector<std::string> files = ReadManifest(manifest_path);
    size_t N = files.size();
    if (N < 2) {
        std::fprintf(stderr, "Error: Need at least 2 baseline images to compute combinations. Found: %zu\n", N);
        return EXIT_FAILURE;
    }

    std::printf("[INFO] Found %zu baseline volumes in manifest.\n", N);

    // Read header from first volume to determine geometry
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
    std::printf("[MEMORY] Single volume size in VRAM: %.2f MB\n", (double)payload_bytes / (1024.0 * 1024.0));
    std::printf("[MEMORY] Total VRAM for %zu volumes: %.2f MB\n", N, (double)(N * payload_bytes) / (1024.0 * 1024.0));

    // Allocate reusable pinned host buffer for staging DMA transfers
    voxel_t* h_pinned_staging = nullptr;
    CHECK_CUDA(cudaHostAlloc((void**)&h_pinned_staging, payload_bytes, cudaHostAllocDefault));

    // Simultaneously allocate VRAM for all N baseline volumes + 1 diff volume
    std::vector<voxel_t*> d_volumes(N, nullptr);
    for (size_t i = 0; i < N; ++i) {
        CHECK_CUDA(cudaMalloc((void**)&d_volumes[i], payload_bytes));
    }
    voxel_t* d_diff_out = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_diff_out, payload_bytes));

    // Stream all baseline files from disk through pinned buffer into VRAM
    for (size_t i = 0; i < N; ++i) {
        FILE* fp = std::fopen(files[i].c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "Error: Failed to open %s\n", files[i].c_str());
            return EXIT_FAILURE;
        }
        VolumeHeader cur_hdr;
        if (std::fread(&cur_hdr, 1, sizeof(VolumeHeader), fp) != sizeof(VolumeHeader)) {
            std::fprintf(stderr, "Error: Failed to read header for volume %zu\n", i);
            std::fclose(fp);
            return EXIT_FAILURE;
        }
        if (cur_hdr.dim_z != ref_header.dim_z || cur_hdr.dim_y != ref_header.dim_y || cur_hdr.dim_x != ref_header.dim_x) {
            std::fprintf(stderr, "Error: Volume %zu dimensions mismatch reference!\n", i);
            std::fclose(fp);
            return EXIT_FAILURE;
        }
        size_t read_count = std::fread(h_pinned_staging, sizeof(voxel_t), num_voxels, fp);
        if (read_count != num_voxels) {
            std::fprintf(stderr, "Error: Expected %zu voxels, read %zu from %s\n", num_voxels, read_count, files[i].c_str());
            std::fclose(fp);
            return EXIT_FAILURE;
        }
        std::fclose(fp);

        // Async DMA copy into device memory
        CHECK_CUDA(cudaMemcpy(d_volumes[i], h_pinned_staging, payload_bytes, cudaMemcpyHostToDevice));
        std::printf("[LOAD] (%zu/%zu) Streamed %s to VRAM\n", i + 1, N, files[i].c_str());
    }

    size_t total_pairs = N * (N - 1) / 2;
    std::printf("[PROCESS] Generating all nC2 = %zu combinatorial baseline difference pairs entirely in VRAM...\n", total_pairs);

    std::string diff_manifest_path = output_dir + "\\baseline_diff_manifest.txt";
    std::ofstream diff_manifest(diff_manifest_path.c_str());

    int blockSize = 256;
    int numBlocks = (int)((num_voxels + blockSize - 1) / blockSize);

    size_t pair_idx = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            pair_idx++;

            // Kernel execution in VRAM
            AbsDiffKernel<<<numBlocks, blockSize>>>(d_volumes[i], d_volumes[j], d_diff_out, num_voxels);
            CHECK_CUDA(cudaGetLastError());

            // DMA transfer result back to pinned host staging buffer
            CHECK_CUDA(cudaMemcpy(h_pinned_staging, d_diff_out, payload_bytes, cudaMemcpyDeviceToHost));

            // Write output file directly to disk
            char out_filename[MAX_PATH];
            std::snprintf(out_filename, sizeof(out_filename), "%s\\baseline_diff_pair_%zu_%zu.bin", output_dir.c_str(), i, j);

            FILE* out_fp = std::fopen(out_filename, "wb");
            if (!out_fp) {
                std::fprintf(stderr, "Error: Failed to create output file %s\n", out_filename);
                continue;
            }
            std::fwrite(&ref_header, 1, sizeof(VolumeHeader), out_fp);
            std::fwrite(h_pinned_staging, sizeof(voxel_t), num_voxels, out_fp);
            std::fclose(out_fp);

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

    // Graceful cleanup
    for (size_t i = 0; i < N; ++i) {
        if (d_volumes[i]) cudaFree(d_volumes[i]);
    }
    if (d_diff_out) cudaFree(d_diff_out);
    if (h_pinned_staging) cudaFreeHost(h_pinned_staging);

    std::printf("[COMPLETE] All %zu baseline difference maps computed and written to disk successfully!\n", total_pairs);
    return EXIT_SUCCESS;
}
