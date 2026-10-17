#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <filesystem>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "[CUDA Error] " << cudaGetErrorString(err)           \
                      << " at line " << __LINE__ << "\n";                     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

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

// 16-bit voxel-wise absolute difference kernel
__global__ void absDiffKernelU16(const uint16_t* __restrict__ imgA,
								 const uint16_t* __restrict__ imgB,
								 uint16_t* __restrict__ diffOut,
								 size_t total_voxels) {
	size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < total_voxels) {
		int valA = (int)imgA[idx];
		int valB = (int)imgB[idx];
		diffOut[idx] = (uint16_t)abs(valA - valB);
	}
}

bool readHeader(const std::string& path, VolumeHeader& hdr) {
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) return false;
	file.read(reinterpret_cast<char*>(&hdr), sizeof(VolumeHeader));
	return file.good();
}

int main() {
	std::string baseline_list_path;
	std::string output_dir;

	std::cout << "Enter path to baseline images data list file: ";
	std::cin >> baseline_list_path;
	std::cout << "Enter output directory for baseline differences: ";
	std::cin >> output_dir;

	// 1. Ensure output directory exists
	try {
		fs::create_directories(output_dir);
	}
	catch (const std::exception& e) {
		std::cerr << "[ERROR] Could not create output directory: " << e.what() << "\n";
		return 1;
	}

	// 2. Parse file manifest
	std::ifstream list_file(baseline_list_path);
	if (!list_file.is_open()) {
		std::cerr << "[ERROR] Cannot open list file: " << baseline_list_path << "\n";
		return 1;
	}

	std::vector<std::string> file_paths;
	std::string line;
	while (std::getline(list_file, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();
		if (!line.empty() && line[0] != '#') file_paths.push_back(line);
	}

	const size_t N = file_paths.size();
	if (N < 2) {
		std::cerr << "[ERROR] At least 2 images are required to perform nC2 combinations. Found: " << N << "\n";
		return 1;
	}

	// 3. Inspect reference header
	VolumeHeader ref_hdr;
	if (!readHeader(file_paths[0], ref_hdr)) {
		std::cerr << "[ERROR] Failed to read header from: " << file_paths[0] << "\n";
		return 1;
	}

	const size_t total_voxels = (size_t)ref_hdr.dim_z * ref_hdr.dim_y * ref_hdr.dim_x;
	const size_t raw_voxel_bytes = total_voxels * sizeof(uint16_t);
	const size_t total_pairs = (N * (N - 1)) / 2;

	std::cout << "[INFO] Geometry: [" << ref_hdr.dim_z << " x " << ref_hdr.dim_y << " x " << ref_hdr.dim_x << "]\n";
	std::cout << "[INFO] Total voxels per volume: " << total_voxels << " (" << (raw_voxel_bytes / (1024 * 1024)) << " MB)\n";
	std::cout << "[INFO] Loading all " << N << " images simultaneously into VRAM ("
		<< ((raw_voxel_bytes * N) / (1024 * 1024)) << " MB total VRAM)...\n";
	std::cout << "[INFO] Total nC2 combinations to compute: " << total_pairs << "\n";

	// 4. Allocate reusable pinned host staging buffer (minimal host memory usage)
	uint8_t* h_pinned_staging = nullptr;
	CUDA_CHECK(cudaHostAlloc((void**)&h_pinned_staging, raw_voxel_bytes, cudaHostAllocDefault));

	// 5. Load all N images into VRAM simultaneously
	std::vector<uint16_t*> d_images(N, nullptr);
	for (size_t i = 0; i < N; ++i) {
		CUDA_CHECK(cudaMalloc((void**)&d_images[i], raw_voxel_bytes));

		std::ifstream f(file_paths[i], std::ios::binary);
		if (!f.is_open()) {
			std::cerr << "[ERROR] Cannot open file: " << file_paths[i] << "\n";
			return 1;
		}

		// Skip 18-byte header and read voxel payload
		f.seekg(sizeof(VolumeHeader));
		f.read(reinterpret_cast<char*>(h_pinned_staging), raw_voxel_bytes);
		f.close();

		// Direct DMA transfer to VRAM
		CUDA_CHECK(cudaMemcpy(d_images[i], h_pinned_staging, raw_voxel_bytes, cudaMemcpyHostToDevice));
		std::cout << "  [VRAM LOADED] Image " << i << ": " << file_paths[i] << "\n";
	}

	// 6. Allocate single output VRAM buffer for difference result
	uint16_t* d_diff = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&d_diff, raw_voxel_bytes));

	const int blockSize = 256;
	const int gridSize = (int)((total_voxels + blockSize - 1) / blockSize);

	// 7. Perform all nC2 difference operations
	size_t pair_idx = 0;
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = i + 1; j < N; ++j) {
			// Launch absolute difference kernel between image i and image j
			absDiffKernelU16 << <gridSize, blockSize >> > (d_images[i], d_images[j], d_diff, total_voxels);
			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());

			// Transfer result to pinned host memory
			CUDA_CHECK(cudaMemcpy(h_pinned_staging, d_diff, raw_voxel_bytes, cudaMemcpyDeviceToHost));

			// Construct output filename including pair indices
			std::string out_filename = "baseline_diff_pair_" + std::to_string(i) + "_" + std::to_string(j) + ".bin";
			std::string out_path = (fs::path(output_dir) / out_filename).string();

			// Write 18-byte header followed by voxel payload
			std::ofstream fout(out_path, std::ios::binary);
			if (!fout.is_open()) {
				std::cerr << "[ERROR] Cannot write to file: " << out_path << "\n";
				continue;
			}

			fout.write(reinterpret_cast<const char*>(&ref_hdr), sizeof(VolumeHeader));
			fout.write(reinterpret_cast<const char*>(h_pinned_staging), raw_voxel_bytes);
			fout.close();

			std::cout << "[PROGRESS " << ++pair_idx << "/" << total_pairs << "] Computed |Img"
				<< i << " - Img" << j << "| -> " << out_path << "\n";
		}
	}

	// 8. Cleanup
	for (size_t i = 0; i < N; ++i) {
		CUDA_CHECK(cudaFree(d_images[i]));
	}
	CUDA_CHECK(cudaFree(d_diff));
	CUDA_CHECK(cudaFreeHost(h_pinned_staging));

	std::cout << "[SUCCESS] Computed and saved all " << total_pairs << " baseline combinations.\n";
	return 0;
}