#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cuda_runtime.h>

#define NUM_COPIES 1            // Set copy count for scalability testing
const size_t SECTOR_SIZE = 4096; // 4 KB physical NVMe sector alignment

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
struct VolumeDimensionsU16 {
	uint16_t dim_z;
	uint16_t dim_y;
	uint16_t dim_x;
};
#pragma pack(pop)

// Safe 1D linear compute kernel (bypasses 3D block & grid hardware ceilings)
__global__ void divideVolumeByTwoKernel(uint16_t* __restrict__ d_flat, size_t total_voxels) {
	size_t tid = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < total_voxels) {
		d_flat[tid] >>= 1;
	}
}

int main(int argc, char* argv[]) {
	const std::string in_path = argv[1];
	const std::string out_path = in_path + "_out.bin";

	// 1. Direct NVMe read: header inspection (4 KB sector alignment)
	HANDLE hIn = CreateFileA(in_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
							 OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (hIn == INVALID_HANDLE_VALUE) {
		std::cerr << "[FAIL] Could not open " << in_path << " (Win32 Error: " << GetLastError() << ")\n";
		return 1;
	}

	uint8_t* h_pinned_staging = nullptr;
	CUDA_CHECK(cudaMallocHost((void**)&h_pinned_staging, SECTOR_SIZE));

	DWORD bytesRead = 0;
	BOOL readOk = ReadFile(hIn, h_pinned_staging, (DWORD)SECTOR_SIZE, &bytesRead, NULL);
	if (!readOk || bytesRead < sizeof(VolumeDimensionsU16)) {
		std::cerr << "[FAIL] Failed to read initial sector from " << in_path << "\n";
		CUDA_CHECK(cudaFreeHost(h_pinned_staging));
		CloseHandle(hIn);
		return 1;
	}

	VolumeDimensionsU16 dims = *reinterpret_cast<VolumeDimensionsU16*>(h_pinned_staging);

	std::cout << "[INFO] Parsed Header Dimensions: "
		<< "dim_z = " << dims.dim_z << ", "
		<< "dim_y = " << dims.dim_y << ", "
		<< "dim_x = " << dims.dim_x << "\n";

	// Validation Guard: Ensure dimensions are not zero or corrupted
	if (dims.dim_z == 0 || dims.dim_y == 0 || dims.dim_x == 0) {
		std::cerr << "[FAIL] Invalid dimensions! One or more dimensions are 0.\n"
			<< "Verify that input_data.bin starts with 3x uint16_t (6 bytes).\n";
		CUDA_CHECK(cudaFreeHost(h_pinned_staging));
		CloseHandle(hIn);
		return 1;
	}

	const size_t total_voxels = (size_t)dims.dim_z * dims.dim_y * dims.dim_x;
	const size_t raw_vol_bytes = total_voxels * sizeof(uint16_t);
	const size_t total_file_bytes = sizeof(VolumeDimensionsU16) + raw_vol_bytes;
	const size_t aligned_read_bytes = (total_file_bytes + SECTOR_SIZE - 1) & ~(SECTOR_SIZE - 1);

	std::cout << "[INFO] Total voxels: " << total_voxels
		<< " | Raw volume size: " << (raw_vol_bytes / (1024 * 1024)) << " MB\n";
	std::cout << "[INFO] Allocating VRAM for " << NUM_COPIES << " copies: "
		<< ((raw_vol_bytes * NUM_COPIES) / (1024 * 1024)) << " MB\n";

	// 2. Re-allocate pinned staging buffer to full aligned file size and stream file from NVMe
	CUDA_CHECK(cudaFreeHost(h_pinned_staging));
	CUDA_CHECK(cudaMallocHost((void**)&h_pinned_staging, aligned_read_bytes));

	LARGE_INTEGER zero = { 0 };
	SetFilePointerEx(hIn, zero, NULL, FILE_BEGIN);
	readOk = ReadFile(hIn, h_pinned_staging, (DWORD)aligned_read_bytes, &bytesRead, NULL);
	CloseHandle(hIn);

	if (!readOk) {
		std::cerr << "[FAIL] Failed reading full volume data from disk.\n";
		CUDA_CHECK(cudaFreeHost(h_pinned_staging));
		return 1;
	}

	// Pointer offset: skip 6-byte header to point directly at uint16_t voxels
	void* h_voxel_src = h_pinned_staging + sizeof(VolumeDimensionsU16);

	// 3. Allocate device memory for all copies
	std::vector<uint16_t*> d_copies(NUM_COPIES, nullptr);
	for (int i = 0; i < NUM_COPIES; ++i) {
		CUDA_CHECK(cudaMalloc((void**)&d_copies[i], raw_vol_bytes));
	}

	// Direct PCIe DMA for copy 0 (~25 GB/s on PCIe Gen 4)
	CUDA_CHECK(cudaMemcpy(d_copies[0], h_voxel_src, raw_vol_bytes, cudaMemcpyHostToDevice));

	// High-speed Intra-VRAM duplication for remaining copies (~1.5 TB/s)
	for (int i = 1; i < NUM_COPIES; ++i) {
		CUDA_CHECK(cudaMemcpy(d_copies[i], d_copies[0], raw_vol_bytes, cudaMemcpyDeviceToDevice));
	}
	std::cout << "[INFO] " << NUM_COPIES << " copies populated in VRAM.\n";

	// 4. Compute on all N copies using 1D grid layout
	const int threadsPerBlock = 256;
	const int blocksPerGrid = (int)((total_voxels + threadsPerBlock - 1) / threadsPerBlock);

	for (int i = 0; i < NUM_COPIES; ++i) {
		divideVolumeByTwoKernel << <blocksPerGrid, threadsPerBlock >> > (d_copies[i], total_voxels);
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	std::cout << "[INFO] Kernels executed successfully on all " << NUM_COPIES << " copies.\n";

	// 5. Store copy 0 to disk with 6-byte header
	HANDLE hOut = CreateFileA(out_path.c_str(), GENERIC_WRITE, 0, NULL,
							  CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hOut != INVALID_HANDLE_VALUE) {
		DWORD written = 0;
		WriteFile(hOut, &dims, sizeof(VolumeDimensionsU16), &written, NULL);

		// Copy computed result back to pinned host buffer and flush
		CUDA_CHECK(cudaMemcpy(h_voxel_src, d_copies[0], raw_vol_bytes, cudaMemcpyDeviceToHost));
		WriteFile(hOut, h_voxel_src, (DWORD)raw_vol_bytes, &written, NULL);
		CloseHandle(hOut);
		std::cout << "[SUCCESS] Exported result: " << out_path << " (" << total_file_bytes << " bytes)\n";
	}
	else {
		std::cerr << "[FAIL] Failed to open " << out_path << " for writing.\n";
	}

	// 6. Hold VRAM active for 20 seconds for Task Manager / Nsight inspection
	std::cout << "[INFO] Holding VRAM active for 20 seconds...\n";
	std::this_thread::sleep_for(std::chrono::seconds(20));

	// 7. Cleanup
	CUDA_CHECK(cudaFreeHost(h_pinned_staging));
	for (int i = 0; i < NUM_COPIES; ++i) {
		CUDA_CHECK(cudaFree(d_copies[i]));
	}
	std::cout << "[INFO] Exiting cleanly.\n";
	return 0;
}