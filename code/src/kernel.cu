#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d12.h>
#include <dxgi.h>
#include <dxgi1_6.h>
#include <dstorage.h>
#include <wrl/client.h>

#include <iostream>
#include <string>
#include <cstdint>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dstorage.lib")

using Microsoft::WRL::ComPtr;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "[CUDA Error] " << cudaGetErrorString(err)           \
                      << " at line " << __LINE__ << "\n";                     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define HR_CHECK(hr, msg)                                                      \
    do {                                                                       \
        HRESULT _hr = (hr);                                                    \
        if (FAILED(_hr)) {                                                     \
            std::cerr << "[FAIL] " << msg << " (HRESULT: 0x"                   \
                      << std::hex << _hr << std::dec << ") at line "          \
                      << __LINE__ << "\n";                                    \
            return 1;                                                          \
        }                                                                      \
    } while (0)

const int NUM_COPIES = 32;
const size_t SECTOR_SIZE = 4096;

// -----------------------------------------------------------------------------
// 1. Raw Binary Dimensions: Strictly uint16_t (6 Bytes Total)
// -----------------------------------------------------------------------------
#pragma pack(push, 1)
struct VolumeDimensionsU16 {
	uint16_t dim_z;
	uint16_t dim_y;
	uint16_t dim_x;
};
#pragma pack(pop)

// -----------------------------------------------------------------------------
// 2. 3D View Struct: Dynamic multi-dimensional array access [z][y][x]
// -----------------------------------------------------------------------------
template <typename T>
struct Dynamic3DView {
	T* data;
	uint32_t dim_x;
	uint32_t slice_stride;

	__host__ __device__ Dynamic3DView(T* raw, uint16_t y, uint16_t x)
		: data(raw), dim_x(x), slice_stride((uint32_t)y* x) {
	}

	struct RowProxy {
		T* row_ptr;
		__device__ __forceinline__ T& operator[](uint16_t x) { return row_ptr[x]; }
	};

	struct SliceProxy {
		T* slice_ptr;
		uint32_t dim_x;
		__device__ __forceinline__ RowProxy operator[](uint16_t y) {
			return RowProxy{ slice_ptr + ((size_t)y * dim_x) };
		}
	};

	__device__ __forceinline__ SliceProxy operator[](uint16_t z) {
		return SliceProxy{ data + ((size_t)z * slice_stride), dim_x };
	}
};

// -----------------------------------------------------------------------------
// 3. CUDA Kernel: In-place voxel halving
// -----------------------------------------------------------------------------
__global__ void divideVolumeByTwoKernel(uint16_t* __restrict__ d_flat, uint16_t dim_z, uint16_t dim_y, uint16_t dim_x) {
	uint16_t x = blockDim.x * blockIdx.x + threadIdx.x;
	uint16_t y = blockDim.y * blockIdx.y + threadIdx.y;
	uint16_t z = blockDim.z * blockIdx.z + threadIdx.z;

	Dynamic3DView<uint16_t> volume(d_flat, dim_y, dim_x);

	if (x < dim_x && y < dim_y && z < dim_z) {
		volume[z][y][x] = volume[z][y][x] >> 1;
	}
}

// -----------------------------------------------------------------------------
// 4. Path Helpers: Convert relative std::string to canonical absolute std::wstring
// -----------------------------------------------------------------------------
std::wstring toAbsoluteWidePath(const std::string& str) {
	if (str.empty()) return std::wstring();

	int sizeNeeded = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), nullptr, 0);
	std::wstring wstr(sizeNeeded, 0);
	MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], sizeNeeded);

	DWORD bufferLen = GetFullPathNameW(wstr.c_str(), 0, nullptr, nullptr);
	if (bufferLen == 0) return wstr;

	std::wstring fullPath(bufferLen, 0);
	GetFullPathNameW(wstr.c_str(), bufferLen, &fullPath[0], nullptr);
	fullPath.resize(bufferLen - 1); // remove null terminator

	return fullPath;
}

// Reads raw 6-byte dimension header (0 dynamic heap bytes)
bool readBinaryDimensionsU16(const std::string& path, uint16_t& dim_z, uint16_t& dim_y, uint16_t& dim_x) {
	HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE) return false;

	VolumeDimensionsU16 dims = {};
	DWORD bytesRead = 0;
	BOOL ok = ReadFile(hFile, &dims, sizeof(VolumeDimensionsU16), &bytesRead, NULL);
	CloseHandle(hFile);

	if (!ok || bytesRead != sizeof(VolumeDimensionsU16)) return false;

	dim_z = dims.dim_z;
	dim_y = dims.dim_y;
	dim_x = dims.dim_x;
	return (dim_z > 0 && dim_y > 0 && dim_x > 0);
}

// -----------------------------------------------------------------------------
// 5. Adapter Selection: Explicitly bind D3D12 to NVIDIA GPU (Vendor ID 0x10DE)
// -----------------------------------------------------------------------------
ComPtr<IDXGIAdapter1> GetNvidiaAdapter() {
	ComPtr<IDXGIFactory4> dxgiFactory;
	CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory));

	ComPtr<IDXGIAdapter1> adapter;
	ComPtr<IDXGIAdapter1> selectedAdapter;

	for (UINT i = 0; dxgiFactory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
		DXGI_ADAPTER_DESC1 desc;
		adapter->GetDesc1(&desc);

		if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
			if (desc.VendorId == 0x10DE) { // NVIDIA Vendor ID
				std::wcout << L"[INFO] Bound D3D12 to NVIDIA GPU: " << desc.Description << L"\n";
				return adapter;
			}
			if (!selectedAdapter) {
				selectedAdapter = adapter;
			}
		}
	}
	return selectedAdapter;
}

int main() {
	// Normal string paths
	const std::string dim_bin_path = "image_dim.bin";
	const std::string in_vol_path = "image_data.bin";
	const std::string out_vol_path = "output_data.bin";

	// 1. Read binary dimensions
	uint16_t dim_z = 0, dim_y = 0, dim_x = 0;
	if (!readBinaryDimensionsU16(dim_bin_path, dim_z, dim_y, dim_x)) {
		std::cerr << "[FAIL] Could not read dimension file: " << dim_bin_path << "\n";
		return 1;
	}

	const size_t raw_volume_bytes = (size_t)dim_z * dim_y * dim_x * sizeof(uint16_t);
	const size_t aligned_bytes = (raw_volume_bytes + SECTOR_SIZE - 1) & ~(SECTOR_SIZE - 1);

	std::cout << "Geometry: [" << dim_z << " x " << dim_y << " x " << dim_x << "]\n";
	std::cout << "Raw Bytes: " << raw_volume_bytes << " | Sector-Aligned: " << aligned_bytes << "\n";
	std::cout << "Allocating VRAM for 16 copies: "
		<< ((aligned_bytes * NUM_COPIES) / (1024 * 1024)) << " MB\n";

	// 2. Initialize Direct3D 12 & DirectStorage
	ComPtr<IDXGIAdapter1> adapter = GetNvidiaAdapter();
	ComPtr<ID3D12Device> d3d12Device;
	HR_CHECK(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3d12Device)),
			 "D3D12CreateDevice");

	ComPtr<IDStorageFactory> dsFactory;
	HR_CHECK(DStorageGetFactory(IID_PPV_ARGS(&dsFactory)), "DStorageGetFactory");

	DSTORAGE_QUEUE_DESC queueDesc = {};
	queueDesc.Capacity = DSTORAGE_MIN_QUEUE_CAPACITY;
	queueDesc.Priority = DSTORAGE_PRIORITY_NORMAL;
	queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
	queueDesc.Device = d3d12Device.Get();

	ComPtr<IDStorageQueue> dsQueue;
	HR_CHECK(dsFactory->CreateQueue(&queueDesc, IID_PPV_ARGS(&dsQueue)), "CreateQueue");

	// 3. Allocate Shared Ingest Buffer in VRAM
	D3D12_HEAP_PROPERTIES heapProps = {};
	heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

	D3D12_RESOURCE_DESC resDesc = {};
	resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	resDesc.Width = aligned_bytes;
	resDesc.Height = 1;
	resDesc.DepthOrArraySize = 1;
	resDesc.MipLevels = 1;
	resDesc.Format = DXGI_FORMAT_UNKNOWN;
	resDesc.SampleDesc.Count = 1;
	resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

	ComPtr<ID3D12Resource> vramResource;
	HR_CHECK(d3d12Device->CreateCommittedResource(
		&heapProps,
		D3D12_HEAP_FLAG_SHARED,
		&resDesc,
		D3D12_RESOURCE_STATE_COMMON,
		nullptr,
		IID_PPV_ARGS(&vramResource)
	), "CreateCommittedResource");

	SECURITY_ATTRIBUTES sa = {};
	sa.nLength = sizeof(SECURITY_ATTRIBUTES);
	sa.bInheritHandle = FALSE;

	HANDLE sharedHandle = nullptr;
	HR_CHECK(d3d12Device->CreateSharedHandle(
		vramResource.Get(),
		&sa,
		GENERIC_ALL,
		nullptr,
		&sharedHandle
	), "CreateSharedHandle");

	// 4. Map Shared VRAM Buffer into CUDA
	cudaExternalMemoryHandleDesc extMemDesc = {};
	extMemDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
	extMemDesc.handle.win32.handle = sharedHandle;
	extMemDesc.size = aligned_bytes;
	extMemDesc.flags = cudaExternalMemoryDedicated;

	cudaExternalMemory_t extMem;
	CUDA_CHECK(cudaImportExternalMemory(&extMem, &extMemDesc));

	cudaExternalMemoryBufferDesc bufDesc = {};
	bufDesc.offset = 0;
	bufDesc.size = aligned_bytes;

	uint16_t* d_volume_src = nullptr;
	CUDA_CHECK(cudaExternalMemoryGetMappedBuffer((void**)&d_volume_src, extMem, &bufDesc));

	// 5. DirectStorage Ingestion via Canonical Absolute Path
	std::wstring in_vol_wide = toAbsoluteWidePath(in_vol_path);
	std::wcout << L"[INFO] Opening file via DirectStorage: " << in_vol_wide << L"\n";

	DWORD fileAttr = GetFileAttributesW(in_vol_wide.c_str());
	if (fileAttr == INVALID_FILE_ATTRIBUTES) {
		std::wcerr << L"[ERROR] Input file does not exist at path: " << in_vol_wide << L"\n";
		return 1;
	}

	ComPtr<IDStorageFile> dsInFile;
	HR_CHECK(dsFactory->OpenFile(in_vol_wide.c_str(), IID_PPV_ARGS(&dsInFile)), "OpenFile");

	DSTORAGE_REQUEST readReq = {};
	readReq.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
	readReq.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
	readReq.Source.File.Source = dsInFile.Get();
	readReq.Source.File.Offset = 0;
	readReq.Source.File.Size = (UINT32)raw_volume_bytes;
	readReq.Destination.Buffer.Resource = vramResource.Get();
	readReq.Destination.Buffer.Offset = 0;
	readReq.Destination.Buffer.Size = (UINT32)raw_volume_bytes;
	readReq.UncompressedSize = (UINT32)raw_volume_bytes;

	ComPtr<ID3D12Fence> fence;
	HR_CHECK(d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
	HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	UINT64 fenceValue = 1;

	dsQueue->EnqueueRequest(&readReq);
	fence->SetEventOnCompletion(fenceValue, fenceEvent);
	dsQueue->EnqueueSignal(fence.Get(), fenceValue);
	dsQueue->Submit();

	WaitForSingleObject(fenceEvent, INFINITE);
	fenceValue++;
	std::cout << "[INFO] DirectStorage DMA into VRAM complete.\n";

	// 6. Allocate 16 Independent VRAM Buffers and Duplicate
	uint16_t* d_copies[NUM_COPIES];
	for (int i = 0; i < NUM_COPIES; ++i) {
		CUDA_CHECK(cudaMalloc((void**)&d_copies[i], aligned_bytes));
		CUDA_CHECK(cudaMemcpy(d_copies[i], d_volume_src, aligned_bytes, cudaMemcpyDeviceToDevice));
	}
	std::cout << "[INFO] 16 VRAM copies created.\n";

	// 7. Compute on All Copies
	dim3 blockSize(16, 16, 4);
	dim3 gridSize((dim_x + blockSize.x - 1) / blockSize.x,
				  (dim_y + blockSize.y - 1) / blockSize.y,
				  (dim_z + blockSize.z - 1) / blockSize.z);

	for (int i = 0; i < NUM_COPIES; ++i) {
		divideVolumeByTwoKernel << <gridSize, blockSize >> > (d_copies[i], dim_z, dim_y, dim_x);
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	std::cout << "[INFO] Kernels executed successfully.\n";

	// 8. Store Output to NVMe via Pinned Sector-Aligned DMA Write
	HANDLE hOutFile = CreateFileA(
		out_vol_path.c_str(), GENERIC_WRITE, 0, NULL,
		CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, NULL
	);

	if (hOutFile != INVALID_HANDLE_VALUE) {
		uint16_t* h_pinned_out = nullptr;
		CUDA_CHECK(cudaMallocHost((void**)&h_pinned_out, aligned_bytes));
		CUDA_CHECK(cudaMemcpy(h_pinned_out, d_copies[0], aligned_bytes, cudaMemcpyDeviceToHost));

		DWORD bytesWritten = 0;
		BOOL ok = WriteFile(hOutFile, h_pinned_out, (DWORD)aligned_bytes, &bytesWritten, NULL);
		if (ok) {
			if (raw_volume_bytes != aligned_bytes) {
				LARGE_INTEGER exactSize;
				exactSize.QuadPart = raw_volume_bytes;
				SetFilePointerEx(hOutFile, exactSize, NULL, FILE_BEGIN);
				SetEndOfFile(hOutFile);
			}
			std::cout << "[SUCCESS] Exported computed volume to: " << out_vol_path
				<< " (" << raw_volume_bytes << " bytes)\n";
		}
		else {
			std::cerr << "[FAIL] WriteFile failed. Win32 Error: " << GetLastError() << "\n";
		}

		CUDA_CHECK(cudaFreeHost(h_pinned_out));
		CloseHandle(hOutFile);
	}
	else {
		std::cerr << "[FAIL] CreateFileA failed for " << out_vol_path
			<< " (Win32 Error: " << GetLastError() << ")\n";
	}

	// 9. Hold VRAM Allocations Active for 20 Seconds (For Nsight Profiling)
	std::cout << "[INFO] Holding 16 VRAM allocations alive for 20 seconds...\n";
	std::this_thread::sleep_for(std::chrono::seconds(20));

	// 10. Deallocate and Cleanup
	for (int i = 0; i < NUM_COPIES; ++i) {
		CUDA_CHECK(cudaFree(d_copies[i]));
	}
	CloseHandle(fenceEvent);
	CloseHandle(sharedHandle);
	CUDA_CHECK(cudaDestroyExternalMemory(extMem));

	std::cout << "[INFO] Exiting cleanly.\n";
	return 0;
}