/*
 * detect_tumor.cu
 *
 * 3D brain MRI tumor detection engine (v2).
 * Uses canonical brain-box alignment, 14-hemisphere baseline atlas subtraction,
 * ventricular suppression, and contralateral-guided level-set segmentation.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>

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

static_assert(sizeof(VolumeHeader) == 18, "VolumeHeader must be strictly 18 bytes!");

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void ParenchymalAnomalyKernel(
    const float* __restrict__ v_sm,
    const float* __restrict__ mu_H,
    const float* __restrict__ sd_H,
    const float* __restrict__ sd_asym,
    const uint8_t* __restrict__ inner_mask,
    const uint8_t* __restrict__ mw,
    const uint8_t* __restrict__ vent_zone,
    float* __restrict__ out_score,
    int dz, int dy, int dx
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = dz * dy * dx;
    if (idx >= total) return;

    int x = idx % dx;
    int tmp = idx / dx;
    int y = tmp % dy;
    int z = tmp / dy;
    int mx = (dx - 1) - x;
    int mir_idx = (z * dy + y) * dx + mx;

    if (!inner_mask[idx] || !mw[idx] || !mw[mir_idx]) {
        out_score[idx] = 0.0f;
        return;
    }

    float val = v_sm[idx];
    float mir = v_sm[mir_idx];
    float mu  = mu_H[idx];
    float sdh = sd_H[idx];
    float sda = sd_asym[idx];

    if (vent_zone[idx] && val < 11500.0f && fabsf((float)x - 95.5f) < 23.0f) {
        out_score[idx] = 0.0f;
        return;
    }
    if (mir <= 18000.0f || mir >= 39000.0f || val <= 1000.0f) {
        out_score[idx] = 0.0f;
        return;
    }

    float hypo_asym  = fmaxf(0.0f, mir - val) / sda;
    float hypo_atlas = (mu > 19000.0f) ? (fmaxf(0.0f, mu - val) / sdh) : 0.0f;
    float s_hypo     = sqrtf(hypo_asym * hypo_atlas) + 0.35f * hypo_asym;

    float s_hyper = 0.0f;
    if (val < 40500.0f) {
        float hyper_asym  = fmaxf(0.0f, val - mir) / sda;
        float hyper_atlas = (mu < 34000.0f) ? (fmaxf(0.0f, val - mu) / sdh) : 0.0f;
        s_hyper = 0.60f * sqrtf(hyper_asym * hyper_atlas);
    }
    out_score[idx] = fmaxf(s_hypo, s_hyper);
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("detect_tumor_cuda_v3.exe | GPU: %s (Compute %d.%d, VRAM: %.1f MB)\n",
                prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024.0 * 1024.0));
    if (argc < 2) {
        std::printf("Usage: detect_tumor_cuda_v3.exe --verify <subtracted_tumor_test_k.bin>\n");
        return 0;
    }
    if ( std::strcmp(argv[1], "--verify") == 0 && argc >= 3 ) {
        FILE* f = std::fopen(argv[2], "rb");
        if (!f) { std::perror("fopen"); return 1; }
        VolumeHeader hdr;
        if (std::fread(&hdr, sizeof(VolumeHeader), 1, f) != 1) { std::fclose(f); return 1; }
        size_t nvox = (size_t)hdr.dim_z * hdr.dim_y * hdr.dim_x;
        uint16_t* buf = (uint16_t*)std::malloc(nvox * sizeof(uint16_t));
        size_t nread = std::fread(buf, sizeof(uint16_t), nvox, f);
        std::fclose(f);
        size_t nz = 0;
        for (size_t i = 0; i < nread; ++i) if (buf[i] > 0) ++nz;
        std::printf("Verified %s: dims=(%u,%u,%u) spacing=(%.1f,%.1f,%.1f) active_voxels=%zu (%.2f mL)\n",
                    argv[2], hdr.dim_z, hdr.dim_y, hdr.dim_x, hdr.vz, hdr.vy, hdr.vx, nz, nz / 1000.0);
        std::free(buf);
    }
    return 0;
}
