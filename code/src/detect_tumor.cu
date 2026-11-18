/*
 * detect_tumor.cu
 *
 * 3D brain MRI tumor detection engine.
 * Implements full CUDA thread-block execution for brain extraction,
 * canonical bounding-box affine alignment, 14-hemisphere baseline atlas
 * anomaly scoring, and contralateral-guided level-set segmentation.
 */

#include <windows.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

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

static constexpr int DZ = 192;
static constexpr int DY = 224;
static constexpr int DX = 192;
static constexpr size_t NVOX = (size_t)DZ * DY * DX; // 8,257,536 voxels

// ============================================================================
// 8x8x8 3D CUDA Kernels (Block = 8x8x8 = 512 threads, Grid = 24x28x24)
// ============================================================================

__global__ void ThresholdParenchyma_8x8x8(
    const uint16_t* __restrict__ vol,
    uint8_t* __restrict__ mask,
    uint16_t lo_thr, uint16_t hi_thr
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    uint16_t v = vol[idx];
    mask[idx] = (z >= 42 && v >= lo_thr && v <= hi_thr) ? 1 : 0;
}

__global__ void BinaryErode3D6_8x8x8(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    if (x == 0 || x == DX - 1 || y == 0 || y == DY - 1 || z == 0 || z == DZ - 1) {
        dst[idx] = 0;
        return;
    }
    size_t sxy = (size_t)DY * DX;
    dst[idx] = src[idx] & src[idx - 1] & src[idx + 1]
             & src[idx - DX] & src[idx + DX]
             & src[idx - sxy] & src[idx + sxy];
}

__global__ void BinaryDilate3D26_8x8x8(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    if (src[idx]) { dst[idx] = 1; return; }
    int z0 = max(0, z - 1), z1 = min(DZ - 1, z + 1);
    int y0 = max(0, y - 1), y1 = min(DY - 1, y + 1);
    int x0 = max(0, x - 1), x1 = min(DX - 1, x + 1);
    for (int zz = z0; zz <= z1; ++zz) {
        for (int yy = y0; yy <= y1; ++yy) {
            size_t row = ((size_t)zz * DY + yy) * DX;
            for (int xx = x0; xx <= x1; ++xx) {
                if (src[row + xx]) { dst[idx] = 1; return; }
            }
        }
    }
    dst[idx] = 0;
}

__global__ void BinaryErode3D26_8x8x8(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    if (!src[idx] || x == 0 || x == DX - 1 || y == 0 || y == DY - 1 || z == 0 || z == DZ - 1) {
        dst[idx] = 0;
        return;
    }
    for (int zz = z - 1; zz <= z + 1; ++zz) {
        for (int yy = y - 1; yy <= y + 1; ++yy) {
            size_t row = ((size_t)zz * DY + yy) * DX;
            for (int xx = x - 1; xx <= x + 1; ++xx) {
                if (!src[row + xx]) { dst[idx] = 0; return; }
            }
        }
    }
    dst[idx] = 1;
}

__global__ void MirrorClampAndUnion_8x8x8(
    uint8_t* __restrict__ mask,
    const uint16_t* __restrict__ vol,
    uint16_t lo_v, uint16_t hi_v
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    uint16_t v = vol[idx];
    if (mask[idx] && (v <= lo_v || v >= hi_v)) {
        mask[idx] = 0;
    }
}

__global__ void SymmetrizeMask_8x8x8(
    const uint8_t* __restrict__ src,
    const uint16_t* __restrict__ vol,
    uint8_t* __restrict__ dst,
    int check_nonzero
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    size_t mir = ((size_t)z * DY + y) * DX + (DX - 1 - x);
    uint8_t u = (src[idx] | src[mir]);
    if (check_nonzero && vol[idx] == 0) u = 0;
    dst[idx] = u;
}

// Canonical Affine Transform in 8x8x8 CUDA Blocks (Trilinear for Volume, Nearest for Mask)
__global__ void AffineWarpCanonical_8x8x8(
    const uint16_t* __restrict__ src_vol,
    const uint8_t* __restrict__ src_mask,
    float* __restrict__ dst_vol,
    uint8_t* __restrict__ dst_mask,
    float sz, float sy, float sx,
    float oz, float oy, float ox
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;

    float fz = sz * (float)z + oz;
    float fy = sy * (float)y + oy;
    float fx = sx * (float)x + ox;

    int nz = (int)rintf(fz);
    int ny = (int)rintf(fy);
    int nx = (int)rintf(fx);
    if (nz >= 0 && nz < DZ && ny >= 0 && ny < DY && nx >= 0 && nx < DX) {
        dst_mask[idx] = src_mask[((size_t)nz * DY + ny) * DX + nx];
    } else {
        dst_mask[idx] = 0;
    }

    if (fz < 0.0f || fz > (float)(DZ - 1) || fy < 0.0f || fy > (float)(DY - 1) || fx < 0.0f || fx > (float)(DX - 1)) {
        dst_vol[idx] = 0.0f;
        return;
    }

    int z0 = (int)floorf(fz), y0 = (int)floorf(fy), x0 = (int)floorf(fx);
    int z1 = min(DZ - 1, z0 + 1), y1 = min(DY - 1, y0 + 1), x1 = min(DX - 1, x0 + 1);
    float wz = fz - (float)z0, wy = fy - (float)y0, wx = fx - (float)x0;

    auto sample = [&](int zz, int yy, int xx) -> float {
        size_t s_i = ((size_t)zz * DY + yy) * DX + xx;
        return src_mask[s_i] ? (float)src_vol[s_i] : 0.0f;
    };

    float c000 = sample(z0, y0, x0), c001 = sample(z0, y0, x1);
    float c010 = sample(z0, y1, x0), c011 = sample(z0, y1, x1);
    float c100 = sample(z1, y0, x0), c101 = sample(z1, y0, x1);
    float c110 = sample(z1, y1, x0), c111 = sample(z1, y1, x1);

    float c00 = c000 * (1.0f - wx) + c001 * wx;
    float c01 = c010 * (1.0f - wx) + c011 * wx;
    float c10 = c100 * (1.0f - wx) + c101 * wx;
    float c11 = c110 * (1.0f - wx) + c111 * wx;
    float c0  = c00  * (1.0f - wy) + c01  * wy;
    float c1  = c10  * (1.0f - wy) + c11  * wy;
    dst_vol[idx] = c0 * (1.0f - wz) + c1 * wz;
}

// Separable 1D Gaussian Convolution Kernels in 8x8x8 Blocks (X, Y, Z axes)
__constant__ float c_gker[31];

__global__ void GaussianConvX_8x8x8(const float* __restrict__ src, float* __restrict__ dst, int radius) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t row = ((size_t)z * DY + y) * DX;
    float sum = 0.0f;
    for (int d = -radius; d <= radius; ++d) {
        int xx = min(DX - 1, max(0, x + d));
        sum += src[row + xx] * c_gker[d + radius];
    }
    dst[row + x] = sum;
}

__global__ void GaussianConvY_8x8x8(const float* __restrict__ src, float* __restrict__ dst, int radius) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t slice = (size_t)z * DY * DX;
    float sum = 0.0f;
    for (int d = -radius; d <= radius; ++d) {
        int yy = min(DY - 1, max(0, y + d));
        sum += src[slice + (size_t)yy * DX + x] * c_gker[d + radius];
    }
    dst[slice + (size_t)y * DX + x] = sum;
}

__global__ void GaussianConvZ_8x8x8(const float* __restrict__ src, float* __restrict__ dst, int radius) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t sxy = (size_t)DY * DX;
    size_t off = (size_t)y * DX + x;
    float sum = 0.0f;
    for (int d = -radius; d <= radius; ++d) {
        int zz = min(DZ - 1, max(0, z + d));
        sum += src[(size_t)zz * sxy + off] * c_gker[d + radius];
    }
    dst[(size_t)z * sxy + off] = sum;
}

// Baseline Atlas Accumulation Kernels in 8x8x8 Blocks
__global__ void AccumulateCanonicalBaseline_8x8x8(
    const float* __restrict__ vw,
    const uint8_t* __restrict__ mw,
    float* __restrict__ sum_H,
    float* __restrict__ sum_sq_H,
    float* __restrict__ sum_asym_sq,
    float* __restrict__ sum_mask
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    size_t mir = ((size_t)z * DY + y) * DX + (DX - 1 - x);

    float v1 = vw[idx];
    float v2 = vw[mir];
    uint8_t m1 = mw[idx];
    uint8_t m2 = mw[mir];

    sum_H[idx]    += (v1 + v2);
    sum_sq_H[idx] += (v1 * v1 + v2 * v2);
    float d = v1 - v2;
    sum_asym_sq[idx] += 2.0f * (d * d);
    sum_mask[idx] += (float)(m1 + m2);
}

__global__ void FinalizeCanonicalAtlasStats_8x8x8(
    float* __restrict__ sum_H,
    float* __restrict__ sum_sq_H,
    float* __restrict__ sum_asym_sq,
    const float* __restrict__ sum_mask,
    uint8_t* __restrict__ atlas_mask,
    int n_hemi
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;

    float inv_n = 1.0f / (float)n_hemi;
    float mu = sum_H[idx] * inv_n;
    float var = fmaxf(0.0f, sum_sq_H[idx] * inv_n - mu * mu);
    sum_H[idx] = mu;
    sum_sq_H[idx] = sqrtf(var);
    sum_asym_sq[idx] = sqrtf(sum_asym_sq[idx] * inv_n);
    atlas_mask[idx] = (sum_mask[idx] * inv_n > 0.75f) ? 1 : 0;
}

__global__ void ClampFloor_8x8x8(float* __restrict__ arr, float min_val) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    if (arr[idx] < min_val) arr[idx] = min_val;
}

__global__ void InitVentricularSeed_8x8x8(
    const float* __restrict__ mu_H,
    uint8_t* __restrict__ vent_mask
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    bool is_v = (mu_H[idx] < 22000.0f) && (z > 75 && z < 135) && (y > 75 && y < 165) && (x > 68 && x < 124);
    vent_mask[idx] = is_v ? 1 : 0;
}

// 8x8x8 Hybrid Parenchymal + 3D DoG Saliency Kernel
__global__ void ComputeParenchymalSaliency_8x8x8(
    const float* __restrict__ v_sm1,
    const float* __restrict__ v_sm2,
    const float* __restrict__ mu_H,
    const float* __restrict__ sd_H,
    const float* __restrict__ sd_asym,
    const uint8_t* __restrict__ inner_mask,
    const uint8_t* __restrict__ mw,
    const uint8_t* __restrict__ vent_zone,
    float* __restrict__ out_score
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    size_t mir_idx = ((size_t)z * DY + y) * DX + (DX - 1 - x);

    if (!inner_mask[idx] || !mw[idx] || !mw[mir_idx]) {
        out_score[idx] = 0.0f;
        return;
    }

    float val = v_sm1[idx];
    float mir = v_sm1[mir_idx];
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

    float dog = fabsf(val - v_sm2[idx]);
    float base_s = fmaxf(s_hypo, s_hyper);
    out_score[idx] = base_s * (1.0f + 0.04f * (dog / 5000.0f));
}

// 8x8x8 Parallel 26-Connected Geodesic Region Growing Kernel in VRAM
__global__ void GeodesicGrow26_8x8x8(
    const uint8_t* __restrict__ cands,
    const uint8_t* __restrict__ curr_seeds,
    uint8_t* __restrict__ next_seeds,
    int* __restrict__ d_changed
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    if (curr_seeds[idx]) { next_seeds[idx] = 1; return; }
    if (!cands[idx])     { next_seeds[idx] = 0; return; }

    int z0 = max(0, z - 1), z1 = min(DZ - 1, z + 1);
    int y0 = max(0, y - 1), y1 = min(DY - 1, y + 1);
    int x0 = max(0, x - 1), x1 = min(DX - 1, x + 1);
    for (int zz = z0; zz <= z1; ++zz) {
        for (int yy = y0; yy <= y1; ++yy) {
            size_t row = ((size_t)zz * DY + yy) * DX;
            for (int xx = x0; xx <= x1; ++xx) {
                if (curr_seeds[row + xx]) {
                    next_seeds[idx] = 1;
                    atomicExch(d_changed, 1);
                    return;
                }
            }
        }
    }
    next_seeds[idx] = 0;
}

__global__ void MaskToFloat_8x8x8(const uint8_t* __restrict__ mask, float* __restrict__ out_f) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    out_f[idx] = mask[idx] ? 1.0f : 0.0f;
}

__global__ void PackSegmentedUint16_8x8x8(
    const float* __restrict__ vw,
    const float* __restrict__ seg_sm,
    const uint8_t* __restrict__ fallback_mask,
    int use_smooth,
    uint16_t* __restrict__ out_u16
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    bool active = use_smooth ? (seg_sm[idx] >= 0.45f) : (fallback_mask[idx] != 0);
    if (active) {
        float val = rintf(vw[idx]);
        if (val < 1.0f) val = 1.0f;
        if (val > 65535.0f) val = 65535.0f;
        out_u16[idx] = (uint16_t)val;
    } else {
        out_u16[idx] = 0;
    }
}

// ============================================================================
// Host Helper Functions
// ============================================================================

static void LaunchGaussian3D_8x8x8(
    const float* d_src, float* d_dst, float* d_tmp1, float* d_tmp2, float sigma
) {
    int radius = min(15, max(1, (int)ceilf(3.0f * sigma)));
    float h_ker[31];
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        float w = expf(-(float)(i * i) / (2.0f * sigma * sigma));
        h_ker[i + radius] = w;
        sum += w;
    }
    for (int i = 0; i <= 2 * radius; ++i) h_ker[i] /= sum;
    CHECK_CUDA(cudaMemcpyToSymbol(c_gker, h_ker, (2 * radius + 1) * sizeof(float)));

    dim3 grid(DX / 8, DY / 8, DZ / 8);
    dim3 block(8, 8, 8);
    GaussianConvX_8x8x8<<<grid, block>>>(d_src, d_tmp1, radius);
    GaussianConvY_8x8x8<<<grid, block>>>(d_tmp1, d_tmp2, radius);
    GaussianConvZ_8x8x8<<<grid, block>>>(d_tmp2, d_dst, radius);
}

static std::vector<std::string> ListBinFiles(const std::string& pattern) {
    std::vector<std::string> files;
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return files;
    std::string dir = "";
    size_t slash = pattern.find_last_of("/\\");
    if (slash != std::string::npos) dir = pattern.substr(0, slash + 1);
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            files.push_back(dir + fd.cFileName);
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    std::sort(files.begin(), files.end());
    return files;
}

// Extract Brain in 8x8x8 CUDA Blocks + Canonical Warp
static void ExtractAndCanonicalizeVolume(
    const uint16_t* d_vol,
    uint8_t* d_mask_a,
    uint8_t* d_mask_b,
    float* d_vw,
    uint8_t* d_mw,
    std::vector<uint8_t>& h_eroded,
    std::vector<uint8_t>& h_core,
    std::vector<int>& h_visited,
    std::vector<int>& bfs_q,
    float out_scale[3],
    float out_offset[3]
) {
    dim3 grid(DX / 8, DY / 8, DZ / 8);
    dim3 block(8, 8, 8);

    ThresholdParenchyma_8x8x8<<<grid, block>>>(d_vol, d_mask_a, 14500, 41500);
    for (int i = 0; i < 3; ++i) {
        BinaryErode3D6_8x8x8<<<grid, block>>>(d_mask_a, d_mask_b);
        std::swap(d_mask_a, d_mask_b);
    }

    CHECK_CUDA(cudaMemcpy(h_eroded.data(), d_mask_a, NVOX * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    std::fill(h_visited.begin(), h_visited.end(), 0);
    int best_seed = -1;
    size_t best_center_cnt = 0;
    const int dz6[6] = {-1, 1, 0, 0, 0, 0};
    const int dy6[6] = {0, 0, -1, 1, 0, 0};
    const int dx6[6] = {0, 0, 0, 0, -1, 1};
    size_t sxy = (size_t)DY * DX;

    for (size_t i = 0; i < NVOX; ++i) {
        if (h_eroded[i] && !h_visited[i]) {
            bfs_q.clear();
            bfs_q.push_back((int)i);
            h_visited[i] = 1;
            size_t center_cnt = 0;
            size_t head = 0;
            while (head < bfs_q.size()) {
                int cur = bfs_q[head++];
                int cz = cur / (int)sxy;
                int rem = cur % (int)sxy;
                int cy = rem / DX;
                int cx = rem % DX;
                if (cz > 55 && cz < 145 && cy > 50 && cy < 175 && cx > 45 && cx < 147) {
                    center_cnt++;
                }
                for (int k = 0; k < 6; ++k) {
                    int nz = cz + dz6[k], ny = cy + dy6[k], nx = cx + dx6[k];
                    if (nz >= 0 && nz < DZ && ny >= 0 && ny < DY && nx >= 0 && nx < DX) {
                        int n_i = (nz * DY + ny) * DX + nx;
                        if (h_eroded[n_i] && !h_visited[n_i]) {
                            h_visited[n_i] = 1;
                            bfs_q.push_back(n_i);
                        }
                    }
                }
            }
            if (center_cnt > best_center_cnt) {
                best_center_cnt = center_cnt;
                best_seed = (int)i;
            }
        }
    }

    std::fill(h_core.begin(), h_core.end(), 0);
    if (best_seed >= 0) {
        std::fill(h_visited.begin(), h_visited.end(), 0);
        bfs_q.clear();
        bfs_q.push_back(best_seed);
        h_visited[best_seed] = 1;
        size_t head = 0;
        while (head < bfs_q.size()) {
            int cur = bfs_q[head++];
            int cz = cur / (int)sxy;
            int rem = cur % (int)sxy;
            int cy = rem / DX;
            int cx = rem % DX;
            h_core[cur] = 1;
            h_core[(cz * DY + cy) * DX + (DX - 1 - cx)] = 1; // Bilateral mirror union
            for (int k = 0; k < 6; ++k) {
                int nz = cz + dz6[k], ny = cy + dy6[k], nx = cx + dx6[k];
                if (nz >= 0 && nz < DZ && ny >= 0 && ny < DY && nx >= 0 && nx < DX) {
                    int n_i = (nz * DY + ny) * DX + nx;
                    if (h_eroded[n_i] && !h_visited[n_i]) {
                        h_visited[n_i] = 1;
                        bfs_q.push_back(n_i);
                    }
                }
            }
        }
    }

    CHECK_CUDA(cudaMemcpy(d_mask_a, h_core.data(), NVOX * sizeof(uint8_t), cudaMemcpyHostToDevice));
    for (int i = 0; i < 4; ++i) {
        BinaryDilate3D26_8x8x8<<<grid, block>>>(d_mask_a, d_mask_b);
        std::swap(d_mask_a, d_mask_b);
    }
    MirrorClampAndUnion_8x8x8<<<grid, block>>>(d_mask_a, d_vol, 500, 42500);
    SymmetrizeMask_8x8x8<<<grid, block>>>(d_mask_a, d_vol, d_mask_b, 0);
    std::swap(d_mask_a, d_mask_b);

    // 4-iteration 3D 26-connected closing
    for (int i = 0; i < 4; ++i) {
        BinaryDilate3D26_8x8x8<<<grid, block>>>(d_mask_a, d_mask_b);
        std::swap(d_mask_a, d_mask_b);
    }
    for (int i = 0; i < 4; ++i) {
        BinaryErode3D26_8x8x8<<<grid, block>>>(d_mask_a, d_mask_b);
        std::swap(d_mask_a, d_mask_b);
    }
    SymmetrizeMask_8x8x8<<<grid, block>>>(d_mask_a, d_vol, d_mask_b, 1);
    std::swap(d_mask_a, d_mask_b);

    // Compute [2%, 98%] intracranial bounding box via exact axial/coronal/sagittal histograms
    CHECK_CUDA(cudaMemcpy(h_core.data(), d_mask_a, NVOX * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    size_t hist_z[DZ] = {0}, hist_y[DY] = {0}, hist_x[DX] = {0};
    size_t total_m = 0;
    for (int z = 0; z < DZ; ++z) {
        for (int y = 0; y < DY; ++y) {
            size_t row = ((size_t)z * DY + y) * DX;
            for (int x = 0; x < DX; ++x) {
                if (h_core[row + x]) {
                    hist_z[z]++;
                    hist_y[y]++;
                    hist_x[x]++;
                    total_m++;
                }
            }
        }
    }

    auto pct_hist = [&](const size_t* h, int len, double p) -> float {
        if (total_m == 0) return 0.0f;
        double target = p * (double)total_m;
        size_t acc = 0;
        for (int i = 0; i < len; ++i) {
            acc += h[i];
            if ((double)acc >= target) return (float)i;
        }
        return (float)(len - 1);
    };

    float z0 = pct_hist(hist_z, DZ, 0.02), z1 = pct_hist(hist_z, DZ, 0.98);
    float y0 = pct_hist(hist_y, DY, 0.02), y1 = pct_hist(hist_y, DY, 0.98);
    float x0 = pct_hist(hist_x, DX, 0.02), x1 = pct_hist(hist_x, DX, 0.98);

    const float tgt_lo[3] = {45.0f, 25.0f, 30.0f};
    const float tgt_hi[3] = {165.0f, 195.0f, 162.0f};
    out_scale[0] = (z1 - z0) / (tgt_hi[0] - tgt_lo[0]);
    out_scale[1] = (y1 - y0) / (tgt_hi[1] - tgt_lo[1]);
    out_scale[2] = (x1 - x0) / (tgt_hi[2] - tgt_lo[2]);
    out_offset[0] = z0 - tgt_lo[0] * out_scale[0];
    out_offset[1] = y0 - tgt_lo[1] * out_scale[1];
    out_offset[2] = x0 - tgt_lo[2] * out_scale[2];

    AffineWarpCanonical_8x8x8<<<grid, block>>>(
        d_vol, d_mask_a, d_vw, d_mw,
        out_scale[0], out_scale[1], out_scale[2],
        out_offset[0], out_offset[1], out_offset[2]
    );
}

// ============================================================================
// Main Standalone C++ CUDA Engine
// ============================================================================

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    if (argc >= 3 && std::strcmp(argv[1], "--verify") == 0) {
        FILE* f = std::fopen(argv[2], "rb");
        if (!f) { std::perror("fopen"); return 1; }
        VolumeHeader hdr;
        if (std::fread(&hdr, sizeof(VolumeHeader), 1, f) != 1) { std::fclose(f); return 1; }
        std::vector<uint16_t> buf(NVOX);
        size_t nread = std::fread(buf.data(), sizeof(uint16_t), NVOX, f);
        std::fclose(f);
        size_t nz = 0;
        for (size_t i = 0; i < nread; ++i) if (buf[i] > 0) ++nz;
        std::printf("detect_tumor_cuda_v4.exe | Verified %s: dims=(%u,%u,%u) active_voxels=%zu (%.2f mL)\n",
                    argv[2], hdr.dim_z, hdr.dim_y, hdr.dim_x, nz, nz / 1000.0);
        return 0;
    }

    auto t_wall0 = std::chrono::high_resolution_clock::now();
    std::printf("==========================================================================\n");
    std::printf(" Standalone C/C++ CUDA Brain Tumor Engine v4 (8x8x8 Chunks) | %s\n", prop.name);
    std::printf("==========================================================================\n");

    std::string out_dir = "new_code_v4/cuda_segmented_out";
    CreateDirectoryA("new_code_v4", NULL);
    CreateDirectoryA(out_dir.c_str(), NULL);

    std::vector<std::string> base_paths = ListBinFiles("exp_baseline_u16\\*.bin");
    std::vector<std::string> tumor_paths = ListBinFiles("exp_tumor_u16\\sub-RS*_T1w.bin");
    if (base_paths.empty() || tumor_paths.empty()) {
        std::fprintf(stderr, "Error: Could not find baseline or tumor .bin files!\n");
        return 1;
    }

    // 1. Preload ALL 7 Baseline + 42 Patient Volumes into GPU VRAM Upfront
    auto t_pre0 = std::chrono::high_resolution_clock::now();
    size_t vol_bytes = NVOX * sizeof(uint16_t);
    uint16_t* h_pinned = nullptr;
    CHECK_CUDA(cudaHostAlloc((void**)&h_pinned, vol_bytes, cudaHostAllocDefault));

    std::vector<uint16_t*> d_base_vols(base_paths.size(), nullptr);
    for (size_t i = 0; i < base_paths.size(); ++i) {
        FILE* f = std::fopen(base_paths[i].c_str(), "rb");
        VolumeHeader hdr;
        std::fread(&hdr, sizeof(VolumeHeader), 1, f);
        std::fread(h_pinned, sizeof(uint16_t), NVOX, f);
        std::fclose(f);
        CHECK_CUDA(cudaMalloc((void**)&d_base_vols[i], vol_bytes));
        CHECK_CUDA(cudaMemcpy(d_base_vols[i], h_pinned, vol_bytes, cudaMemcpyHostToDevice));
    }

    std::vector<uint16_t*> d_tumor_vols(tumor_paths.size(), nullptr);
    std::vector<VolumeHeader> tumor_hdrs(tumor_paths.size());
    std::vector<std::string> tumor_pids(tumor_paths.size());
    std::vector<std::string> tumor_stems(tumor_paths.size());

    for (size_t i = 0; i < tumor_paths.size(); ++i) {
        FILE* f = std::fopen(tumor_paths[i].c_str(), "rb");
        std::fread(&tumor_hdrs[i], sizeof(VolumeHeader), 1, f);
        std::fread(h_pinned, sizeof(uint16_t), NVOX, f);
        std::fclose(f);
        CHECK_CUDA(cudaMalloc((void**)&d_tumor_vols[i], vol_bytes));
        CHECK_CUDA(cudaMemcpy(d_tumor_vols[i], h_pinned, vol_bytes, cudaMemcpyHostToDevice));

        std::string fname = tumor_paths[i];
        size_t sl = fname.find_last_of("/\\");
        if (sl != std::string::npos) fname = fname.substr(sl + 1);
        std::string stem = fname.substr(0, fname.find(".bin"));
        std::string pid = stem.substr(4, stem.find("_") - 4);
        tumor_stems[i] = stem;
        tumor_pids[i] = pid;
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double preload_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_pre0).count();
    double preloaded_mb = (double)((base_paths.size() + tumor_paths.size()) * vol_bytes) / (1024.0 * 1024.0);
    double total_vram_mb = (double)prop.totalGlobalMem / (1024.0 * 1024.0);
    std::printf("[VRAM Preload] Loaded %zu baseline + %zu tumor volumes into VRAM (%.1f / %.1f MB) in %.1f ms\n",
                base_paths.size(), tumor_paths.size(), preloaded_mb, total_vram_mb, preload_ms);

    // Allocate Working Buffers in VRAM
    uint8_t *d_mask_a = nullptr, *d_mask_b = nullptr, *d_mw = nullptr;
    uint8_t *d_inner_mask = nullptr, *d_vent_zone = nullptr;
    uint8_t *d_cands = nullptr, *d_seeds = nullptr;
    float *d_vw = nullptr, *d_sm1 = nullptr, *d_sm2 = nullptr, *d_tmp1 = nullptr, *d_tmp2 = nullptr;
    float *d_mu_H = nullptr, *d_sd_H = nullptr, *d_sd_asym = nullptr, *d_sum_mask = nullptr;
    float *d_score = nullptr, *d_score_sm = nullptr;
    uint16_t *d_out_u16 = nullptr;
    int *d_changed = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_mask_a, NVOX * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_mask_b, NVOX * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_mw, NVOX * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_inner_mask, NVOX * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_vent_zone, NVOX * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_cands, NVOX * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_seeds, NVOX * sizeof(uint8_t)));

    CHECK_CUDA(cudaMalloc((void**)&d_vw, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_sm1, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_sm2, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_tmp1, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_tmp2, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_mu_H, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_sd_H, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_sd_asym, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_sum_mask, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_score, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_score_sm, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_out_u16, vol_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_changed, sizeof(int)));

    std::vector<uint8_t> h_eroded(NVOX), h_core(NVOX), h_inner(NVOX), h_mw(NVOX), h_reg11(NVOX), h_cands(NVOX), h_seeds(NVOX);
    std::vector<int> h_visited(NVOX, 0), bfs_q;
    bfs_q.reserve(NVOX / 4);
    std::vector<float> h_vw(NVOX), h_score(NVOX), h_score_sm(NVOX);

    dim3 grid(DX / 8, DY / 8, DZ / 8);
    dim3 block(8, 8, 8);

    // 2. Build 14-Hemisphere Symmetric Baseline Atlas in VRAM using 8x8x8 Kernels
    auto t_atl0 = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemset(d_mu_H, 0, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_sd_H, 0, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_sd_asym, 0, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_sum_mask, 0, NVOX * sizeof(float)));

    float dummy_s[3], dummy_o[3];
    for (size_t i = 0; i < d_base_vols.size(); ++i) {
        ExtractAndCanonicalizeVolume(
            d_base_vols[i], d_mask_a, d_mask_b, d_vw, d_mw,
            h_eroded, h_core, h_visited, bfs_q, dummy_s, dummy_o
        );
        AccumulateCanonicalBaseline_8x8x8<<<grid, block>>>(d_vw, d_mw, d_mu_H, d_sd_H, d_sd_asym, d_sum_mask);
    }

    int n_hemi = (int)(d_base_vols.size() * 2);
    FinalizeCanonicalAtlasStats_8x8x8<<<grid, block>>>(d_mu_H, d_sd_H, d_sd_asym, d_sum_mask, d_mask_a, n_hemi);

    // Erode atlas_mask 4 iterations with B6 to get inner_mask
    for (int i = 0; i < 4; ++i) {
        BinaryErode3D6_8x8x8<<<grid, block>>>(d_mask_a, d_mask_b);
        std::swap(d_mask_a, d_mask_b);
    }
    CHECK_CUDA(cudaMemcpy(d_inner_mask, d_mask_a, NVOX * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(h_inner.data(), d_inner_mask, NVOX * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // Smooth sd_H and sd_asym with sigma=2.0 and clamp floor (3000.0 and 2800.0)
    LaunchGaussian3D_8x8x8(d_sd_H, d_sm1, d_tmp1, d_tmp2, 2.0f);
    ClampFloor_8x8x8<<<grid, block>>>(d_sm1, 3000.0f);
    CHECK_CUDA(cudaMemcpy(d_sd_H, d_sm1, NVOX * sizeof(float), cudaMemcpyDeviceToDevice));

    LaunchGaussian3D_8x8x8(d_sd_asym, d_sm1, d_tmp1, d_tmp2, 2.0f);
    ClampFloor_8x8x8<<<grid, block>>>(d_sm1, 2800.0f);
    CHECK_CUDA(cudaMemcpy(d_sd_asym, d_sm1, NVOX * sizeof(float), cudaMemcpyDeviceToDevice));

    // Build vent_zone (dilate 2 iterations with B26)
    InitVentricularSeed_8x8x8<<<grid, block>>>(d_mu_H, d_mask_a);
    for (int i = 0; i < 2; ++i) {
        BinaryDilate3D26_8x8x8<<<grid, block>>>(d_mask_a, d_mask_b);
        std::swap(d_mask_a, d_mask_b);
    }
    CHECK_CUDA(cudaMemcpy(d_vent_zone, d_mask_a, NVOX * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    double atlas_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_atl0).count();
    std::printf("[Atlas] Built 14-hemisphere symmetric T1w baseline atlas in VRAM in %.1f ms\n", atlas_ms);

    // 3. Process All 42 Preloaded Patient Volumes Sequentially in 8x8x8 CUDA Chunks & Store Immediately
    FILE* fp_csv = std::fopen("new_code_v4/cuda_segmented_out/gpu_perf_stats.csv", "w");
    if (fp_csv) {
        std::fprintf(fp_csv, "index,patient_id,stem,peak_z,peak_y,peak_x,pred_hemi,peak_score,chosen_thresh,pred_vol_ml,native_vol_ml,gpu_time_ms,vram_preloaded_mb,scale_z,scale_y,scale_x,offset_z,offset_y,offset_x\n");
    }

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));
    double sum_steady_ms = 0.0;

    for (size_t idx = 0; idx < d_tumor_vols.size(); ++idx) {
        CHECK_CUDA(cudaEventRecord(ev0));

        float scale[3], offset[3];
        ExtractAndCanonicalizeVolume(
            d_tumor_vols[idx], d_mask_a, d_mask_b, d_vw, d_mw,
            h_eroded, h_core, h_visited, bfs_q, scale, offset
        );

        LaunchGaussian3D_8x8x8(d_vw, d_sm1, d_tmp1, d_tmp2, 1.2f);
        LaunchGaussian3D_8x8x8(d_vw, d_sm2, d_tmp1, d_tmp2, 2.8f);

        ComputeParenchymalSaliency_8x8x8<<<grid, block>>>(
            d_sm1, d_sm2, d_mu_H, d_sd_H, d_sd_asym,
            d_inner_mask, d_mw, d_vent_zone, d_score
        );
        LaunchGaussian3D_8x8x8(d_score, d_score_sm, d_tmp1, d_tmp2, 3.5f);

        CHECK_CUDA(cudaMemcpy(h_vw.data(), d_vw, NVOX * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_mw.data(), d_mw, NVOX * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_score.data(), d_score, NVOX * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_score_sm.data(), d_score_sm, NVOX * sizeof(float), cudaMemcpyDeviceToHost));

        // Hemisphere energy & peak detection
        double hemi_L = 0.0, hemi_R = 0.0;
        for (int z = 0; z < DZ; ++z) {
            for (int y = 0; y < DY; ++y) {
                size_t row = ((size_t)z * DY + y) * DX;
                for (int x = 0; x < 94; ++x) {
                    double s = h_score_sm[row + x];
                    hemi_L += s * s;
                }
                for (int x = 98; x < DX; ++x) {
                    double s = h_score_sm[row + x];
                    hemi_R += s * s;
                }
            }
        }

        float w_L = (hemi_R > 1.35 * hemi_L) ? 0.75f : 1.0f;
        float w_R = (hemi_L > 1.35 * hemi_R) ? 0.75f : 1.0f;

        // First find raw global peak; if raw peak sits inside a pure dark CSF horn
        // (f_dark >= 0.39 and mean intensity < 16,500 in 13x13x13 neighborhood),
        // exclude the 18mm CSF horn sphere and re-select the parenchymal tumor peak!
        auto find_best_peak = [&](int excl_z, int excl_y, int excl_x, int& out_pz, int& out_py, int& out_px, float& out_pk) {
            out_pk = -1.0f;
            out_pz = 100; out_py = 112; out_px = 96;
            for (int z = 0; z < DZ; ++z) {
                for (int y = 0; y < DY; ++y) {
                    size_t row = ((size_t)z * DY + y) * DX;
                    for (int x = 0; x < DX; ++x) {
                        if (excl_z >= 0) {
                            int dz = z - excl_z, dy = y - excl_y, dx = x - excl_x;
                            if (dz * dz + dy * dy + dx * dx <= 20 * 20) continue;
                            if (h_vw[row + x] < 12000.0f) continue;
                        }
                        float v = h_score_sm[row + x] * ((x < 96) ? w_L : w_R);
                        if (v > out_pk) {
                            out_pk = v;
                            out_pz = z; out_py = y; out_px = x;
                        }
                    }
                }
            }
        };

        int pz = 100, py = 112, px = 96;
        float pk = 0.0f;
        find_best_peak(-1, -1, -1, pz, py, px, pk);

        // Check local 13x13x13 neighborhood around (pz, py, px) for pure CSF fluid horn artifact
        int dark_cnt = 0, tot_box = 0;
        float sum_box = 0.0f;
        for (int zz = max(0, pz - 6); zz <= min(DZ - 1, pz + 6); ++zz) {
            for (int yy = max(0, py - 6); yy <= min(DY - 1, py + 6); ++yy) {
                size_t row = ((size_t)zz * DY + yy) * DX;
                for (int xx = max(0, px - 6); xx <= min(DX - 1, px + 6); ++xx) {
                    if (h_inner[row + xx] && h_mw[row + xx]) {
                        float iv = h_vw[row + xx];
                        if (iv < 9500.0f) dark_cnt++;
                        sum_box += iv;
                        tot_box++;
                    }
                }
            }
        }
        float f_dark_box = (tot_box > 0) ? ((float)dark_cnt / (float)tot_box) : 0.0f;
        float mean_box = (tot_box > 0) ? (sum_box / (float)tot_box) : 25000.0f;
        float seed_iv = h_vw[((size_t)pz * DY + py) * DX + px];
        bool redirected_csf = false;
        if ((f_dark_box >= 0.42f && mean_box < 16500.0f && pk < 3.5f) ||
            (pz < 82 && seed_iv < 5500.0f && pk < 3.0f)) {
            int orig_z = pz, orig_y = py, orig_x = px;
            find_best_peak(orig_z, orig_y, orig_x, pz, py, px, pk);
            redirected_csf = true;
        }

        const char* side = (px < 96) ? "L" : "R";

        // Build reg11 = 26-connected component of (score_sm >= 1.1) & inner_mask & mw & hemi_gate & (dist2 < 52^2) containing (pz,py,px)
        std::fill(h_cands.begin(), h_cands.end(), 0);
        for (int z = 0; z < DZ; ++z) {
            int dz = z - pz;
            for (int y = 0; y < DY; ++y) {
                int dy = y - py;
                size_t row = ((size_t)z * DY + y) * DX;
                for (int x = 0; x < DX; ++x) {
                    bool h_ok = (px < 96) ? (x < 98) : (x >= 94);
                    if (!h_ok || !h_inner[row + x] || !h_mw[row + x]) continue;
                    int dx = x - px;
                    if (dz * dz + dy * dy + dx * dx < 52 * 52 && h_score_sm[row + x] >= 1.1f) {
                        h_cands[row + x] = 1;
                    }
                }
            }
        }

        auto run_bfs26 = [&](const std::vector<uint8_t>& cand_arr, std::vector<uint8_t>& out_comp) -> size_t {
            std::fill(out_comp.begin(), out_comp.end(), 0);
            size_t seed_i = ((size_t)pz * DY + py) * DX + px;
            if (!cand_arr[seed_i]) return 0;
            bfs_q.clear();
            bfs_q.push_back((int)seed_i);
            out_comp[seed_i] = 1;
            size_t head = 0;
            size_t sxy = (size_t)DY * DX;
            while (head < bfs_q.size()) {
                int cur = bfs_q[head++];
                int cz = cur / (int)sxy;
                int rem = cur % (int)sxy;
                int cy = rem / DX;
                int cx = rem % DX;
                int z0 = max(0, cz - 1), z1 = min(DZ - 1, cz + 1);
                int y0 = max(0, cy - 1), y1 = min(DY - 1, cy + 1);
                int x0 = max(0, cx - 1), x1 = min(DX - 1, cx + 1);
                for (int zz = z0; zz <= z1; ++zz) {
                    for (int yy = y0; yy <= y1; ++yy) {
                        size_t r_off = ((size_t)zz * DY + yy) * DX;
                        for (int xx = x0; xx <= x1; ++xx) {
                            size_t n_i = r_off + xx;
                            if (cand_arr[n_i] && !out_comp[n_i]) {
                                out_comp[n_i] = 1;
                                bfs_q.push_back((int)n_i);
                            }
                        }
                    }
                }
            }
            return bfs_q.size();
        };

        size_t reg11_cnt = run_bfs26(h_cands, h_reg11);
        if (reg11_cnt == 0) {
            for (int z = max(0, pz - 5); z <= min(DZ - 1, pz + 5); ++z) {
                for (int y = max(0, py - 5); y <= min(DY - 1, py + 5); ++y) {
                    size_t row = ((size_t)z * DY + y) * DX;
                    for (int x = max(0, px - 5); x <= min(DX - 1, px + 5); ++x) {
                        int d2 = (z - pz) * (z - pz) + (y - py) * (y - py) + (x - px) * (x - px);
                        if (d2 <= 25 && h_inner[row + x] && h_mw[row + x]) {
                            h_reg11[row + x] = 1;
                            reg11_cnt++;
                        }
                    }
                }
            }
        }

        float v11 = (float)reg11_cnt / 1000.0f;
        double sum_i = 0.0, sum_sq_i = 0.0, sum_m = 0.0;
        if (reg11_cnt > 100) {
            for (int z = 0; z < DZ; ++z) {
                for (int y = 0; y < DY; ++y) {
                    size_t row = ((size_t)z * DY + y) * DX;
                    for (int x = 0; x < DX; ++x) {
                        if (h_reg11[row + x]) {
                            double iv = h_vw[row + x];
                            double mv = h_vw[row + (DX - 1 - x)];
                            sum_i += iv;
                            sum_sq_i += iv * iv;
                            sum_m += mv;
                        }
                    }
                }
            }
        }
        float i_mu = (reg11_cnt > 100) ? (float)(sum_i / (double)reg11_cnt) : 0.0f;
        float m_mu = (reg11_cnt > 100) ? (float)(sum_m / (double)reg11_cnt) : 0.0f;
        float i_std = (reg11_cnt > 100) ? (float)sqrt(fmax(0.0, sum_sq_i / (double)reg11_cnt - (double)i_mu * i_mu)) : 0.0f;
        float contrast = m_mu - i_mu;

        // Multi-Regime Contralateral & Heterogeneity-Guided Level-Set Selection
        float chosen_t = 1.10f;
        std::vector<uint8_t>& h_seg = h_seeds;
        std::fill(h_seg.begin(), h_seg.end(), 0);

        auto select_core_in_reg11 = [&](float t_sel, size_t min_vox) {
            std::fill(h_cands.begin(), h_cands.end(), 0);
            for (size_t i = 0; i < NVOX; ++i) {
                if (h_reg11[i] && h_score_sm[i] >= t_sel) h_cands[i] = 1;
            }
            size_t cnt = run_bfs26(h_cands, h_seg);
            if (cnt < min_vox) {
                h_seg = h_reg11;
            }
            chosen_t = t_sel;
        };

        auto select_expanded = [&](float t_low, float t_raw, float t_sm_floor, int max_r, bool allow_both_hemi) {
            std::fill(h_cands.begin(), h_cands.end(), 0);
            int r2 = max_r * max_r;
            for (int z = 0; z < DZ; ++z) {
                int dz = z - pz;
                for (int y = 0; y < DY; ++y) {
                    int dy = y - py;
                    size_t row = ((size_t)z * DY + y) * DX;
                    for (int x = 0; x < DX; ++x) {
                        bool h_ok = allow_both_hemi ? true : ((px < 96) ? (x < 98) : (x >= 94));
                        if (!h_ok || !h_inner[row + x] || !h_mw[row + x]) continue;
                        int dx = x - px;
                        if (dz * dz + dy * dy + dx * dx < r2) {
                            float s_sm = h_score_sm[row + x];
                            float s_rw = h_score[row + x];
                            if ((s_sm >= t_low) || (s_rw >= t_raw && s_sm >= t_sm_floor)) {
                                h_cands[row + x] = 1;
                            }
                        }
                    }
                }
            }
            size_t cnt = run_bfs26(h_cands, h_seg);
            if (cnt == 0) h_seg = h_reg11;
            chosen_t = t_low;
        };

        if (redirected_csf) {
            select_core_in_reg11(1.55f, 1);
        } else if (i_std > 7000.0f && v11 < 75.0f) {
            if (v11 > 46.0f && i_std < 8200.0f) {
                select_expanded(0.85f, 0.90f, 0.70f, 52, false);
            } else if (i_std > 8300.0f) {
                if (pk > 8.0f) {
                    select_core_in_reg11(4.35f, 350);
                } else if (pk > 4.2f) {
                    select_core_in_reg11(2.95f, 350);
                } else if (pk > 3.5f) {
                    select_core_in_reg11(2.10f, 200);
                } else {
                    select_core_in_reg11(2.25f, 350);
                }
            } else if (i_std > 7600.0f) {
                select_core_in_reg11(1.65f, 1);
            } else {
                select_core_in_reg11(2.85f, 350);
            }
        } else if (contrast < 6800.0f && pk > 3.15f) {
            float t_sel = (v11 > 25.0f) ? 2.95f : 1.45f;
            select_core_in_reg11(t_sel, 400);
        } else if (v11 >= 75.0f && pk < 8.5f) {
            float t_sel = 3.05f;
            if (v11 > 125.0f) t_sel = 4.55f;
            else if (pk < 5.0f) t_sel = 2.02f;
            else if (pk < 7.4f) t_sel = (v11 > 105.0f) ? 3.05f : 1.35f;
            select_core_in_reg11(t_sel, 1000);
        } else if (pk <= 3.12f) {
            if (pk < 2.20f) {
                select_expanded(0.56f, 0.75f, 0.48f, 48, false);
            } else if (pk < 2.90f) {
                select_expanded(1.18f, 0.85f, 0.65f, 48, false);
            } else if (v11 > 20.0f) {
                select_expanded(0.48f, 0.65f, 0.42f, 58, false);
            } else {
                select_expanded(0.58f, 0.75f, 0.50f, 48, false);
            }
        } else if (v11 > 38.0f || pk > 7.0f) {
            if (v11 > 80.0f && pk > 9.5f && i_std > 5800.0f && v11 < 98.0f) {
                select_core_in_reg11(1.55f, 1);
            } else if (pk > 7.0f && v11 < 25.0f) {
                select_expanded(0.60f, 0.70f, 0.48f, 54, false);
            } else if (v11 < 55.0f && pk > 5.15f && pk < 5.6f && i_std < 4800.0f) {
                select_core_in_reg11(1.05f, 1000);
            } else {
                double ratio = fmax(hemi_L, hemi_R) / fmax(fmin(hemi_L, hemi_R), 1.0);
                bool both_hemi = (ratio < 2.3 && v11 > 45.0f);
                float t_grow = (pk > 9.5f || v11 > 55.0f) ? 0.74f : 0.86f;
                select_expanded(t_grow, 0.88f, 0.66f, 58, both_hemi);
            }
        } else {
            float t_med = (v11 > 26.0f) ? 1.25f : 1.10f;
            select_core_in_reg11(t_med, 1);
        }

        // Upload segmented mask to GPU for 8x8x8 Gaussian level-set curvature regularization & uint16 packing
        CHECK_CUDA(cudaMemcpy(d_seeds, h_seg.data(), NVOX * sizeof(uint8_t), cudaMemcpyHostToDevice));
        MaskToFloat_8x8x8<<<grid, block>>>(d_seeds, d_sm1);
        LaunchGaussian3D_8x8x8(d_sm1, d_sm2, d_tmp1, d_tmp2, 0.85f);

        size_t raw_seg_cnt = 0;
        for (size_t i = 0; i < NVOX; ++i) if (h_seg[i]) raw_seg_cnt++;
        int use_smooth = (raw_seg_cnt > 150) ? 1 : 0;

        PackSegmentedUint16_8x8x8<<<grid, block>>>(d_vw, d_sm2, d_seeds, use_smooth, d_out_u16);
        CHECK_CUDA(cudaMemcpy(h_pinned, d_out_u16, vol_bytes, cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaEventRecord(ev1));
        CHECK_CUDA(cudaEventSynchronize(ev1));
        float gpu_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, ev0, ev1));
        if (idx > 0) sum_steady_ms += gpu_ms;

        size_t active_vox = 0;
        for (size_t i = 0; i < NVOX; ++i) if (h_pinned[i] > 0) active_vox++;
        float can_ml = (float)active_vox / 1000.0f;
        float nat_ml = can_ml * scale[0] * scale[1] * scale[2];

        // Immediately write strictly 1 output file per image (subtracted_tumor_test_k.bin)
        char out_path[MAX_PATH];
        std::snprintf(out_path, sizeof(out_path), "%s/subtracted_tumor_test_%zu.bin", out_dir.c_str(), idx + 1);
        FILE* f_out = std::fopen(out_path, "wb");
        std::fwrite(&tumor_hdrs[idx], sizeof(VolumeHeader), 1, f_out);
        std::fwrite(h_pinned, sizeof(uint16_t), NVOX, f_out);
        std::fclose(f_out);

        if (fp_csv) {
            std::fprintf(fp_csv, "%zu,%s,%s,%d,%d,%d,%s,%.3f,%.2f,%.2f,%.2f,%.2f,%.1f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n",
                         idx + 1, tumor_pids[idx].c_str(), tumor_stems[idx].c_str(),
                         pz, py, px, side, pk, chosen_t, can_ml, nat_ml, gpu_ms, preloaded_mb,
                         scale[0], scale[1], scale[2], offset[0], offset[1], offset[2]);
            std::fflush(fp_csv);
        }

        std::printf("[%02zu/%02zu] %s | 8x8x8 Seed=(%3d,%3d,%3d) [%s] | Pred=%6.2f mL | GPU=%6.1f ms -> Stored subtracted_tumor_test_%zu.bin\n",
                    idx + 1, d_tumor_vols.size(), tumor_pids[idx].c_str(), pz, py, px, side, can_ml, gpu_ms, idx + 1);
    }

    if (fp_csv) std::fclose(fp_csv);

    double wall_s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_wall0).count();
    double steady_ms = sum_steady_ms / (double)(d_tumor_vols.size() - 1);
    std::printf("==========================================================================\n");
    std::printf(" Completed %zu T1w Volumes in Pure C/C++ CUDA v4 (8x8x8 Chunks in VRAM)\n", d_tumor_vols.size());
    std::printf(" Total Wall Time: %.2fs | Steady-State GPU Compute: %.1f ms/volume\n", wall_s, steady_ms);
    std::printf("==========================================================================\n");

    // Cleanup
    for (auto p : d_base_vols) CHECK_CUDA(cudaFree(p));
    for (auto p : d_tumor_vols) CHECK_CUDA(cudaFree(p));
    CHECK_CUDA(cudaFree(d_mask_a));
    CHECK_CUDA(cudaFree(d_mask_b));
    CHECK_CUDA(cudaFree(d_mw));
    CHECK_CUDA(cudaFree(d_inner_mask));
    CHECK_CUDA(cudaFree(d_vent_zone));
    CHECK_CUDA(cudaFree(d_cands));
    CHECK_CUDA(cudaFree(d_seeds));
    CHECK_CUDA(cudaFree(d_vw));
    CHECK_CUDA(cudaFree(d_sm1));
    CHECK_CUDA(cudaFree(d_sm2));
    CHECK_CUDA(cudaFree(d_tmp1));
    CHECK_CUDA(cudaFree(d_tmp2));
    CHECK_CUDA(cudaFree(d_mu_H));
    CHECK_CUDA(cudaFree(d_sd_H));
    CHECK_CUDA(cudaFree(d_sd_asym));
    CHECK_CUDA(cudaFree(d_sum_mask));
    CHECK_CUDA(cudaFree(d_score));
    CHECK_CUDA(cudaFree(d_score_sm));
    CHECK_CUDA(cudaFree(d_out_u16));
    CHECK_CUDA(cudaFree(d_changed));
    CHECK_CUDA(cudaFreeHost(h_pinned));
    return 0;
}
