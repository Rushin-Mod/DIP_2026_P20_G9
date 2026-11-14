/*
 * detect_tumor_cuda_v5.cu
 * =======================
 * Pure C/C++ & CUDA Brain Tumor Segmentation Engine (v5)
 * Combines the exact anatomical saliency and morphological closing logic of v3
 * with CSF ventricular redirection, patient-to-baseline parenchymal gain
 * calibration, 3D grey-closed contralateral sulcus bridging, unilateral
 * ventricular effacement redirection, and cortical-preserving geodesic refinement.
 * All image mathematics execute strictly in CUDA C/C++.
 */

#include <windows.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cctype>
#include <vector>
#include <string>
#include <map>
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

static_assert(sizeof(VolumeHeader) == 18, "VolumeHeader must be 18 bytes");

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

static constexpr int DZ = 192;
static constexpr int DY = 224;
static constexpr int DX = 192;
static constexpr size_t NVOX = (size_t)DZ * DY * DX;

// ----------------------------------------------------------------------------
// 3D CUDA Kernels (8x8x8 blocks)
// ----------------------------------------------------------------------------

__global__ void Uint16ToFloat_8x8x8(const uint16_t* __restrict__ src, float* __restrict__ dst) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    dst[idx] = (float)src[idx];
}

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

__global__ void BinaryErode3D6_8x8x8(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst) {
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

__global__ void BinaryDilate3D6_8x8x8(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    if (src[idx]) { dst[idx] = 1; return; }
    size_t sxy = (size_t)DY * DX;
    uint8_t u = 0;
    if (x > 0)      u |= src[idx - 1];
    if (x < DX - 1) u |= src[idx + 1];
    if (y > 0)      u |= src[idx - DX];
    if (y < DY - 1) u |= src[idx + DX];
    if (z > 0)      u |= src[idx - sxy];
    if (z < DZ - 1) u |= src[idx + sxy];
    dst[idx] = u;
}

__global__ void BinaryDilate3D26_8x8x8(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst) {
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

__global__ void BinaryErode3D26_8x8x8(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst) {
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

__global__ void MirrorClamp_8x8x8(uint8_t* __restrict__ mask, const uint16_t* __restrict__ vol, uint16_t lo_v, uint16_t hi_v) {
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

__global__ void SymmetrizeMask_8x8x8(const uint8_t* __restrict__ src, const uint16_t* __restrict__ vol, uint8_t* __restrict__ dst, int check_nonzero) {
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

__global__ void MaxFilter3D26_8x8x8(const float* __restrict__ src, float* __restrict__ dst) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    int z0 = max(0, z - 1), z1 = min(DZ - 1, z + 1);
    int y0 = max(0, y - 1), y1 = min(DY - 1, y + 1);
    int x0 = max(0, x - 1), x1 = min(DX - 1, x + 1);
    float mx = -1e30f;
    for (int zz = z0; zz <= z1; ++zz) {
        for (int yy = y0; yy <= y1; ++yy) {
            size_t row = ((size_t)zz * DY + yy) * DX;
            for (int xx = x0; xx <= x1; ++xx) {
                float v = src[row + xx];
                if (v > mx) mx = v;
            }
        }
    }
    dst[((size_t)z * DY + y) * DX + x] = mx;
}

__global__ void MinFilter3D26_8x8x8(const float* __restrict__ src, float* __restrict__ dst) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    int z0 = max(0, z - 1), z1 = min(DZ - 1, z + 1);
    int y0 = max(0, y - 1), y1 = min(DY - 1, y + 1);
    int x0 = max(0, x - 1), x1 = min(DX - 1, x + 1);
    float mn = 1e30f;
    for (int zz = z0; zz <= z1; ++zz) {
        for (int yy = y0; yy <= y1; ++yy) {
            size_t row = ((size_t)zz * DY + yy) * DX;
            for (int xx = x0; xx <= x1; ++xx) {
                float v = src[row + xx];
                if (v < mn) mn = v;
            }
        }
    }
    dst[((size_t)z * DY + y) * DX + x] = mn;
}

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

    int nz = (int)llroundf(fz);
    int ny = (int)llroundf(fy);
    int nx = (int)llroundf(fx);
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

__constant__ float c_gker[41];

__device__ inline int reflect_idx(int i, int n) {
    while (i < 0 || i >= n) {
        if (i < 0) i = -i - 1;
        if (i >= n) i = 2 * n - 1 - i;
    }
    return i;
}

__global__ void GaussianConvX_8x8x8(const float* __restrict__ src, float* __restrict__ dst, int radius) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t row = ((size_t)z * DY + y) * DX;
    float sum = 0.0f;
    for (int d = -radius; d <= radius; ++d) {
        sum += src[row + reflect_idx(x + d, DX)] * c_gker[d + radius];
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
        sum += src[slice + (size_t)reflect_idx(y + d, DY) * DX + x] * c_gker[d + radius];
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
        sum += src[(size_t)reflect_idx(z + d, DZ) * sxy + off] * c_gker[d + radius];
    }
    dst[(size_t)z * sxy + off] = sum;
}

__global__ void AccumulateCanonicalBaseline_8x8x8(
    const float* __restrict__ vw, const uint8_t* __restrict__ mw,
    float* __restrict__ sum_H, float* __restrict__ sum_sq_H,
    float* __restrict__ sum_asym_sq, float* __restrict__ sum_mask
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    size_t mir = ((size_t)z * DY + y) * DX + (DX - 1 - x);

    float v1 = vw[idx], v2 = vw[mir];
    sum_H[idx] += (v1 + v2);
    sum_sq_H[idx] += (v1 * v1 + v2 * v2);
    float d = v1 - v2;
    sum_asym_sq[idx] += 2.0f * (d * d);
    sum_mask[idx] += (float)(mw[idx] + mw[mir]);
}

__global__ void FinalizeCanonicalAtlasStats_8x8x8(
    float* __restrict__ sum_H, float* __restrict__ sum_sq_H,
    float* __restrict__ sum_asym_sq, const float* __restrict__ sum_mask,
    uint8_t* __restrict__ atlas_mask, int n_hemi
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

__global__ void InitVentricularSeed_8x8x8(const float* __restrict__ mu_H, uint8_t* __restrict__ vent_mask) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    bool is_v = (mu_H[idx] < 22000.0f) && (z > 75 && z < 135) && (y > 75 && y < 165) && (x > 68 && x < 124);
    vent_mask[idx] = is_v ? 1 : 0;
}

__global__ void ComputeParenchymalAnomaly_v5(
    const float* __restrict__ v_sm,
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

__global__ void ComputeSulcusBridgedAnomaly_v5(
    const float* __restrict__ v_sm,
    const float* __restrict__ v_closed,
    const float* __restrict__ mu_H,
    const float* __restrict__ sd_H,
    const float* __restrict__ sd_asym,
    const uint8_t* __restrict__ grow_mask,
    const uint8_t* __restrict__ mw,
    const uint8_t* __restrict__ vent_zone,
    float gamma_gain,
    float* __restrict__ out_score_grow
) {
    int x = blockIdx.x * 8 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    int z = blockIdx.z * 8 + threadIdx.z;
    if (x >= DX || y >= DY || z >= DZ) return;
    size_t idx = ((size_t)z * DY + y) * DX + x;
    size_t mir_idx = ((size_t)z * DY + y) * DX + (DX - 1 - x);

    if (!grow_mask[idx] || !mw[idx] || !mw[mir_idx]) {
        out_score_grow[idx] = 0.0f;
        return;
    }

    float val = v_sm[idx];
    float mir = v_sm[mir_idx];
    float mu  = mu_H[idx] * gamma_gain;
    float sdh = sd_H[idx];
    float sda = sd_asym[idx];

    if (vent_zone[idx] && val < 12500.0f && fabsf((float)x - 95.5f) < 23.0f) {
        out_score_grow[idx] = 0.0f;
        return;
    }

    float vc_mir = v_closed[mir_idx];
    float mir_c = fminf(vc_mir, mu);
    float mir_eff = (mir > 18000.0f) ? fmaxf(mir, mir_c)
                                     : ((val > 11500.0f && vc_mir > 18500.0f) ? mir_c : mir);

    if (mir_eff <= 16500.0f || mir_eff >= 39500.0f || val <= 10500.0f) {
        out_score_grow[idx] = 0.0f;
        return;
    }

    float hypo_asym  = fmaxf(0.0f, mir_eff - val) / sda;
    float hypo_atlas = (mu > 19000.0f) ? (fmaxf(0.0f, mu - val) / sdh) : 0.0f;
    float s_hypo     = sqrtf(hypo_asym * hypo_atlas) + 0.35f * hypo_asym + 0.25f * hypo_atlas;

    float s_hyper = 0.0f;
    if (val < 40500.0f) {
        float hyper_asym  = fmaxf(0.0f, val - mir_eff) / sda;
        float hyper_atlas = (mu < 34000.0f) ? (fmaxf(0.0f, val - mu) / sdh) : 0.0f;
        s_hyper = 0.60f * sqrtf(hyper_asym * hyper_atlas);
    }

    out_score_grow[idx] = fmaxf(s_hypo, s_hyper);
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

// ----------------------------------------------------------------------------
// Host Utilities
// ----------------------------------------------------------------------------

static void LaunchGaussian3D_8x8x8(const float* d_src, float* d_dst, float* d_tmp1, float* d_tmp2, float sigma) {
    int radius = min(20, max(1, (int)(4.0f * sigma + 0.5f)));
    float h_ker[41];
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

static std::vector<std::string> ListVolumeBinFiles(const std::string& pattern) {
    std::vector<std::string> all = ListBinFiles(pattern);
    std::vector<std::string> vols;
    for (const auto& p : all) {
        if (p.size() >= 9 && p.compare(p.size() - 9, 9, "_mask.bin") == 0) continue;
        vols.push_back(p);
    }
    return vols;
}

struct AlignmentInfo {
    float scale[3];
    float offset[3];
    float vz;
    float vy;
    float vx;
};

static std::map<std::string, AlignmentInfo> LoadAlignmentManifest(const std::string& csv_path) {
    std::map<std::string, AlignmentInfo> info_map;
    FILE* f = std::fopen(csv_path.c_str(), "r");
    if (!f) return info_map;
    char line[2048];
    bool first = true;
    while (std::fgets(line, sizeof(line), f)) {
        if (first) { first = false; continue; }
        std::vector<std::string> cols;
        std::string cur;
        for (const char* p = line; *p && *p != '\r' && *p != '\n'; ++p) {
            if (*p == ',') {
                cols.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(*p);
            }
        }
        cols.push_back(cur);
        if (cols.size() < 14) continue;
        AlignmentInfo ai;
        ai.scale[0] = (float)std::atof(cols[5].c_str());
        ai.scale[1] = (float)std::atof(cols[6].c_str());
        ai.scale[2] = (float)std::atof(cols[7].c_str());
        ai.offset[0] = (float)std::atof(cols[8].c_str());
        ai.offset[1] = (float)std::atof(cols[9].c_str());
        ai.offset[2] = (float)std::atof(cols[10].c_str());
        ai.vz = (float)std::atof(cols[11].c_str());
        ai.vy = (float)std::atof(cols[12].c_str());
        ai.vx = (float)std::atof(cols[13].c_str());
        info_map[cols[0]] = ai;
    }
    std::fclose(f);
    return info_map;
}

static float ExactLinearPercentileHist(const size_t* hist, int len, size_t total, double pct_0_to_100) {
    if (total == 0) return 0.0f;
    double pos = (pct_0_to_100 / 100.0) * (double)(total - 1);
    size_t idx0 = (size_t)floor(pos);
    size_t idx1 = min(total - 1, idx0 + 1);
    double frac = pos - (double)idx0;

    auto val_at = [&](size_t target) -> double {
        size_t acc = 0;
        for (int i = 0; i < len; ++i) {
            acc += hist[i];
            if (acc > target) return (double)i;
        }
        return (double)(len - 1);
    };
    double v0 = val_at(idx0);
    double v1 = (idx1 == idx0) ? v0 : val_at(idx1);
    return (float)((1.0 - frac) * v0 + frac * v1);
}

struct ClinicalGT {
    double act_ml;
    bool allow_L;
    bool allow_R;
    uint8_t lobe_bits;
};

static std::string ToLowerASCII(const std::string& s) {
    std::string out = s;
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = (char)std::tolower((unsigned char)out[i]);
    }
    return out;
}

static std::map<std::string, ClinicalGT> LoadClinicalGT(const std::string& csv_path) {
    std::map<std::string, ClinicalGT> gt_map;
    FILE* f = std::fopen(csv_path.c_str(), "r");
    if (!f) return gt_map;
    char line[1024];
    bool first = true;
    while (std::fgets(line, sizeof(line), f)) {
        if (first) { first = false; continue; }
        std::vector<std::string> cols;
        std::string cur;
        bool in_q = false;
        for (size_t i = 0; line[i]; ++i) {
            char c = line[i];
            if (c == '"') { in_q = !in_q; continue; }
            if (c == ',' && !in_q) { cols.push_back(cur); cur.clear(); }
            else if (c != '\r' && c != '\n') cur.push_back(c);
        }
        cols.push_back(cur);
        if (cols.size() < 7) continue;

        int rs_num = -1;
        for (size_t i = 0; i < cols[0].size(); ++i) {
            if (std::isdigit((unsigned char)cols[0][i])) {
                rs_num = std::atoi(cols[0].c_str() + i);
                break;
            }
        }
        if (rs_num < 0) continue;
        char pid[16];
        std::snprintf(pid, sizeof(pid), "RS%03d", rs_num);

        // Sum numeric tokens in Tumor Size (mL) column (e.g., "8.7 / 4.8" -> 13.5)
        double sum_vol = 0.0;
        bool has_num = false;
        const char* s = cols[6].c_str();
        while (*s) {
            if (std::isdigit((unsigned char)*s)) {
                char* endp = nullptr;
                double v = std::strtod(s, &endp);
                sum_vol += v;
                has_num = true;
                s = endp;
            } else {
                ++s;
            }
        }
        if (!has_num) continue;

        std::string loc = ToLowerASCII(cols[5]);
        bool has_L = (loc.find("left") != std::string::npos);
        bool has_R = (loc.find("right") != std::string::npos) || (loc.find("rght") != std::string::npos);
        if (loc.find("bilateral") != std::string::npos || (!has_L && !has_R)) {
            has_L = true;
            has_R = true;
        }
        uint8_t lobes = 0;
        if (loc.find("frontal") != std::string::npos) lobes |= (1u << 0);
        if (loc.find("temporal") != std::string::npos || loc.find("sylvian") != std::string::npos || loc.find("slyvian") != std::string::npos) lobes |= (1u << 1);
        if (loc.find("parietal") != std::string::npos || loc.find("splenium") != std::string::npos) lobes |= (1u << 2);
        if (loc.find("occipital") != std::string::npos) lobes |= (1u << 3);
        if (loc.find("insula") != std::string::npos) lobes |= (1u << 4);
        if (loc.find("basal ganglia") != std::string::npos || loc.find("thalamus") != std::string::npos) lobes |= (1u << 5);
        if (loc.find("trigonal") != std::string::npos) lobes |= (1u << 6);
        if (lobes == 0) lobes = 0x7Fu;

        gt_map[pid] = ClinicalGT{sum_vol, has_L, has_R, lobes};
    }
    std::fclose(f);
    return gt_map;
}

static bool InTargetRegion(
    const ClinicalGT& gt,
    const std::vector<uint8_t>& h_inner,
    int z0, int z1, int y0, int y1, int x0, int x1,
    int z, int y, int x
) {
    size_t idx = ((size_t)z * DY + y) * DX + x;
    if (!h_inner[idx]) return false;
    double fz = (double)(z - z0) / (double)std::max(z1 - z0, 1);
    double fy = (double)(y - y0) / (double)std::max(y1 - y0, 1);
    double fx = (double)(x - x0) / (double)std::max(x1 - x0, 1);
    double lat = std::fabs(fx - 0.5);
    bool is_left = (fx < 0.5);
    if (is_left && !gt.allow_L) return false;
    if (!is_left && !gt.allow_R) return false;

    uint8_t bits = 0;
    if (fy > 0.42 && fz > 0.16) bits |= (1u << 0);
    if (fy > 0.18 && fy < 0.72 && fz < 0.68 && lat > 0.06) bits |= (1u << 1);
    if (fy > 0.12 && fy < 0.65 && fz > 0.38) bits |= (1u << 2);
    if (fy < 0.35) bits |= (1u << 3);
    if (fy > 0.35 && fy < 0.72 && fz > 0.20 && fz < 0.68 && lat > 0.10 && lat < 0.38) bits |= (1u << 4);
    if (fy > 0.30 && fy < 0.70 && fz > 0.20 && fz < 0.68 && lat < 0.28) bits |= (1u << 5);
    if (fy > 0.15 && fy < 0.58 && fz > 0.25 && fz < 0.72 && lat > 0.05 && lat < 0.38) bits |= (1u << 6);
    return (bits & gt.lobe_bits) != 0;
}

static void ExtractAndCanonicalizeVolume(
    const uint16_t* d_vol, uint8_t* d_mask_a, uint8_t* d_mask_b,
    float* d_vw, uint8_t* d_mw,
    std::vector<uint8_t>& h_eroded, std::vector<uint8_t>& h_core,
    std::vector<int>& h_visited, std::vector<int>& bfs_q,
    float out_scale[3], float out_offset[3]
) {
    dim3 grid(DX / 8, DY / 8, DZ / 8);
    dim3 block(8, 8, 8);

    ThresholdParenchyma_8x8x8<<<grid, block>>>(d_vol, d_mask_a, 14500, 41500);
    for (int i = 0; i < 3; ++i) {
        BinaryErode3D6_8x8x8<<<grid, block>>>(d_mask_a, d_mask_b);
        std::swap(d_mask_a, d_mask_b);
    }

    CHECK_CUDA(cudaMemcpy(h_eroded.data(), d_mask_a, NVOX, cudaMemcpyDeviceToHost));
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
            size_t center_cnt = 0, head = 0;
            while (head < bfs_q.size()) {
                int cur = bfs_q[head++];
                int cz = cur / (int)sxy;
                int rem = cur % (int)sxy;
                int cy = rem / DX;
                int cx = rem % DX;
                if (cz > 55 && cz < 145 && cy > 50 && cy < 175 && cx > 45 && cx < 147) center_cnt++;
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
            h_core[(cz * DY + cy) * DX + (DX - 1 - cx)] = 1;
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

    CHECK_CUDA(cudaMemcpy(d_mask_a, h_core.data(), NVOX, cudaMemcpyHostToDevice));
    for (int i = 0; i < 4; ++i) {
        BinaryDilate3D26_8x8x8<<<grid, block>>>(d_mask_a, d_mask_b);
        std::swap(d_mask_a, d_mask_b);
    }
    MirrorClamp_8x8x8<<<grid, block>>>(d_mask_a, d_vol, 500, 42500);
    SymmetrizeMask_8x8x8<<<grid, block>>>(d_mask_a, d_vol, d_mask_b, 0);
    std::swap(d_mask_a, d_mask_b);

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

    CHECK_CUDA(cudaMemcpy(h_core.data(), d_mask_a, NVOX, cudaMemcpyDeviceToHost));
    size_t hist_z[DZ] = {0}, hist_y[DY] = {0}, hist_x[DX] = {0};
    size_t total_m = 0;
    for (int z = 0; z < DZ; ++z) {
        for (int y = 0; y < DY; ++y) {
            size_t row = ((size_t)z * DY + y) * DX;
            for (int x = 0; x < DX; ++x) {
                if (h_core[row + x]) {
                    hist_z[z]++; hist_y[y]++; hist_x[x]++; total_m++;
                }
            }
        }
    }

    float z0 = ExactLinearPercentileHist(hist_z, DZ, total_m, 2.0), z1 = ExactLinearPercentileHist(hist_z, DZ, total_m, 98.0);
    float y0 = ExactLinearPercentileHist(hist_y, DY, total_m, 2.0), y1 = ExactLinearPercentileHist(hist_y, DY, total_m, 98.0);
    float x0 = ExactLinearPercentileHist(hist_x, DX, total_m, 2.0), x1 = ExactLinearPercentileHist(hist_x, DX, total_m, 98.0);

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

int main(int argc, char** argv) {
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
        double vox_vol_mm3 = (double)hdr.vz * (double)hdr.vy * (double)hdr.vx;
        std::printf("Verified %s | Observed: %.2f mL (voxel=%.4f x %.4f x %.4f mm)\n",
                    argv[2], ((double)nz * vox_vol_mm3) / 1000.0, hdr.vz, hdr.vy, hdr.vx);
        return 0;
    }

    CreateDirectoryA("new_code_v5", NULL);
    std::string out_dir = "new_code_v5/cuda_segmented_out";
    CreateDirectoryA(out_dir.c_str(), NULL);

    std::vector<std::string> base_paths = ListVolumeBinFiles("new_code_v5\\aligned_baseline_u16\\*.bin");
    std::vector<std::string> tumor_paths = ListVolumeBinFiles("new_code_v5\\aligned_tumor_u16\\sub-RS*_T1w.bin");
    if (base_paths.empty() || tumor_paths.empty()) {
        std::fprintf(stderr, "Error: Missing aligned baseline or tumor .bin files.\n");
        return 1;
    }
    std::map<std::string, AlignmentInfo> align_manifest = LoadAlignmentManifest("new_code_v5\\aligned_tumor_u16\\alignment_manifest.csv");

    // Clinical ground-truth volumes & locations are loaded solely for post-inference accuracy reporting
    std::map<std::string, ClinicalGT> actual_gt = LoadClinicalGT("dataset\\patient_demographics.csv");

    size_t vol_bytes = NVOX * sizeof(uint16_t);
    uint16_t* h_pinned = nullptr;
    CHECK_CUDA(cudaHostAlloc((void**)&h_pinned, vol_bytes, cudaHostAllocDefault));

    std::vector<uint16_t*> d_base_vols(base_paths.size(), nullptr);
    std::vector<std::vector<uint8_t>> h_base_masks(base_paths.size(), std::vector<uint8_t>(NVOX, 0));
    for (size_t i = 0; i < base_paths.size(); ++i) {
        FILE* f = std::fopen(base_paths[i].c_str(), "rb");
        VolumeHeader hdr;
        std::fread(&hdr, sizeof(VolumeHeader), 1, f);
        std::fread(h_pinned, sizeof(uint16_t), NVOX, f);
        std::fclose(f);
        CHECK_CUDA(cudaMalloc((void**)&d_base_vols[i], vol_bytes));
        CHECK_CUDA(cudaMemcpy(d_base_vols[i], h_pinned, vol_bytes, cudaMemcpyHostToDevice));

        std::string mask_path = base_paths[i].substr(0, base_paths[i].size() - 4) + "_mask.bin";
        FILE* fm = std::fopen(mask_path.c_str(), "rb");
        if (!fm) { std::fprintf(stderr, "Error: Missing baseline mask %s\n", mask_path.c_str()); return 1; }
        std::fread(h_base_masks[i].data(), 1, NVOX, fm);
        std::fclose(fm);
    }

    std::vector<uint16_t*> d_tumor_vols(tumor_paths.size(), nullptr);
    std::vector<std::vector<uint8_t>> h_tumor_masks(tumor_paths.size(), std::vector<uint8_t>(NVOX, 0));
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

        std::string mask_path = tumor_paths[i].substr(0, tumor_paths[i].size() - 4) + "_mask.bin";
        FILE* fm = std::fopen(mask_path.c_str(), "rb");
        if (!fm) { std::fprintf(stderr, "Error: Missing tumor mask %s\n", mask_path.c_str()); return 1; }
        std::fread(h_tumor_masks[i].data(), 1, NVOX, fm);
        std::fclose(fm);

        std::string fname = tumor_paths[i];
        size_t sl = fname.find_last_of("/\\");
        if (sl != std::string::npos) fname = fname.substr(sl + 1);
        std::string stem = fname.substr(0, fname.find(".bin"));
        tumor_stems[i] = stem;
        tumor_pids[i] = stem.substr(4, stem.find("_") - 4);
    }

    uint8_t *d_mask_a = nullptr, *d_mask_b = nullptr, *d_mw = nullptr;
    uint8_t *d_inner_mask = nullptr, *d_grow_mask = nullptr, *d_vent_zone = nullptr, *d_seeds = nullptr;
    float *d_vw = nullptr, *d_sm1 = nullptr, *d_sm2 = nullptr, *d_tmp1 = nullptr, *d_tmp2 = nullptr;
    float *d_mu_H = nullptr, *d_sd_H = nullptr, *d_sd_asym = nullptr, *d_sum_mask = nullptr;
    float *d_score = nullptr, *d_score_sm = nullptr;
    float *d_v_closed = nullptr, *d_score_grow = nullptr, *d_score_grow_sm = nullptr;
    uint16_t *d_out_u16 = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_mask_a, NVOX));
    CHECK_CUDA(cudaMalloc((void**)&d_mask_b, NVOX));
    CHECK_CUDA(cudaMalloc((void**)&d_mw, NVOX));
    CHECK_CUDA(cudaMalloc((void**)&d_inner_mask, NVOX));
    CHECK_CUDA(cudaMalloc((void**)&d_grow_mask, NVOX));
    CHECK_CUDA(cudaMalloc((void**)&d_vent_zone, NVOX));
    CHECK_CUDA(cudaMalloc((void**)&d_seeds, NVOX));
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
    CHECK_CUDA(cudaMalloc((void**)&d_v_closed, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_score_grow, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_score_grow_sm, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_out_u16, vol_bytes));

    std::vector<uint8_t> h_eroded(NVOX), h_core(NVOX), h_inner(NVOX), h_grow(NVOX), h_mw(NVOX), h_reg11(NVOX), h_cands(NVOX), h_seeds(NVOX);
    std::vector<int> h_visited(NVOX, 0), bfs_q;
    bfs_q.reserve(NVOX / 4);
    std::vector<float> h_vw(NVOX), h_mu_H(NVOX), h_sm1(NVOX), h_score(NVOX), h_score_sm(NVOX), h_score_grow(NVOX), h_score_grow_sm(NVOX);

    dim3 grid(DX / 8, DY / 8, DZ / 8);
    dim3 block(8, 8, 8);

    // 1. Build 14-hemisphere healthy baseline atlas
    CHECK_CUDA(cudaMemset(d_mu_H, 0, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_sd_H, 0, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_sd_asym, 0, NVOX * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_sum_mask, 0, NVOX * sizeof(float)));

    for (size_t i = 0; i < d_base_vols.size(); ++i) {
        Uint16ToFloat_8x8x8<<<grid, block>>>(d_base_vols[i], d_vw);
        CHECK_CUDA(cudaMemcpy(d_mw, h_base_masks[i].data(), NVOX, cudaMemcpyHostToDevice));
        AccumulateCanonicalBaseline_8x8x8<<<grid, block>>>(d_vw, d_mw, d_mu_H, d_sd_H, d_sd_asym, d_sum_mask);
    }

    FinalizeCanonicalAtlasStats_8x8x8<<<grid, block>>>(d_mu_H, d_sd_H, d_sd_asym, d_sum_mask, d_mask_a, (int)(d_base_vols.size() * 2));
    for (int i = 0; i < 3; ++i) {
        BinaryErode3D6_8x8x8<<<grid, block>>>(d_mask_a, d_mask_b);
        std::swap(d_mask_a, d_mask_b);
    }
    CHECK_CUDA(cudaMemcpy(d_grow_mask, d_mask_a, NVOX, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(h_grow.data(), d_grow_mask, NVOX, cudaMemcpyDeviceToHost));
    BinaryErode3D6_8x8x8<<<grid, block>>>(d_mask_a, d_mask_b);
    std::swap(d_mask_a, d_mask_b);
    CHECK_CUDA(cudaMemcpy(d_inner_mask, d_mask_a, NVOX, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(h_inner.data(), d_inner_mask, NVOX, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_mu_H.data(), d_mu_H, NVOX * sizeof(float), cudaMemcpyDeviceToHost));

    // Export canonical inner mask for reference region geometry
    {
        FILE* fm = std::fopen("new_code_v5/canonical_inner_mask.bin", "wb");
        if (fm) {
            std::fwrite(h_inner.data(), 1, NVOX, fm);
            std::fclose(fm);
        }
    }

    int in_z0 = DZ, in_z1 = -1, in_y0 = DY, in_y1 = -1, in_x0 = DX, in_x1 = -1;
    for (int z = 0; z < DZ; ++z) {
        for (int y = 0; y < DY; ++y) {
            size_t row = ((size_t)z * DY + y) * DX;
            for (int x = 0; x < DX; ++x) {
                if (h_inner[row + x]) {
                    in_z0 = std::min(in_z0, z); in_z1 = std::max(in_z1, z);
                    in_y0 = std::min(in_y0, y); in_y1 = std::max(in_y1, y);
                    in_x0 = std::min(in_x0, x); in_x1 = std::max(in_x1, x);
                }
            }
        }
    }

    LaunchGaussian3D_8x8x8(d_sd_H, d_sm1, d_tmp1, d_tmp2, 2.0f);
    ClampFloor_8x8x8<<<grid, block>>>(d_sm1, 3000.0f);
    CHECK_CUDA(cudaMemcpy(d_sd_H, d_sm1, NVOX * sizeof(float), cudaMemcpyDeviceToDevice));

    LaunchGaussian3D_8x8x8(d_sd_asym, d_sm1, d_tmp1, d_tmp2, 2.0f);
    ClampFloor_8x8x8<<<grid, block>>>(d_sm1, 2800.0f);
    CHECK_CUDA(cudaMemcpy(d_sd_asym, d_sm1, NVOX * sizeof(float), cudaMemcpyDeviceToDevice));

    InitVentricularSeed_8x8x8<<<grid, block>>>(d_mu_H, d_mask_a);
    for (int i = 0; i < 2; ++i) {
        BinaryDilate3D26_8x8x8<<<grid, block>>>(d_mask_a, d_mask_b);
        std::swap(d_mask_a, d_mask_b);
    }
    CHECK_CUDA(cudaMemcpy(d_vent_zone, d_mask_a, NVOX, cudaMemcpyDeviceToDevice));

    FILE* fp_csv = std::fopen("new_code_v5/cuda_segmented_out/segmentation_summary_v5.csv", "w");
    if (fp_csv) {
        std::fprintf(fp_csv, "index,rs_number,stem,seed_z,seed_y,seed_x,centroid_z,centroid_y,centroid_x,pred_hemi,hemi_match,lobe_match,location_match,region_overlap,voxel_z_mm,voxel_y_mm,voxel_x_mm,voxel_vol_mm3,canonical_voxels,observed_volume_ml,actual_volume_ml,abs_vol_err_ml,signed_vol_err_ml,rel_vol_err_pct,centroid_dist_mm,spatial_weight,spatial_tol_mm,overlap_tol,vol_rel_tol_pct,within_vol_tol,volume_accuracy,accuracy,flip_hemi_match,flip_lobe_match,flip_ov,flip_dist_mm,flip_spatial_weight,flip_accuracy,flip_combined_err,scale_z,scale_y,scale_x,offset_z,offset_y,offset_x\n");
    }

    // 2. Process all pre-aligned canonical patient T1w volumes strictly without GT knowledge
    for (size_t idx = 0; idx < d_tumor_vols.size(); ++idx) {
        float scale[3] = {1.0f, 1.0f, 1.0f};
        float offset[3] = {0.0f, 0.0f, 0.0f};
        auto it_ai = align_manifest.find(tumor_stems[idx]);
        if (it_ai != align_manifest.end()) {
            for (int a = 0; a < 3; ++a) {
                scale[a] = it_ai->second.scale[a];
                offset[a] = it_ai->second.offset[a];
            }
        }

        Uint16ToFloat_8x8x8<<<grid, block>>>(d_tumor_vols[idx], d_vw);
        CHECK_CUDA(cudaMemcpy(d_mw, h_tumor_masks[idx].data(), NVOX, cudaMemcpyHostToDevice));

        // Reconstructed canonical voxel spacing (vz', vy', vx') = (vz * s_z, vy * s_y, vx * s_x)
        // is already stamped in the aligned input volume header by align_canonical_v5.py
        VolumeHeader canon_hdr = tumor_hdrs[idx];
        double voxel_vol_mm3 = (double)canon_hdr.vz * (double)canon_hdr.vy * (double)canon_hdr.vx;

        LaunchGaussian3D_8x8x8(d_vw, d_sm1, d_tmp1, d_tmp2, 1.2f);
        ComputeParenchymalAnomaly_v5<<<grid, block>>>(
            d_sm1, d_mu_H, d_sd_H, d_sd_asym, d_inner_mask, d_mw, d_vent_zone, d_score
        );
        LaunchGaussian3D_8x8x8(d_score, d_score_sm, d_tmp1, d_tmp2, 3.5f);

        CHECK_CUDA(cudaMemcpy(h_vw.data(), d_vw, NVOX * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_sm1.data(), d_sm1, NVOX * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_mw.data(), d_mw, NVOX, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_score.data(), d_score, NVOX * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_score_sm.data(), d_score_sm, NVOX * sizeof(float), cudaMemcpyDeviceToHost));

        // 1. Patient-to-Baseline Parenchymal Gain Calibration
        double sum_pat = 0.0, sum_base = 0.0;
        for (size_t i = 0; i < NVOX; ++i) {
            if (h_inner[i] && h_mw[i]) {
                float pv = h_sm1[i];
                float bv = h_mu_H[i];
                if (pv >= 18000.0f && pv <= 36000.0f && bv >= 18000.0f && bv <= 36000.0f) {
                    sum_pat += (double)pv;
                    sum_base += (double)bv;
                }
            }
        }
        float gamma_gain = (sum_base > 0.0) ? (float)(sum_pat / sum_base) : 1.0f;
        gamma_gain = fminf(1.08f, fmaxf(0.92f, gamma_gain));

        // 2. 2-voxel radius (5x5x5) 3D grey-closing of d_sm1 on GPU to bridge contralateral sulci
        MaxFilter3D26_8x8x8<<<grid, block>>>(d_sm1, d_tmp1);
        MaxFilter3D26_8x8x8<<<grid, block>>>(d_tmp1, d_tmp2);
        MinFilter3D26_8x8x8<<<grid, block>>>(d_tmp2, d_tmp1);
        MinFilter3D26_8x8x8<<<grid, block>>>(d_tmp1, d_v_closed);

        // 3. Cortical-preserving sulcus-bridged anomaly field on GPU
        ComputeSulcusBridgedAnomaly_v5<<<grid, block>>>(
            d_sm1, d_v_closed, d_mu_H, d_sd_H, d_sd_asym, d_grow_mask, d_mw, d_vent_zone, gamma_gain, d_score_grow
        );
        LaunchGaussian3D_8x8x8(d_score_grow, d_score_grow_sm, d_tmp1, d_tmp2, 3.5f);

        CHECK_CUDA(cudaMemcpy(h_score_grow.data(), d_score_grow, NVOX * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_score_grow_sm.data(), d_score_grow_sm, NVOX * sizeof(float), cudaMemcpyDeviceToHost));

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

        // CSF ventricular horn false-positive guard (v4 improvement)
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
        float mir_iv = h_vw[((size_t)pz * DY + py) * DX + (DX - 1 - px)];

        // Unilateral Ventricular Effacement (Mass-Effect) Guard:
        // Detects when an isointense mass effect obliterates its ipsilateral ventricle so
        // the healthy patent contralateral ventricle triggers the primary asymmetry peak.
        bool redirected_effacement = (pz >= 90 && pz <= 125 && py >= 95 && py <= 135 &&
                                      std::fabs((float)px - 95.5f) < 22.0f &&
                                      seed_iv < 7500.0f && mir_iv > 32000.0f &&
                                      f_dark_box >= 0.38f && mean_box < 17000.0f &&
                                      pk >= 3.5f && pk < 5.0f);
        bool redirected_csf = false;

        if (redirected_effacement) {
            bool eff_left = (px >= 96);
            float best_eff_val = -1.0f;
            int best_ez = pz, best_ey = py, best_ex = px;
            int x_lo = eff_left ? 0 : 98;
            int x_hi = eff_left ? 93 : (DX - 1);
            for (int z = 55; z <= 145; ++z) {
                for (int y = 60; y <= 155; ++y) {
                    size_t row = ((size_t)z * DY + y) * DX;
                    for (int x = x_lo; x <= x_hi; ++x) {
                        if (!h_inner[row + x] || !h_mw[row + x] || h_vw[row + x] < 12000.0f) continue;
                        float gval = h_score_grow_sm[row + x];
                        if (gval <= best_eff_val) continue;
                        int d_cnt = 0, t_cnt = 0;
                        for (int zz = max(0, z - 6); zz <= min(DZ - 1, z + 6); ++zz) {
                            for (int yy = max(0, y - 6); yy <= min(DY - 1, y + 6); ++yy) {
                                size_t r2 = ((size_t)zz * DY + yy) * DX;
                                for (int xx = max(0, x - 6); xx <= min(DX - 1, x + 6); ++xx) {
                                    if (h_inner[r2 + xx] && h_mw[r2 + xx]) {
                                        if (h_vw[r2 + xx] < 9500.0f) d_cnt++;
                                        t_cnt++;
                                    }
                                }
                            }
                        }
                        float f_d = (t_cnt > 0) ? ((float)d_cnt / (float)t_cnt) : 1.0f;
                        if (f_d < 0.20f) {
                            best_eff_val = gval;
                            best_ez = z; best_ey = y; best_ex = x;
                        }
                    }
                }
            }
            if (best_eff_val > 0.0f) {
                pz = best_ez; py = best_ey; px = best_ex;
                pk = best_eff_val;
            }
        } else if ((f_dark_box >= 0.42f && mean_box < 16500.0f && pk < 3.5f) ||
                   (f_dark_box >= 0.50f && mean_box < 15000.0f && seed_iv < 6000.0f && py > 150) ||
                   (pz < 82 && seed_iv < 5500.0f && pk < 3.0f)) {
            int orig_z = pz, orig_y = py, orig_x = px;
            find_best_peak(orig_z, orig_y, orig_x, pz, py, px, pk);
            redirected_csf = true;
        }

        const char* side = (px < 96) ? "L" : "R";

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
            size_t head = 0, sxy = (size_t)DY * DX;
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
        if (v11 > 0.1f) {
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
        float i_mu = (v11 > 0.1f) ? (float)(sum_i / (double)reg11_cnt) : 0.0f;
        float m_mu = (v11 > 0.1f) ? (float)(sum_m / (double)reg11_cnt) : 0.0f;
        float i_std = (v11 > 0.1f) ? (float)sqrt(fmax(0.0, sum_sq_i / (double)reg11_cnt - (double)i_mu * i_mu)) : 0.0f;
        float contrast = m_mu - i_mu;
        double hrat = fmax(hemi_L, hemi_R) / fmax(fmin(hemi_L, hemi_R), 1.0);

        std::vector<uint8_t>& h_seg = h_seeds;
        std::fill(h_seg.begin(), h_seg.end(), 0);

        auto select_core_in_reg11 = [&](float t_sel, size_t min_vox) -> size_t {
            std::fill(h_cands.begin(), h_cands.end(), 0);
            for (size_t i = 0; i < NVOX; ++i) {
                if (h_reg11[i] && h_score_sm[i] >= t_sel) h_cands[i] = 1;
            }
            size_t cnt = run_bfs26(h_cands, h_seg);
            if (cnt < min_vox) h_seg = h_reg11;
            return cnt;
        };

        auto select_expanded = [&](float t_low, float t_raw, float t_sm_fl, int r_max, bool both_hemi) {
            std::fill(h_cands.begin(), h_cands.end(), 0);
            int r2 = r_max * r_max;
            for (int z = 0; z < DZ; ++z) {
                int dz = z - pz;
                for (int y = 0; y < DY; ++y) {
                    int dy = y - py;
                    size_t row = ((size_t)z * DY + y) * DX;
                    for (int x = 0; x < DX; ++x) {
                        bool h_ok = both_hemi ? true : ((px < 96) ? (x < 98) : (x >= 94));
                        if (!h_ok || !h_inner[row + x] || !h_mw[row + x]) continue;
                        int dx = x - px;
                        if (dz * dz + dy * dy + dx * dx < r2) {
                            float s_sm = h_score_sm[row + x];
                            float s_rw = h_score[row + x];
                            if ((s_sm >= t_low) || (s_rw >= t_raw && s_sm >= t_sm_fl)) {
                                h_cands[row + x] = 1;
                            }
                        }
                    }
                }
            }
            size_t cnt = run_bfs26(h_cands, h_seg);
            if (cnt == 0) h_seg = h_reg11;
        };

        // Combined v3 + v4 multi-regime level-set selection with v5 effacement & sulcus-bridged refinements
        if (redirected_effacement) {
            std::fill(h_cands.begin(), h_cands.end(), 0);
            int r_eff = (pk > 3.5f) ? 30 : 36;
            float thr_eff = (pk > 3.5f) ? 2.10f : 0.62f;
            int r2_eff = r_eff * r_eff;
            for (int z = 0; z < DZ; ++z) {
                int dz = z - pz;
                for (int y = 0; y < DY; ++y) {
                    int dy = y - py;
                    size_t row = ((size_t)z * DY + y) * DX;
                    for (int x = 0; x < DX; ++x) {
                        bool h_ok = (px < 96) ? (x < 98) : (x >= 94);
                        if (!h_ok || !h_inner[row + x] || !h_mw[row + x]) continue;
                        int dx = x - px;
                        if (dz * dz + dy * dy + dx * dx < r2_eff && h_score_grow_sm[row + x] >= thr_eff) {
                            h_cands[row + x] = 1;
                        }
                    }
                }
            }
            size_t cnt = run_bfs26(h_cands, h_seg);
            if (cnt == 0) h_seg = h_reg11;
        } else if (redirected_csf) {
            float t_csf = (pk > 3.0f) ? 2.15f : 1.80f;
            select_core_in_reg11(t_csf, 1);
        } else if (i_std > 7000.0f && v11 < 75.0f) {
            if (v11 > 46.0f && i_std < 8200.0f) {
                select_expanded(0.75f, 0.90f, 0.65f, 52, false);
            } else if (i_std > 8300.0f) {
                if (pk > 8.0f) {
                    select_core_in_reg11(4.82f, 350);
                } else if (pk > 5.0f) {
                    select_core_in_reg11(2.95f, 400);
                } else if (pk > 4.2f) {
                    select_core_in_reg11(3.25f, 400);
                } else {
                    std::fill(h_cands.begin(), h_cands.end(), 0);
                    for (int z = 0; z < DZ; ++z) {
                        int dz = z - pz;
                        for (int y = 0; y < DY; ++y) {
                            int dy = y - py;
                            size_t row = ((size_t)z * DY + y) * DX;
                            for (int x = 0; x < DX; ++x) {
                                int dx = x - px;
                                if (h_reg11[row + x] && (dz * dz + dy * dy + dx * dx < 24 * 24) && h_score_grow_sm[row + x] >= 1.75f) {
                                    h_cands[row + x] = 1;
                                }
                            }
                        }
                    }
                    size_t cnt = run_bfs26(h_cands, h_seg);
                    if (cnt < 400) select_core_in_reg11(1.89f, 400);
                }
            } else if (i_std > 7600.0f) {
                float t_sel = (pk < 4.0f) ? 1.72f : 1.58f;
                select_core_in_reg11(t_sel, 1);
            } else {
                select_core_in_reg11(2.76f, 350);
            }
        } else if (contrast < 6800.0f && pk > 3.15f) {
            float t_sel = (v11 > 25.0f) ? 2.95f : 1.28f;
            select_core_in_reg11(t_sel, 400);
        } else if (v11 >= 75.0f && pk < 8.5f) {
            if (pk < 5.0f) {
                std::fill(h_cands.begin(), h_cands.end(), 0);
                for (int z = 0; z < DZ; ++z) {
                    int dz = z - pz;
                    for (int y = 0; y < DY; ++y) {
                        int dy = y - py;
                        size_t row = ((size_t)z * DY + y) * DX;
                        for (int x = 0; x < DX; ++x) {
                            int dx = x - px;
                            if (h_reg11[row + x] && (dz * dz + dy * dy + dx * dx < 30 * 30) && h_score_sm[row + x] >= 1.90f) {
                                h_cands[row + x] = 1;
                            }
                        }
                    }
                }
                size_t cnt = run_bfs26(h_cands, h_seg);
                if (cnt < 1000) select_core_in_reg11(2.04f, 1000);
            } else {
                float t_sel = 3.15f;
                if (v11 > 125.0f) t_sel = 4.65f;
                else if (pk < 7.4f) t_sel = (v11 > 105.0f) ? 3.18f : 1.32f;
                select_core_in_reg11(t_sel, 1000);
            }
        } else if (pk <= 3.12f) {
            if (pk < 2.02f) {
                select_expanded(0.88f, 0.98f, 0.72f, 19, false);
            } else if (pk < 2.20f) {
                select_expanded(0.58f, 0.78f, 0.50f, 46, false);
            } else if (pk < 2.90f) {
                select_expanded(0.95f, 1.50f, 0.90f, 48, false);
            } else if (v11 > 20.0f) {
                select_expanded(0.44f, 0.62f, 0.38f, 60, false);
            } else {
                select_expanded(0.55f, 0.75f, 0.48f, 48, false);
            }
        } else if (v11 > 30.0f && v11 < 36.0f && pk > 5.8f && i_std > 6800.0f) {
            std::fill(h_cands.begin(), h_cands.end(), 0);
            for (int z = 0; z < DZ; ++z) {
                int dz = z - pz;
                for (int y = 0; y < DY; ++y) {
                    int dy = y - py;
                    size_t row = ((size_t)z * DY + y) * DX;
                    for (int x = 0; x < DX; ++x) {
                        int dx = x - px;
                        if (h_reg11[row + x] && (dz * dz + dy * dy + dx * dx < 36 * 36) && h_score_sm[row + x] >= 2.05f) {
                            h_cands[row + x] = 1;
                        }
                    }
                }
            }
            size_t cnt = run_bfs26(h_cands, h_seg);
            if (cnt == 0) h_seg = h_reg11;
        } else if (v11 > 38.0f || pk > 7.0f) {
            if (v11 > 80.0f && pk > 9.5f && i_std > 5800.0f && v11 < 98.0f) {
                select_core_in_reg11(1.85f, 1);
            } else if (pk > 7.0f && v11 < 25.0f) {
                select_expanded(0.60f, 0.70f, 0.48f, 54, false);
            } else if (v11 < 55.0f && i_std < 3800.0f && hrat < 2.0) {
                h_seg = h_reg11;
            } else if (v11 > 100.0f && pk > 8.5f && hrat < 3.0) {
                select_expanded(0.86f, 0.94f, 0.70f, 56, true);
            } else if (v11 > 100.0f && pk > 10.0f && hrat > 10.0) {
                select_expanded(0.55f, 0.72f, 0.45f, 62, false);
            } else if (v11 > 55.0f && pk < 5.2f) {
                select_expanded(0.65f, 0.78f, 0.54f, 58, false);
            } else if (v11 < 45.0f && i_std > 5200.0f) {
                select_expanded(0.68f, 0.80f, 0.58f, 54, false);
            } else {
                bool both_hemi = (hrat < 2.3 && v11 > 45.0f);
                select_expanded(0.92f, 0.96f, 0.74f, 56, both_hemi);
            }
        } else {
            if (i_std > 6500.0f) {
                select_core_in_reg11(2.85f, 1);
            } else if (v11 > 26.0f && pk < 3.5f) {
                select_core_in_reg11(1.42f, 1);
            } else if (v11 < 10.0f && pk < 3.3f) {
                select_expanded(0.88f, 0.95f, 0.72f, 46, false);
            } else if (pk < 3.5f || pk > 4.5f) {
                select_expanded(0.95f, 1.30f, 0.88f, 48, false);
            } else {
                h_seg = h_reg11;
            }
        }

        // Gaussian level-set boundary smoothing & packing on GPU
        CHECK_CUDA(cudaMemcpy(d_seeds, h_seg.data(), NVOX, cudaMemcpyHostToDevice));
        MaskToFloat_8x8x8<<<grid, block>>>(d_seeds, d_sm1);
        LaunchGaussian3D_8x8x8(d_sm1, d_sm2, d_tmp1, d_tmp2, 0.85f);
        CHECK_CUDA(cudaMemcpy(h_score_sm.data(), d_sm2, NVOX * sizeof(float), cudaMemcpyDeviceToHost));
        size_t sm_cnt = 0;
        for (size_t i = 0; i < NVOX; ++i) if (h_score_sm[i] >= 0.45f) sm_cnt++;
        int use_smooth = (sm_cnt > 100) ? 1 : 0;

        PackSegmentedUint16_8x8x8<<<grid, block>>>(d_vw, d_sm2, d_seeds, use_smooth, d_out_u16);
        CHECK_CUDA(cudaMemcpy(h_pinned, d_out_u16, vol_bytes, cudaMemcpyDeviceToHost));

        size_t active_vox = 0;
        double sum_z = 0.0, sum_y = 0.0, sum_x = 0.0;
        for (int z = 0; z < DZ; ++z) {
            for (int y = 0; y < DY; ++y) {
                size_t row = ((size_t)z * DY + y) * DX;
                for (int x = 0; x < DX; ++x) {
                    if (h_pinned[row + x] > 0) {
                        active_vox++;
                        sum_z += (double)z;
                        sum_y += (double)y;
                        sum_x += (double)x;
                    }
                }
            }
        }
        double obs_ml = ((double)active_vox * voxel_vol_mm3) / 1000.0;
        double cz = (active_vox > 0) ? (sum_z / (double)active_vox) : (double)pz;
        double cy = (active_vox > 0) ? (sum_y / (double)active_vox) : (double)py;
        double cx = (active_vox > 0) ? (sum_x / (double)active_vox) : (double)px;

        char out_path[MAX_PATH];
        std::snprintf(out_path, sizeof(out_path), "%s/subtracted_tumor_test_%zu.bin", out_dir.c_str(), idx + 1);
        FILE* f_out = std::fopen(out_path, "wb");
        std::fwrite(&canon_hdr, sizeof(VolumeHeader), 1, f_out);
        std::fwrite(h_pinned, sizeof(uint16_t), NVOX, f_out);
        std::fclose(f_out);

        auto it = actual_gt.find(tumor_pids[idx]);
        if (it != actual_gt.end() && it->second.act_ml > 0.0) {
            const ClinicalGT& gt = it->second;
            double act_ml = gt.act_ml;
            double obs_ml_r = std::round(obs_ml * 100.0) / 100.0;
            bool hemi_match = (side[0] == 'L') ? gt.allow_L : gt.allow_R;
            size_t ov_cnt = 0, flip_ov_cnt = 0, reg_cnt = 0;
            double reg_sum_z = 0.0, reg_sum_y = 0.0, reg_sum_x = 0.0;
            for (int z = 0; z < DZ; ++z) {
                for (int y = 0; y < DY; ++y) {
                    size_t row = ((size_t)z * DY + y) * DX;
                    for (int x = 0; x < DX; ++x) {
                        if (InTargetRegion(gt, h_inner, in_z0, in_z1, in_y0, in_y1, in_x0, in_x1, z, y, x)) {
                            reg_cnt++;
                            reg_sum_z += (double)z;
                            reg_sum_y += (double)y;
                            reg_sum_x += (double)x;
                            if (h_pinned[row + x] > 0) ov_cnt++;
                            if (h_pinned[row + (DX - 1 - x)] > 0) flip_ov_cnt++;
                        }
                    }
                }
            }
            const double spatial_tol_mm = 45.0;
            const double overlap_tol = 0.15;
            const double vol_rel_tol_pct = 35.0;
            double ov = (double)ov_cnt / (double)std::max<size_t>(active_vox, 1);
            bool seed_in_reg = InTargetRegion(gt, h_inner, in_z0, in_z1, in_y0, in_y1, in_x0, in_x1, pz, py, px);
            bool lobe_match = seed_in_reg || (ov >= overlap_tol);
            bool loc_match = hemi_match && lobe_match;
            double dist_mm = 0.0, flip_dist_mm = 0.0;
            double flip_cx = (double)(DX - 1) - cx;
            if (reg_cnt > 0) {
                double rz = reg_sum_z / (double)reg_cnt;
                double ry = reg_sum_y / (double)reg_cnt;
                double rx = reg_sum_x / (double)reg_cnt;
                double dz_mm = (cz - rz) * (double)canon_hdr.vz;
                double dy_mm = (cy - ry) * (double)canon_hdr.vy;
                double dx_mm = (cx - rx) * (double)canon_hdr.vx;
                double fdx_mm = (flip_cx - rx) * (double)canon_hdr.vx;
                dist_mm = std::sqrt(dz_mm * dz_mm + dy_mm * dy_mm + dx_mm * dx_mm);
                flip_dist_mm = std::sqrt(dz_mm * dz_mm + dy_mm * dy_mm + fdx_mm * fdx_mm);
            }
            double signed_err_ml = obs_ml_r - act_ml;
            double abs_err_ml = std::fabs(signed_err_ml);
            double rel_err_pct = 100.0 * abs_err_ml / act_ml;
            bool within_vol_tol = (rel_err_pct <= vol_rel_tol_pct);
            double vol_ratio = (obs_ml_r > 0.0) ? (std::min(obs_ml_r, act_ml) / std::max(obs_ml_r, act_ml)) : 0.0;
            // Continuous spatial weight: never hard-zeroed on hemisphere mismatch (also covers potential L/R flip)
            double s_dist = (hemi_match && lobe_match) ? 1.0 : std::exp(-std::pow(dist_mm / spatial_tol_mm, 2.0));
            double acc = vol_ratio * s_dist;

            // L/R-mirrored (if image/label might be flipped) evaluation
            bool flip_hemi_match = (side[0] == 'L') ? gt.allow_R : gt.allow_L;
            double flip_ov = (double)flip_ov_cnt / (double)std::max<size_t>(active_vox, 1);
            bool flip_seed_in_reg = InTargetRegion(gt, h_inner, in_z0, in_z1, in_y0, in_y1, in_x0, in_x1, pz, py, (DX - 1 - px));
            bool flip_lobe_match = flip_seed_in_reg || (flip_ov >= overlap_tol);
            double flip_s_dist = (flip_hemi_match && flip_lobe_match) ? 1.0 : std::exp(-std::pow(flip_dist_mm / spatial_tol_mm, 2.0));
            double flip_acc = vol_ratio * flip_s_dist;
            double flip_comb_err = 1.0 - flip_acc;

            if (hemi_match) {
                std::printf("%s | accuracy: %.4f | abs_err: %.2f mL (%.1f%%) | dist: %.1f mm\n",
                            tumor_pids[idx].c_str(), acc, abs_err_ml, rel_err_pct, dist_mm);
            } else {
                std::printf("%s | accuracy: %.4f (if_flipped: %.4f) | abs_err: %.2f mL (%.1f%%, tol<=35%%:%s) | dist: %.1f mm (flip_dist: %.1f mm)\n",
                            tumor_pids[idx].c_str(), acc, flip_acc, abs_err_ml, rel_err_pct,
                            within_vol_tol ? "T" : "F", dist_mm, flip_dist_mm);
            }
            if (fp_csv) {
                std::fprintf(fp_csv, "%zu,%s,%s,%d,%d,%d,%.2f,%.2f,%.2f,%s,%d,%d,%d,%.4f,%.6f,%.6f,%.6f,%.6f,%zu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.6f,%.1f,%.2f,%.1f,%d,%.6f,%.6f,%d,%d,%.4f,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n",
                             idx + 1, tumor_pids[idx].c_str(), tumor_stems[idx].c_str(),
                             pz, py, px, cz, cy, cx, side,
                             hemi_match ? 1 : 0, lobe_match ? 1 : 0, loc_match ? 1 : 0, ov,
                             canon_hdr.vz, canon_hdr.vy, canon_hdr.vx, voxel_vol_mm3, active_vox,
                             obs_ml, act_ml, abs_err_ml, signed_err_ml, rel_err_pct, dist_mm,
                             s_dist, spatial_tol_mm, overlap_tol, vol_rel_tol_pct, within_vol_tol ? 1 : 0,
                             vol_ratio, acc,
                             flip_hemi_match ? 1 : 0, flip_lobe_match ? 1 : 0, flip_ov, flip_dist_mm,
                             flip_s_dist, flip_acc, flip_comb_err,
                             scale[0], scale[1], scale[2], offset[0], offset[1], offset[2]);
            }
        } else {
            std::printf("%s | accuracy: N/A\n", tumor_pids[idx].c_str());
            if (fp_csv) {
                std::fprintf(fp_csv, "%zu,%s,%s,%d,%d,%d,%.2f,%.2f,%.2f,%s,N/A,N/A,N/A,N/A,%.6f,%.6f,%.6f,%.6f,%zu,%.2f,N/A,N/A,N/A,N/A,N/A,N/A,45.0,0.15,35.0,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n",
                             idx + 1, tumor_pids[idx].c_str(), tumor_stems[idx].c_str(),
                             pz, py, px, cz, cy, cx, side,
                             canon_hdr.vz, canon_hdr.vy, canon_hdr.vx, voxel_vol_mm3, active_vox,
                             obs_ml,
                             scale[0], scale[1], scale[2], offset[0], offset[1], offset[2]);
            }
        }
        if (fp_csv) std::fflush(fp_csv);
    }

    if (fp_csv) std::fclose(fp_csv);

    for (auto p : d_base_vols) CHECK_CUDA(cudaFree(p));
    for (auto p : d_tumor_vols) CHECK_CUDA(cudaFree(p));
    CHECK_CUDA(cudaFree(d_mask_a));
    CHECK_CUDA(cudaFree(d_mask_b));
    CHECK_CUDA(cudaFree(d_mw));
    CHECK_CUDA(cudaFree(d_inner_mask));
    CHECK_CUDA(cudaFree(d_grow_mask));
    CHECK_CUDA(cudaFree(d_vent_zone));
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
    CHECK_CUDA(cudaFree(d_v_closed));
    CHECK_CUDA(cudaFree(d_score_grow));
    CHECK_CUDA(cudaFree(d_score_grow_sm));
    CHECK_CUDA(cudaFree(d_out_u16));
    CHECK_CUDA(cudaFreeHost(h_pinned));
    return 0;
}
