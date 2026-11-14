#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
align_canonical.py
==================
Canonical Alignment & Volume-Preserving Header Stage (v4 / new_code_v5).

Reads the exported baseline and patient T1w binary volumes (exp_baseline_u16/*.bin,
exp_tumor_u16/sub-RS*_T1w.bin), performs Stage 1 intracranial mask extraction and
Stage 2 [2%, 98%] canonical affine alignment, and writes volume-preserving pre-aligned
canonical binaries whose 18-byte headers carry the transformed physical voxel dimensions:
    (vz', vy', vx') = (vz * scale_z, vy * scale_y, vx * scale_x)
so original physical brain & tumor volumes are preserved before running CUDA tumor
segmentation in detect_tumor.cu.
"""
import os
import re
import csv
import glob
import struct
import numpy as np
from scipy import ndimage as ndi

DZ, DY, DX = 192, 224, 192
HEADER_FMT = "<3H3f"
TGT_LO = np.array([45.0, 25.0, 30.0], dtype=np.float32)
TGT_HI = np.array([165.0, 195.0, 162.0], dtype=np.float32)

ST6 = ndi.generate_binary_structure(3, 1)
ST26 = np.ones((3, 3, 3), dtype=bool)

Z_IDX, Y_IDX, X_IDX = np.ogrid[:DZ, :DY, :DX]
CENTER_BOX = (Z_IDX > 55) & (Z_IDX < 145) & (Y_IDX > 50) & (Y_IDX < 175) & (X_IDX > 45) & (X_IDX < 147)

# Precompute canonical integer coordinate grids for vectorized trilinear & NN warp
GZ, GY, GX = np.meshgrid(
    np.arange(DZ, dtype=np.float32),
    np.arange(DY, dtype=np.float32),
    np.arange(DX, dtype=np.float32),
    indexing="ij",
)


def extract_and_canonicalize(vol_u16: np.ndarray, orig_spacing: tuple[float, float, float]):
    """Run Stage 1 intracranial extraction & Stage 2 canonical affine alignment."""
    # Stage 1: Intracranial parenchyma mask
    m = (vol_u16 >= 14500) & (vol_u16 <= 41500)
    m[:42, :, :] = False

    er = m.copy()
    for _ in range(3):
        er = ndi.binary_erosion(er, structure=ST6, border_value=0)

    lab, n_comp = ndi.label(er, structure=ST6)
    if n_comp > 0:
        counts = np.bincount(lab[CENTER_BOX], minlength=n_comp + 1)
        counts[0] = 0
        best_id = int(np.argmax(counts))
        core = (lab == best_id) if counts[best_id] > 0 else np.zeros_like(er, dtype=bool)
    else:
        core = np.zeros_like(er, dtype=bool)
    core = core | core[:, :, ::-1]

    dil = core
    for _ in range(4):
        dil = ndi.binary_dilation(dil, structure=ST26)
    dil = dil & (vol_u16 > 500) & (vol_u16 < 42500)
    dil = dil | dil[:, :, ::-1]

    for _ in range(4):
        dil = ndi.binary_dilation(dil, structure=ST26)
    for _ in range(4):
        dil = ndi.binary_erosion(dil, structure=ST26, border_value=0)
    mask_final = (dil | dil[:, :, ::-1]) & (vol_u16 > 0)

    # Stage 2: Exact [2%, 98%] quantile bounding box -> canonical extent
    zz, yy, xx = np.nonzero(mask_final)
    z0, z1 = float(np.percentile(zz, 2.0)), float(np.percentile(zz, 98.0))
    y0, y1 = float(np.percentile(yy, 2.0)), float(np.percentile(yy, 98.0))
    x0, x1 = float(np.percentile(xx, 2.0)), float(np.percentile(xx, 98.0))

    lo = np.array([z0, y0, x0], dtype=np.float32)
    hi = np.array([z1, y1, x1], dtype=np.float32)
    scale = (hi - lo) / (TGT_HI - TGT_LO)
    offset = lo - TGT_LO * scale

    fz = np.float32(scale[0]) * GZ + np.float32(offset[0])
    fy = np.float32(scale[1]) * GY + np.float32(offset[1])
    fx = np.float32(scale[2]) * GX + np.float32(offset[2])

    # Nearest-neighbor mask warp matching CUDA llroundf
    nz = np.rint(fz).astype(np.int32)
    ny = np.rint(fy).astype(np.int32)
    nx = np.rint(fx).astype(np.int32)
    in_nn = (nz >= 0) & (nz < DZ) & (ny >= 0) & (ny < DY) & (nx >= 0) & (nx < DX)
    mw = np.zeros((DZ, DY, DX), dtype=np.uint8)
    mw[in_nn] = mask_final[nz[in_nn], ny[in_nn], nx[in_nn]].astype(np.uint8)

    # Masked trilinear volume warp matching CUDA AffineWarpCanonical_8x8x8
    masked_src = np.where(mask_final, vol_u16, np.uint16(0)).astype(np.float32)
    in_tri = (fz >= 0.0) & (fz <= DZ - 1) & (fy >= 0.0) & (fy <= DY - 1) & (fx >= 0.0) & (fx <= DX - 1)
    fz_c = np.where(in_tri, fz, 0.0)
    fy_c = np.where(in_tri, fy, 0.0)
    fx_c = np.where(in_tri, fx, 0.0)

    iz0 = np.floor(fz_c).astype(np.int32)
    iy0 = np.floor(fy_c).astype(np.int32)
    ix0 = np.floor(fx_c).astype(np.int32)
    iz1 = np.minimum(DZ - 1, iz0 + 1)
    iy1 = np.minimum(DY - 1, iy0 + 1)
    ix1 = np.minimum(DX - 1, ix0 + 1)

    wz = fz_c - iz0.astype(np.float32)
    wy = fy_c - iy0.astype(np.float32)
    wx = fx_c - ix0.astype(np.float32)

    c000 = masked_src[iz0, iy0, ix0]
    c001 = masked_src[iz0, iy0, ix1]
    c010 = masked_src[iz0, iy1, ix0]
    c011 = masked_src[iz0, iy1, ix1]
    c100 = masked_src[iz1, iy0, ix0]
    c101 = masked_src[iz1, iy0, ix1]
    c110 = masked_src[iz1, iy1, ix0]
    c111 = masked_src[iz1, iy1, ix1]

    c00 = c000 * (1.0 - wx) + c001 * wx
    c01 = c010 * (1.0 - wx) + c011 * wx
    c10 = c100 * (1.0 - wx) + c101 * wx
    c11 = c110 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c01 * wy
    c1 = c10 * (1.0 - wy) + c11 * wy
    vw = np.where(in_tri, c0 * (1.0 - wz) + c1 * wz, 0.0)
    vw_u16 = np.clip(np.rint(vw), 0, 65535).astype("<u2")

    # Preserve physical volume per voxel after canonical scaling:
    # (vz', vy', vx') = (vz * s_z, vy * s_y, vx * s_x)
    vz_o, vy_o, vx_o = orig_spacing
    vz_p = float(vz_o * float(scale[0]))
    vy_p = float(vy_o * float(scale[1]))
    vx_p = float(vx_o * float(scale[2]))

    orig_vox = int(mask_final.sum())
    aligned_vox = int(mw.sum())
    orig_ml = orig_vox * vz_o * vy_o * vx_o / 1000.0
    recon_ml = aligned_vox * vz_p * vy_p * vx_p / 1000.0

    return vw_u16, mw, scale, offset, (vz_p, vy_p, vx_p), orig_vox, aligned_vox, orig_ml, recon_ml


def process_folder(paths: list[str], out_dir: str, manifest_csv: str | None = None):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"RS(\d+)", stem, re.IGNORECASE)
        pid = f"RS{int(m.group(1)):03d}" if m else stem
        with open(p, "rb") as f:
            hdr = struct.unpack(HEADER_FMT, f.read(18))
            arr = np.frombuffer(f.read(), dtype="<u2").reshape(hdr[:3])
        orig_sp = (float(hdr[3]), float(hdr[4]), float(hdr[5]))

        vw_u16, mw, scale, offset, aligned_sp, orig_vox, aligned_vox, orig_ml, recon_ml = extract_and_canonicalize(arr, orig_sp)

        out_vol_path = os.path.join(out_dir, f"{stem}.bin")
        out_mask_path = os.path.join(out_dir, f"{stem}_mask.bin")
        out_hdr = struct.pack(HEADER_FMT, DZ, DY, DX, aligned_sp[0], aligned_sp[1], aligned_sp[2])
        with open(out_vol_path, "wb") as fv:
            fv.write(out_hdr)
            fv.write(vw_u16.tobytes(order="C"))
        with open(out_mask_path, "wb") as fm:
            fm.write(mw.tobytes(order="C"))

        rows.append({
            "stem": stem,
            "rs_number": pid,
            "orig_vz": f"{orig_sp[0]:.6f}",
            "orig_vy": f"{orig_sp[1]:.6f}",
            "orig_vx": f"{orig_sp[2]:.6f}",
            "scale_z": f"{float(scale[0]):.6f}",
            "scale_y": f"{float(scale[1]):.6f}",
            "scale_x": f"{float(scale[2]):.6f}",
            "offset_z": f"{float(offset[0]):.6f}",
            "offset_y": f"{float(offset[1]):.6f}",
            "offset_x": f"{float(offset[2]):.6f}",
            "aligned_vz": f"{aligned_sp[0]:.6f}",
            "aligned_vy": f"{aligned_sp[1]:.6f}",
            "aligned_vx": f"{aligned_sp[2]:.6f}",
            "orig_brain_vox": orig_vox,
            "aligned_brain_vox": aligned_vox,
            "orig_brain_ml": f"{orig_ml:.2f}",
            "reconstructed_brain_ml": f"{recon_ml:.2f}",
        })

    if manifest_csv:
        with open(manifest_csv, "w", newline="", encoding="utf-8") as fc:
            w = csv.DictWriter(fc, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    v5_dir = os.path.join(src_dir, "new_code_v5")

    base_paths = sorted(glob.glob(os.path.join(src_dir, "exp_baseline_u16", "*.bin")))
    tumor_paths = sorted(glob.glob(os.path.join(src_dir, "exp_tumor_u16", "sub-RS*_T1w.bin")))
    if not base_paths or not tumor_paths:
        raise FileNotFoundError("Missing baseline or tumor .bin inputs.")

    base_out = os.path.join(v5_dir, "aligned_baseline_u16")
    tumor_out = os.path.join(v5_dir, "aligned_tumor_u16")

    process_folder(base_paths, base_out, os.path.join(base_out, "alignment_manifest.csv"))
    t_rows = process_folder(tumor_paths, tumor_out, os.path.join(tumor_out, "alignment_manifest.csv"))
    print(f"Aligned {len(base_paths)} baseline + {len(t_rows)} tumor volumes in Python with volume-preserving (vz', vy', vx') headers.")


if __name__ == "__main__":
    main()
