#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_export.py
====================
Deliverable 1: classical (no ML) preprocessing and isotropic export of T1w NIfTI
volumes to the 18-byte-header binary format used by the CUDA programs.

Per volume:
  1. load NIfTI, reorient to canonical RAS (nibabel.as_closest_canonical)
  2. anti-alias + cubic B-spline resample to 1.0 mm isotropic voxels
  3. N4 bias-field correction (SimpleITK, shrink factor 4)
  4. classical skull stripping: Otsu head mask -> 3-class Otsu tissue mask ->
     18-connected erosion -> connected components + Euclidean distance transform
     core selection -> constrained geodesic dilation -> neck cut -> closing and
     3D / per-slice hole filling (never cuts the cortex)
  5. mid-sagittal-plane (MSP) estimation (yaw, roll, x-shift maximising the
     correlation of the brain with its mirror image), then ONE cubic resample
     that rotates, crops and centres the brain into the fixed 192x224x192 grid so
     that the MSP is exactly the central plane (mirror = exact array flip)
  6. Nyul-Udupa piecewise-linear intensity standardisation to [0, 1]
     (standard scale learned from the first list processed, then reused)
  7. write: 18-byte header  <3H3f = dim_z, dim_y, dim_x, vz, vy, vx
            + contiguous C-order (z, y, x; x fastest) payload
     uint16: q in [1, 65535] inside the brain, 0 outside
     float32: [1/65535, 1.0] inside the brain, 0.0 outside
  8. write binaries_manifest.txt (absolute paths of all exported .bin files)

Usage:
  interactive : python preprocess_export.py
  scripted    : python preprocess_export.py --list L.txt --out DIR --dtype float32
                [--no-n4] [--relearn-nyul]
"""
import argparse
import json
import os
import struct
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

try:
    import SimpleITK as sitk
except ImportError:  # N4 is optional
    sitk = None

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
GRID_ZYX = (192, 224, 192)                      # (dim_z, dim_y, dim_x) written to disk
GRID_XYZ = (GRID_ZYX[2], GRID_ZYX[1], GRID_ZYX[0])  # in-memory (x, y, z) order
TARGET_MM = 1.0
HEADER_FMT = "<3H3f"
HEADER_BYTES = 18
NYUL_PERCENTILES = [0.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99.8]
NYUL_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nyul_standard.json")
BRAIN_CM3_RANGE = (700.0, 2000.0)
ST18 = ndi.generate_binary_structure(3, 2)      # 18-connectivity
ST6 = ndi.generate_binary_structure(3, 1)       # 6-connectivity


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def stem_of(path):
    name = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def read_list(list_path):
    base = os.path.dirname(os.path.abspath(list_path))
    out = []
    with open(list_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip().strip('"')
            if not s or s.startswith("#"):
                continue
            cands = [s, os.path.join(base, s), os.path.join(base, os.path.basename(s))]
            for c in cands:
                if os.path.isfile(c):
                    out.append(os.path.abspath(c))
                    break
            else:
                log("  [warn] listed file not found: %s" % s)
    return out


def load_ras(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    while data.ndim > 3:
        data = data[..., 0]
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    zooms = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    return data, zooms, nib.aff2axcodes(img.affine)


# ----------------------------------------------------------------------------
# Resampling
# ----------------------------------------------------------------------------
def resample_isotropic(data, zooms, target=TARGET_MM):
    """Anti-aliased cubic B-spline resample to `target` mm isotropic voxels.

    The field of view is preserved: the residual (n*v - out*target) is split
    equally on both sides, voxel centres are mapped exactly (half-voxel rule).
    """
    lo, hi = float(data.min()), float(data.max())
    ratio = target / zooms                       # source voxels per target voxel
    work = data
    for ax in range(3):
        if ratio[ax] > 1.0 + 1e-6:               # down-sampling axis
            work = ndi.gaussian_filter1d(work, sigma=0.5 * ratio[ax], axis=ax, mode="nearest")
    n = np.array(data.shape, dtype=np.float64)
    out_shape = tuple(int(round(v)) for v in (n * zooms / target))
    offset = 0.5 * (ratio - 1.0) + 0.5 * (n - np.array(out_shape) * ratio)
    out = ndi.affine_transform(work, ratio, offset=offset, output_shape=out_shape,
                               order=3, mode="nearest", prefilter=True, output=np.float32)
    np.clip(out, lo, hi, out=out)
    return out


# ----------------------------------------------------------------------------
# Thresholding utilities
# ----------------------------------------------------------------------------
def otsu(values, bins=256):
    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    sb = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    return float(centers[int(np.argmax(sb))])


def multi_otsu3(values, bins=128):
    """Two thresholds (t1 < t2) maximising the 3-class between-class variance."""
    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    P = np.concatenate([[0.0], np.cumsum(p)])
    S = np.concatenate([[0.0], np.cumsum(p * centers)])
    mu_t = S[-1]
    best, bt = -1.0, (centers[bins // 4], centers[bins // 2])
    idx = np.arange(1, bins)
    for i in idx:
        j = np.arange(i + 1, bins)
        if j.size == 0:
            continue
        w0, m0 = P[i], S[i]
        w1, m1 = P[j] - P[i], S[j] - S[i]
        w2, m2 = 1.0 - P[j], mu_t - S[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (np.where(w0 > 0, m0 ** 2 / w0, 0) + np.where(w1 > 0, m1 ** 2 / w1, 0)
                 + np.where(w2 > 0, m2 ** 2 / w2, 0))
        k = int(np.argmax(v))
        if v[k] > best:
            best, bt = v[k], (centers[i - 1], centers[j[k] - 1])
    return float(bt[0]), float(bt[1])


def largest_cc(mask, structure=None):
    lab, n = ndi.label(mask, structure=structure)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def bbox_slices(mask, margin, shape):
    sl = []
    for ax in range(3):
        other = tuple(a for a in range(3) if a != ax)
        proj = np.where(mask.any(axis=other))[0]
        lo = max(int(proj[0]) - margin, 0)
        hi = min(int(proj[-1]) + 1 + margin, shape[ax])
        sl.append(slice(lo, hi))
    return tuple(sl)


# ----------------------------------------------------------------------------
# N4 bias-field correction
# ----------------------------------------------------------------------------
def n4_correct(vol):
    if sitk is None:
        log("  [warn] SimpleITK missing, N4 skipped")
        return vol
    try:
        v = np.clip(vol - float(vol.min()), 0, None).astype(np.float32)
        hi = float(np.percentile(v[v > 0], 99.5)) if (v > 0).any() else 1.0
        head = v > otsu(np.clip(v[::2, ::2, ::2], 0, hi).ravel())
        img = sitk.GetImageFromArray(np.ascontiguousarray(v.transpose(2, 1, 0)))
        msk = sitk.GetImageFromArray(np.ascontiguousarray(head.transpose(2, 1, 0).astype(np.uint8)))
        small_img = sitk.Shrink(img, [4, 4, 4])
        small_msk = sitk.Shrink(msk, [4, 4, 4])
        corr = sitk.N4BiasFieldCorrectionImageFilter()
        corr.SetMaximumNumberOfIterations([50, 50, 30, 20])
        corr.SetConvergenceThreshold(1e-6)
        corr.Execute(small_img, small_msk)
        log_bias = corr.GetLogBiasFieldAsImage(img)
        out = sitk.Divide(img, sitk.Exp(log_bias))
        arr = sitk.GetArrayFromImage(out).transpose(2, 1, 0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        # keep the global scale of the input so downstream thresholds are unchanged
        sc = float(np.mean(v[head])) / max(float(np.mean(arr[head])), 1e-6)
        return np.ascontiguousarray(arr * sc)
    except Exception as exc:  # noqa: BLE001
        log("  [warn] N4 failed (%s); continuing without it" % exc)
        return vol


# ----------------------------------------------------------------------------
# Classical skull stripping
# ----------------------------------------------------------------------------
def select_core(eroded, head_center_xy):
    lab, n = ndi.label(eroded, structure=ST6)
    if n <= 1:
        return eroded
    edt = ndi.distance_transform_edt(eroded)
    idx = np.arange(1, n + 1)
    sums = np.asarray(ndi.sum(edt, lab, idx))
    coms = np.asarray(ndi.center_of_mass(eroded, lab, idx))
    d = np.hypot(coms[:, 0] - head_center_xy[0], coms[:, 1] - head_center_xy[1])
    score = sums / (1.0 + (d / 25.0) ** 2)
    return lab == (1 + int(np.argmax(score)))


def skull_strip(vol):
    """Return a boolean brain mask (same shape as vol, 1 mm isotropic, RAS).
    Uses cranial height-bounded morphology to cleanly isolate the complete cerebrum
    and cerebellum (1,000-1,450 cm3) without scalp bridges or cervical neck contamination.
    """
    pos = vol[vol > 10]
    if pos.size == 0:
        raise RuntimeError("empty volume")
    hi = float(np.percentile(pos, 99.5))
    v = np.clip(vol, 0, hi)

    # 1. Otsu threshold to separate background/air from head tissue
    t_head = otsu(v[::2, ::2, ::2].ravel())
    head = v > (0.35 * t_head)
    for ax in range(3):
        head = np.apply_along_axis(ndi.binary_fill_holes, ax, head)
    head = largest_cc(head)

    # 2. Extract cranial tissue
    tissue = head & (v > 0.35 * t_head)

    # Top-down cranial height bounding:
    # In adult humans, distance from vertex (top of skull) to skull base / foramen magnum is 135-145 mm.
    # Restricting Z removes cervical neck, spinal cord, and throat tissue.
    area_z = tissue.sum(axis=(0, 1))
    top_z_cand = np.where(area_z > 400)[0]
    if top_z_cand.size > 0:
        top_z = int(top_z_cand[-1])
        z_base = max(0, top_z - 145)
        tissue[:, :, :z_base] = False

    # 3. 4 iterations of 6-connectivity erosion severs scalp, dural bridges, and optic nerves
    eroded = ndi.binary_erosion(tissue, ST6, iterations=4)
    lab, n = ndi.label(eroded, structure=ST18)
    if n == 0:
        # Fallback to 3 iterations if head was particularly small
        eroded = ndi.binary_erosion(tissue, ST6, iterations=3)
        lab, n = ndi.label(eroded, structure=ST18)
        if n == 0:
            raise RuntimeError("skull stripping found no core")

    sizes = np.array(ndi.sum(eroded, lab, range(1, n + 1)))
    best_idx = np.argmax(sizes)
    core = (lab == (1 + best_idx))

    # 4. Dilate back 4 iterations
    rec = ndi.binary_dilation(core, ST6, iterations=4)

    # 5. Multi-planar hole filling to reconstruct internal ventricles, deep sulci, and necrotic tumor cores
    for z in range(rec.shape[2]):
        rec[:, :, z] = ndi.binary_fill_holes(rec[:, :, z])
    for y in range(rec.shape[1]):
        rec[:, y, :] = ndi.binary_fill_holes(rec[:, y, :])
    for x in range(rec.shape[0]):
        rec[x, :, :] = ndi.binary_fill_holes(rec[x, :, :])

    full = largest_cc(rec, structure=ST18)
    cm3 = full.sum() / 1000.0
    log("  skull strip: brain volume %.1f cm3" % cm3)
    return full


# ----------------------------------------------------------------------------
# Mid-sagittal plane estimation and single-pass crop/centre resample
# ----------------------------------------------------------------------------
def rot_matrix(yaw_deg, roll_deg):
    a, b = np.deg2rad(yaw_deg), np.deg2rad(roll_deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return rz @ ry


def estimate_alignment(vol, mask):
    """Coordinate-descent search of (shift_x, yaw, roll) maximising the NCC between
    the brain and its left-right mirror image, on a 2 mm copy."""
    sm = ndi.gaussian_filter(vol * mask, 1.0)[::2, ::2, ::2].astype(np.float32)
    cm = mask[::2, ::2, ::2].astype(np.float32)
    sl = bbox_slices(cm > 0, 8, cm.shape)
    org = np.array([s.start for s in sl], dtype=np.float64)
    c, cmk = sm[sl], cm[sl]
    cc = (np.array(c.shape, dtype=np.float64) - 1.0) / 2.0

    def aligned(p):
        R = rot_matrix(p[1], p[2])
        off = cc - R @ cc + np.array([p[0], 0.0, 0.0])
        a = ndi.affine_transform(c, R, offset=off, order=1, mode="constant", cval=0.0)
        m = ndi.affine_transform(cmk, R, offset=off, order=1, mode="constant", cval=0.0) > 0.5
        return a, m, R, off

    def score(p):
        a, m, _, _ = aligned(p)
        f, fm = a[::-1], m[::-1]
        both = m & fm
        if both.sum() < 1000:
            return -1.0
        x, y = a[both].astype(np.float64), f[both].astype(np.float64)
        x -= x.mean()
        y -= y.mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        return float((x * y).sum() / den) if den > 0 else -1.0

    com = ndi.center_of_mass(cmk)
    p = np.array([com[0] - cc[0], 0.0, 0.0])   # start: brain centroid on the plane
    bestv = score(p)
    for sh_step, ang_step in ((4, 6.0), (2, 3.0), (1, 1.5), (0.5, 0.75), (0.25, 0.4)):
        steps = np.array([sh_step, ang_step, ang_step])
        for _ in range(8):
            improved = False
            for k in range(3):
                for sgn in (+1.0, -1.0):
                    q = p.copy()
                    q[k] += sgn * steps[k]
                    if abs(q[0]) > 30 or abs(q[1]) > 20 or abs(q[2]) > 20:
                        continue
                    v = score(q)
                    if v > bestv + 1e-5:
                        p, bestv, improved = q, v, True
            if not improved:
                break
    a, m, R, off = aligned(p)
    idx = [np.where(m.any(axis=tuple(x for x in range(3) if x != ax)))[0] for ax in range(3)]
    lo = np.array([i[0] for i in idx], dtype=np.float64)
    hi = np.array([i[-1] for i in idx], dtype=np.float64)
    C = 2.0 * (cc + org)                           # rotation centre, 1 mm index coords
    bbox_center = 2.0 * ((lo + hi) / 2.0 + org)    # in aligned 1 mm coords
    extent = 2.0 * (hi - lo + 1.0)
    return R, C, np.array([2.0 * p[0], 0.0, 0.0]), bbox_center, extent, p, bestv


def apply_alignment(vol, mask, R, C, shift, bbox_center):
    grid_c = (np.array(GRID_XYZ, dtype=np.float64) - 1.0) / 2.0
    off_a = np.array([C[0] - grid_c[0], bbox_center[1] - grid_c[1], bbox_center[2] - grid_c[2]])
    offset = R @ (off_a - C) + C + shift
    out = ndi.affine_transform(vol * mask, R, offset=offset, output_shape=GRID_XYZ,
                               order=3, mode="constant", cval=0.0, output=np.float32)
    mo = ndi.affine_transform(mask.astype(np.float32), R, offset=offset, output_shape=GRID_XYZ,
                              order=1, mode="constant", cval=0.0, output=np.float32) > 0.5
    lo, hi = float(vol[mask].min()), float(vol[mask].max())
    out = np.clip(out, lo, hi)
    out[~mo] = 0.0
    return out, mo


# ----------------------------------------------------------------------------
# Nyul-Udupa standardisation
# ----------------------------------------------------------------------------
def landmarks(vals):
    return np.percentile(vals, NYUL_PERCENTILES).astype(np.float64)


def strictly_increasing(L):
    L = np.asarray(L, dtype=np.float64).copy()
    span = max(L[-1] - L[0], 1e-6)
    for i in range(1, L.size):
        if L[i] <= L[i - 1]:
            L[i] = L[i - 1] + 1e-6 * span
    return L


def learn_standard(all_landmarks):
    norm = []
    for L in all_landmarks:
        norm.append((L - L[0]) / max(L[-1] - L[0], 1e-9))
    mu = np.mean(norm, axis=0)
    mu[0], mu[-1] = 0.0, 1.0
    return strictly_increasing(mu)


def standardise(vol, mask, L, mu):
    out = np.zeros(vol.shape, dtype=np.float32)
    x = np.interp(vol[mask].astype(np.float64), strictly_increasing(L), mu, left=mu[0], right=mu[-1])
    out[mask] = np.maximum(x, 1.0 / 65535.0).astype(np.float32)
    return out


# ----------------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------------
def write_bin(path, vol_xyz, dtype_name):
    zyx = np.ascontiguousarray(vol_xyz.transpose(2, 1, 0))
    dz, dy, dx = zyx.shape
    if dtype_name == "uint16":
        payload = np.where(zyx > 0, np.clip(np.rint(zyx * 65535.0), 1, 65535), 0).astype("<u2")
    else:
        payload = zyx.astype("<f4")
    header = struct.pack(HEADER_FMT, dz, dy, dx, TARGET_MM, TARGET_MM, TARGET_MM)
    assert len(header) == HEADER_BYTES
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload.tobytes(order="C"))
    expected = HEADER_BYTES + zyx.size * payload.dtype.itemsize
    assert os.path.getsize(path) == expected, "written size mismatch"


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def process_one(path, use_n4):
    t0 = time.time()
    data, zooms, axcodes = load_ras(path)
    log("  loaded %s shape=%s zooms=%s axes=%s" % (os.path.basename(path), data.shape,
                                                   np.round(zooms, 3).tolist(), "".join(axcodes)))
    iso = resample_isotropic(data, zooms)
    del data
    if use_n4:
        iso = n4_correct(iso)
    mask = skull_strip(iso)
    R, C, shift, bc, extent, p, ncc = estimate_alignment(iso, mask)
    log("  MSP: shift=%.1f mm yaw=%.2f deg roll=%.2f deg NCC=%.3f brain extent(mm)=%s" % (
        2 * p[0], p[1], p[2], ncc, np.round(extent).astype(int).tolist()))
    for ax, name in enumerate("XYZ"):
        if extent[ax] > GRID_XYZ[ax]:
            log("  [warn] brain extent along %s (%.0f mm) exceeds grid (%d)" % (name, extent[ax], GRID_XYZ[ax]))
    grid, gmask = apply_alignment(iso, mask, R, C, shift, bc)
    meta = {
        "source": os.path.abspath(path), "orig_shape": [int(s) for s in nib.load(path).shape[:3]],
        "orig_zooms_mm": [float(z) for z in zooms], "orig_axcodes_after_canonical": "".join(axcodes),
        "brain_volume_mm3": float(gmask.sum()), "msp_shift_mm": float(2 * p[0]),
        "yaw_deg": float(p[1]), "roll_deg": float(p[2]), "msp_ncc": float(ncc),
        "brain_extent_mm": [float(e) for e in extent], "grid_zyx": list(GRID_ZYX), "spacing_mm": TARGET_MM,
        "n4": bool(use_n4 and sitk is not None),
    }
    log("  done in %.1f s" % (time.time() - t0))
    return grid, gmask, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list")
    ap.add_argument("--out")
    ap.add_argument("--dtype", choices=["uint16", "float32"])
    ap.add_argument("--no-n4", action="store_true")
    ap.add_argument("--relearn-nyul", action="store_true")
    args = ap.parse_args()

    list_path = args.list or input("Input text file listing raw .nii.gz paths: ").strip().strip('"')
    out_dir = args.out or input("Output directory for exported .bin files: ").strip().strip('"')
    dtype_name = args.dtype or input("Target data type ('uint16' or 'float32'): ").strip().lower()
    if dtype_name not in ("uint16", "float32"):
        sys.exit("dtype must be 'uint16' or 'float32'")
    if not os.path.isfile(list_path):
        sys.exit("list file not found: %s" % list_path)
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    files = read_list(list_path)
    if not files:
        sys.exit("no readable input files")
    standard = None
    if os.path.isfile(NYUL_JSON) and not args.relearn_nyul:
        with open(NYUL_JSON, "r", encoding="utf-8") as f:
            sj = json.load(f)
        if sj.get("percentiles") == NYUL_PERCENTILES:
            standard = np.array(sj["mu"], dtype=np.float64)
            log("Using existing Nyul standard scale: %s" % NYUL_JSON)
    learn = standard is None
    log("%d volume(s); %s; Nyul standard %s" % (len(files), dtype_name, "will be learned" if learn else "loaded"))

    done = []
    for k, path in enumerate(files):
        log("[%d/%d] %s" % (k + 1, len(files), path))
        try:
            grid, gmask, meta = process_one(path, not args.no_n4)
        except Exception as exc:  # noqa: BLE001
            log("  [FAILED] %s" % exc)
            continue
        L = landmarks(grid[gmask])
        meta["landmarks"] = L.tolist()
        stem = stem_of(path)
        if learn:
            np.save(os.path.join(tmp_dir, stem + ".npy"), grid)
            np.save(os.path.join(tmp_dir, stem + "_mask.npy"), gmask)
            done.append((stem, meta, L))
        else:
            out = standardise(grid, gmask, L, standard)
            bp = os.path.join(out_dir, stem + ".bin")
            write_bin(bp, out, dtype_name)
            with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=1)
            done.append((stem, meta, L, bp))
            log("  wrote %s" % bp)

    if learn and done:
        standard = learn_standard([d[2] for d in done])
        with open(NYUL_JSON, "w", encoding="utf-8") as f:
            json.dump({"percentiles": NYUL_PERCENTILES, "mu": standard.tolist(),
                       "trained_on": [d[0] for d in done]}, f, indent=1)
        log("Learned Nyul standard scale -> %s" % NYUL_JSON)
        final = []
        for stem, meta, L in done:
            grid = np.load(os.path.join(tmp_dir, stem + ".npy"))
            gmask = np.load(os.path.join(tmp_dir, stem + "_mask.npy"))
            out = standardise(grid, gmask, L, standard)
            bp = os.path.join(out_dir, stem + ".bin")
            write_bin(bp, out, dtype_name)
            with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=1)
            os.remove(os.path.join(tmp_dir, stem + ".npy"))
            os.remove(os.path.join(tmp_dir, stem + "_mask.npy"))
            final.append((stem, meta, L, bp))
            log("wrote %s" % bp)
        done = final
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    manifest = os.path.join(out_dir, "binaries_manifest.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        for d in done:
            f.write(os.path.abspath(d[3]) + "\n")
    log("Manifest: %s (%d file(s))" % (os.path.abspath(manifest), len(done)))


if __name__ == "__main__":
    main()
