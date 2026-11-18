#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_results.py

Root entry point synchronized with v5 Pure CUDA C/C++ segmentation outputs.
Generates multi-planar PNG visualizations and per-image JSON/Markdown reports
strictly displaying RS number, Observed Volume, Actual Volume, and Accuracy
with zero compute-speed or hardware telemetry.
"""
import os
import re
import csv
import json
import glob
import time
import struct
import shutil
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
V5_DIR = os.path.join(SRC_DIR, "new_code_v5")

SEG_DIR = os.path.join(V5_DIR, "cuda_segmented_out")
VIS_DIR = os.path.join(V5_DIR, "visualizations")
PAT_DIR = VIS_DIR
REP_DIR = os.path.join(V5_DIR, "reports")

for d in (VIS_DIR, PAT_DIR, REP_DIR):
    os.makedirs(d, exist_ok=True)

EXCLUDE = {"RS003", "RS005"}   # byte-identical T1w files in source dataset with conflicting labels
LOBES = ["frontal", "temporal", "parietal", "occipital", "insula", "deep", "peritrigonal"]


def load_demo():
    """Load clinical reference volume and tumor location descriptors from dataset/patient_demographics.csv."""
    demo = {}
    with open(os.path.join(SRC_DIR, "dataset", "patient_demographics.csv"), encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            m = re.search(r"RS\s*(\d+)", row[0], re.IGNORECASE)
            if not m:
                continue
            pid = "RS%03d" % int(m.group(1))
            loc = row[5].lower()
            vols = [float(x) for x in re.findall(r"[0-9.]+", row[6])]
            hemi = set()
            if "left" in loc:
                hemi.add("L")
            if "right" in loc or "rght" in loc:
                hemi.add("R")
            if "bilateral" in loc or not hemi:
                hemi = {"L", "R"}
            lobes = set()
            for k, keys in {
                "frontal": ["frontal"],
                "temporal": ["temporal", "sylvian", "slyvian"],
                "parietal": ["parietal", "splenium"],
                "occipital": ["occipital"],
                "insula": ["insula", "insular"],
                "deep": ["basal ganglia", "thalamus"],
                "peritrigonal": ["trigonal"],
            }.items():
                if any(s in loc for s in keys):
                    lobes.add(k)
            demo[pid] = dict(
                loc=row[5].strip(),
                sz_str=row[6].strip(),
                gt=sum(vols) if vols else 0.0,
                hemi=hemi,
                lobes=lobes or set(LOBES),
                patho=row[7].strip(),
            )
    return demo


def lobe_masks(brain):
    """Anatomical lobe regions in canonical intracranial coordinates (z inf->sup, y post->ant, x L->R)."""
    zz, yy, xx = np.nonzero(brain)
    z0, z1, y0, y1, x0, x1 = zz.min(), zz.max(), yy.min(), yy.max(), xx.min(), xx.max()
    fz = (np.arange(brain.shape[0])[:, None, None] - z0) / max(z1 - z0, 1)
    fy = (np.arange(brain.shape[1])[None, :, None] - y0) / max(y1 - y0, 1)
    fx = (np.arange(brain.shape[2])[None, None, :] - x0) / max(x1 - x0, 1)
    lat = np.abs(fx - 0.5)
    M = {
        "frontal":      (fy > 0.42) & (fz > 0.16),
        "temporal":     (fy > 0.18) & (fy < 0.72) & (fz < 0.68) & (lat > 0.06),
        "parietal":     (fy > 0.12) & (fy < 0.65) & (fz > 0.38),
        "occipital":    (fy < 0.35),
        "insula":       (fy > 0.35) & (fy < 0.72) & (fz > 0.20) & (fz < 0.68) & (lat > 0.10) & (lat < 0.38),
        "deep":         (fy > 0.30) & (fy < 0.70) & (fz > 0.20) & (fz < 0.68) & (lat < 0.28),
        "peritrigonal": (fy > 0.15) & (fy < 0.58) & (fz > 0.25) & (fz < 0.72) & (lat > 0.05) & (lat < 0.38),
    }
    left = np.broadcast_to(fx < 0.5, brain.shape)
    return {k: np.broadcast_to(v, brain.shape) & brain for k, v in M.items()}, left


def target_region_mask(d, lm, left, brain):
    """Return the reference anatomical target region mask for a clinical location record."""
    region = np.zeros_like(brain, dtype=bool)
    for lb in d["lobes"]:
        region |= lm[lb]
    hm = np.zeros_like(brain, dtype=bool)
    if "L" in d["hemi"]:
        hm |= left
    if "R" in d["hemi"]:
        hm |= ~left
    return region & hm & brain


def save_png_atomic(fig, out_p: str) -> None:
    """Save figure via temp file + atomic replace to avoid Windows file-lock collisions."""
    tmp_p = f"{out_p}.tmp.{os.getpid()}.png"
    fig.savefig(tmp_p, facecolor=fig.get_facecolor())
    plt.close(fig)
    for attempt in range(6):
        try:
            os.replace(tmp_p, out_p)
            return
        except OSError:
            if attempt == 5:
                raise
            time.sleep(0.15)


def window_t1w(vol_slice: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    """Percentile windowing with gentle gamma for T1w gray/white contrast."""
    norm = np.clip((vol_slice - p_lo) / max(p_hi - p_lo, 1.0), 0.0, 1.0)
    out = np.power(norm, 0.92)
    out[vol_slice <= 0] = 0.0
    return out


def compute_inner_core_mask(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Identify higher-severity inner region inside the segmented lesion."""
    core = np.zeros_like(mask, dtype=bool)
    zz, yy, xx = np.nonzero(mask)
    if zz.size == 0:
        return core
    z0, z1 = max(0, int(zz.min()) - 2), min(mask.shape[0], int(zz.max()) + 3)
    y0, y1 = max(0, int(yy.min()) - 2), min(mask.shape[1], int(yy.max()) + 3)
    x0, x1 = max(0, int(xx.min()) - 2), min(mask.shape[2], int(xx.max()) + 3)
    sub_m = mask[z0:z1, y0:y1, x0:x1]
    sub_v = vol[z0:z1, y0:y1, x0:x1]
    vals = sub_v[sub_m]
    dist = ndi.distance_transform_edt(sub_m)
    dev = np.abs(sub_v - 26500.0)
    dev_thr = float(np.percentile(dev[sub_m], 55.0)) if vals.size > 10 else 0.0
    sub_c = sub_m & ((dist >= 2.2) | (dev >= dev_thr))
    if sub_c.sum() > 0.75 * sub_m.sum():
        sub_c = sub_m & (dist >= max(2.5, float(np.percentile(dist[sub_m], 50.0))))
    core[z0:z1, y0:y1, x0:x1] = sub_c
    return core


def draw_multilayer_slice(ax, sl_w, sl_m, sl_core, sl_mir, sl_r, cross_x, cross_y, title, show_midline=False):
    ax.set_facecolor("#090d14")
    ax.imshow(sl_w, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", aspect="equal", interpolation="bilinear")

    if show_midline:
        ax.axvline(95.5, color="#ffffff", linestyle=":", linewidth=0.75, alpha=0.38)

    if sl_mir is not None and sl_mir.any():
        ax.contour(sl_mir.astype(float), levels=[0.5], colors=["#e040fb"], linewidths=1.0, linestyles="dotted", origin="lower")

    if sl_r is not None and sl_r.any():
        ax.contour(sl_r.astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=1.15, linestyles="dashed", origin="lower")

    if sl_m.any():
        ov_rgba = np.zeros((*sl_m.shape, 4), dtype=np.float32)
        ov_rgba[sl_m] = [1.0, 0.22, 0.12, 0.34]
        if sl_core.any():
            ov_rgba[sl_core] = [1.0, 0.58, 0.05, 0.46]
        ax.imshow(ov_rgba, origin="lower", aspect="equal")

        ax.contour(sl_m.astype(float), levels=[0.5], colors=["#ffeb3b"], linewidths=1.6, origin="lower")
        if sl_core.any() and sl_core.sum() < sl_m.sum():
            ax.contour(sl_core.astype(float), levels=[0.5], colors=["#00e676"], linewidths=1.1, origin="lower")

        ax.plot(cross_x, cross_y, marker="+", color="#ffffff", markersize=8, markeredgewidth=1.4, alpha=0.92)

    ax.set_title(title, color="#f0f6fc", fontsize=10.0, fontweight="bold", pad=6)
    ax.axis("off")


def main():
    demo = load_demo()
    inner_path = os.path.join(V5_DIR, "canonical_inner_mask.bin")
    inner_mask = np.fromfile(inner_path, dtype=np.uint8).reshape((192, 224, 192)) > 0
    lm, left = lobe_masks(inner_mask)

    summary_csv = os.path.join(SEG_DIR, "segmentation_summary_v5.csv")
    with open(summary_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    eval_records = []
    axial_thumbs = []

    for r in rows:
        idx = int(r["index"])
        pid = r["rs_number"]
        bin_path = os.path.join(SEG_DIR, f"subtracted_tumor_test_{idx}.bin")
        byte_sz = os.path.getsize(bin_path)
        assert byte_sz == 16515090, f"Unexpected size for {bin_path}: {byte_sz}"

        with open(bin_path, "rb") as bf:
            bin_hdr = struct.unpack("<3H3f", bf.read(18))
            sub_u16 = np.frombuffer(bf.read(), dtype="<u2").reshape(bin_hdr[:3])

        vz, vy, vx = float(bin_hdr[3]), float(bin_hdr[4]), float(bin_hdr[5])
        voxel_vol_mm3 = vz * vy * vx
        mask = sub_u16 > 0
        canon_vox = int(mask.sum())
        obs_ml = round((float(canon_vox) * voxel_vol_mm3) / 1000.0, 2)
        zz, yy, xx = np.nonzero(mask)
        if zz.size > 0:
            obs_centroid = [round(float(zz.mean()), 2), round(float(yy.mean()), 2), round(float(xx.mean()), 2)]
            cz, cy, cx = int(np.median(zz)), int(np.median(yy)), int(np.median(xx))
        else:
            obs_centroid = [float(r["centroid_z"]), float(r["centroid_y"]), float(r["centroid_x"])]
            cz, cy, cx = int(round(obs_centroid[0])), int(round(obs_centroid[1])), int(round(obs_centroid[2]))

        pz, py, px = int(r["seed_z"]), int(r["seed_y"]), int(r["seed_x"])
        side = r["pred_hemi"]
        scale = np.array([float(r["scale_z"]), float(r["scale_y"]), float(r["scale_x"])], dtype=np.float32)
        offset = np.array([float(r["offset_z"]), float(r["offset_y"]), float(r["offset_x"])], dtype=np.float32)

        raw_matches = glob.glob(os.path.join(SRC_DIR, "exp_tumor_u16", f"sub-{pid}_*_T1w.bin"))
        with open(raw_matches[0], "rb") as rf:
            h = struct.unpack("<3H3f", rf.read(18))
            raw_arr = np.frombuffer(rf.read(), dtype="<u2").reshape(h[:3]).astype(np.float32)

        SPATIAL_TOL_MM = 45.0
        OVERLAP_TOL = 0.15
        VOL_REL_TOL_PCT = 35.0
        ALIGN_VOL_TOL_PCT = 0.5

        if pid in demo:
            d = demo[pid]
            act_ml = round(float(d["gt"]), 2)
            loc_str = d["loc"]
            path_str = d["patho"]
            reg = target_region_mask(d, lm, left, inner_mask)
            rz, ry, rx = np.nonzero(reg)
            ref_center = [round(float(rz.mean()), 2), round(float(ry.mean()), 2), round(float(rx.mean()), 2)] if rz.size > 0 else None
            hemi_match = bool(side in d["hemi"])
            ov = float((mask & reg).sum()) / max(float(mask.sum()), 1.0)
            lobe_match = bool(reg[pz, py, px]) or (ov >= OVERLAP_TOL)
            loc_match = bool(hemi_match and lobe_match)

            flip_px = (mask.shape[2] - 1) - px
            flip_side = "R" if side == "L" else "L"
            flip_hemi_match = bool(flip_side in d["hemi"])
            flip_ov = float((mask[:, :, ::-1] & reg).sum()) / max(float(mask.sum()), 1.0)
            flip_lobe_match = bool(reg[pz, py, flip_px]) or (flip_ov >= OVERLAP_TOL)
            flip_centroid = [obs_centroid[0], obs_centroid[1], round(float((mask.shape[2] - 1) - obs_centroid[2]), 2)]

            if ref_center:
                delta_mm = (np.array(obs_centroid) - np.array(ref_center)) * np.array([vz, vy, vx])
                dist_to_ref = round(float(np.linalg.norm(delta_mm)), 2)
                flip_delta_mm = (np.array(flip_centroid) - np.array(ref_center)) * np.array([vz, vy, vx])
                flip_dist_to_ref = round(float(np.linalg.norm(flip_delta_mm)), 2)
            else:
                dist_to_ref = None
                flip_dist_to_ref = None
            if act_ml > 0:
                signed_err_ml = round(obs_ml - act_ml, 2)
                abs_err_ml = round(abs(signed_err_ml), 2)
                rel_err_pct = round(100.0 * abs_err_ml / act_ml, 2)
                vol_ratio = min(obs_ml, act_ml) / max(obs_ml, act_ml)
                within_vol_tol = bool(rel_err_pct <= VOL_REL_TOL_PCT)
                if hemi_match and lobe_match:
                    s_dist = 1.0
                elif dist_to_ref is not None:
                    s_dist = float(np.exp(-((dist_to_ref / SPATIAL_TOL_MM) ** 2)))
                else:
                    s_dist = 1.0
                acc = vol_ratio * s_dist
                total_err = 1.0 - acc

                if flip_hemi_match and flip_lobe_match:
                    flip_s_dist = 1.0
                elif flip_dist_to_ref is not None:
                    flip_s_dist = float(np.exp(-((flip_dist_to_ref / SPATIAL_TOL_MM) ** 2)))
                else:
                    flip_s_dist = 1.0
                flip_acc = vol_ratio * flip_s_dist
                flip_total_err = 1.0 - flip_acc
            else:
                signed_err_ml = abs_err_ml = rel_err_pct = vol_ratio = s_dist = acc = total_err = None
                flip_s_dist = flip_acc = flip_total_err = None
                within_vol_tol = None
            status = "EXCLUDED_DUP" if pid in EXCLUDE else "EVALUATED"
        else:
            act_ml = None
            signed_err_ml = abs_err_ml = rel_err_pct = vol_ratio = s_dist = acc = total_err = None
            flip_hemi_match = flip_lobe_match = flip_ov = flip_dist_to_ref = flip_s_dist = flip_acc = flip_total_err = None
            within_vol_tol = None
            loc_str = "Unlisted in patient_demographics.csv"
            path_str = "N/A"
            reg = np.zeros_like(inner_mask, dtype=bool)
            ref_center = None
            hemi_match = None
            lobe_match = None
            loc_match = None
            ov = None
            dist_to_ref = None
            status = "NO_DEMO_GT"

        if acc is not None:
            flip_note = f" | if_flipped={flip_acc:.4f} (flip_dist={flip_dist_to_ref:.1f} mm)" if hemi_match is False else ""
            print(f"{pid} | {acc:.4f}{flip_note} | abs_err={abs_err_ml:.2f} mL ({rel_err_pct:.1f}%) | dist={dist_to_ref:.1f} mm")
        else:
            print(f"{pid} | N/A")

        report_json = {
            "rs_number": pid,
            "observed_volume_ml": obs_ml,
            "actual_volume_ml": act_ml if act_ml is not None else "N/A",
            "accuracy": round(acc, 6) if acc is not None else "N/A",
            "volume_ratio": round(vol_ratio, 6) if vol_ratio is not None else "N/A",
            "spatial_weight": round(s_dist, 6) if s_dist is not None else "N/A",
            "accuracy_definition": "(min(V_obs, V_act) / max(V_obs, V_act)) * S_dist; S_dist=1 if Hemi&Lobe, exp(-(d_mm/45)^2) otherwise (continuous even on Hemi mismatch; flip check also reported)",
            "errors": {
                "combined_error": round(total_err, 6) if total_err is not None else "N/A",
                "absolute_volume_error_ml": abs_err_ml if abs_err_ml is not None else "N/A",
                "signed_volume_error_ml": signed_err_ml if signed_err_ml is not None else "N/A",
                "relative_volume_error_pct": rel_err_pct if rel_err_pct is not None else "N/A",
                "centroid_distance_error_mm": dist_to_ref if dist_to_ref is not None else "N/A",
            },
            "if_flipped_lr": {
                "accuracy": round(flip_acc, 6) if flip_acc is not None else "N/A",
                "combined_error": round(flip_total_err, 6) if flip_total_err is not None else "N/A",
                "spatial_weight": round(flip_s_dist, 6) if flip_s_dist is not None else "N/A",
                "centroid_distance_error_mm": flip_dist_to_ref if flip_dist_to_ref is not None else "N/A",
                "hemisphere_match": flip_hemi_match if flip_hemi_match is not None else "N/A",
                "lobe_region_match": flip_lobe_match if flip_lobe_match is not None else "N/A",
                "region_overlap_fraction": round(flip_ov, 4) if flip_ov is not None else "N/A",
                "within_clinical_volume_tol": within_vol_tol if within_vol_tol is not None else "N/A",
            },
            "tolerances": {
                "spatial_decay_sigma_mm": SPATIAL_TOL_MM,
                "lobe_overlap_min_fraction": OVERLAP_TOL,
                "clinical_volume_rel_tol_pct": VOL_REL_TOL_PCT,
                "within_clinical_volume_tol": within_vol_tol if within_vol_tol is not None else "N/A",
                "alignment_volume_conservation_tol_pct": ALIGN_VOL_TOL_PCT,
            },
            "location": {
                "clinical_location": loc_str,
                "pathology": path_str,
                "detected_hemisphere": "Left" if side == "L" else "Right",
                "hemisphere_match": hemi_match if hemi_match is not None else "N/A",
                "lobe_region_match": lobe_match if lobe_match is not None else "N/A",
                "location_match": loc_match if loc_match is not None else "N/A",
                "region_overlap_fraction": round(ov, 4) if ov is not None else "N/A",
            },
            "observed_tumor_centroid_voxel_zyx": obs_centroid,
            "observed_peak_seed_voxel_zyx": [pz, py, px],
            "actual_tumor_centroid_voxel_zyx": {
                "exact_mask_available": False,
                "note": "Dataset provides scalar lesion volumes and clinical lobe descriptions without voxel-wise tumor masks; reference anatomical target lobe region centroid in [192,224,192] space is reported below.",
                "anatomical_target_lobe_centroid_zyx": ref_center if ref_center is not None else "N/A",
                "euclidean_distance_to_target_region_center_mm": dist_to_ref if dist_to_ref is not None else "N/A",
            },
            "canonical_grid_shape_zyx": [192, 224, 192],
            "original_voxel_spacing_mm": [round(float(h[3]), 6), round(float(h[4]), 6), round(float(h[5]), 6)],
            "alignment_scale_zyx": [round(float(scale[0]), 6), round(float(scale[1]), 6), round(float(scale[2]), 6)],
            "voxel_spacing_mm": [round(vz, 6), round(vy, 6), round(vx, 6)],
            "voxel_volume_mm3": round(voxel_vol_mm3, 6),
            "canonical_voxels": canon_vox,
        }
        with open(os.path.join(REP_DIR, f"{pid}.json"), "w", encoding="utf-8") as jf:
            json.dump(report_json, jf, indent=2)

        if acc is not None and hemi_match is False and flip_acc is not None:
            acc_str = f"{acc:.4f} (Flip: {flip_acc:.4f})"
            acc_short = f"{acc:.4f}*"
        elif acc is not None:
            acc_str = f"{acc:.4f}"
            acc_short = acc_str
        else:
            acc_str = acc_short = "N/A"
        err_str = f"{signed_err_ml:+.2f} mL ({rel_err_pct:.1f}%)" if abs_err_ml is not None else "N/A"
        err_pct_short = f"{rel_err_pct:.0f}%" if rel_err_pct is not None else "N/A"
        vol_ratio_str = f"{vol_ratio:.4f}" if vol_ratio is not None else "N/A"
        s_dist_str = f"{s_dist:.4f}" if s_dist is not None else "N/A"
        act_str = f"{act_ml:.2f} mL" if act_ml is not None else "N/A"
        ref_str = f"({ref_center[0]:.2f}, {ref_center[1]:.2f}, {ref_center[2]:.2f})" if ref_center is not None else "N/A"
        dist_str = f"{dist_to_ref:.2f} mm" if dist_to_ref is not None else "N/A"
        flip_acc_str = f"{flip_acc:.4f}" if flip_acc is not None else "N/A"
        flip_err_str = f"{flip_total_err:.4f}" if flip_total_err is not None else "N/A"
        flip_dist_str = f"{flip_dist_to_ref:.2f} mm" if flip_dist_to_ref is not None else "N/A"
        with open(os.path.join(REP_DIR, f"{pid}.md"), "w", encoding="utf-8") as mf:
            mf.write(f"# {pid} Segmentation Report\n\n")
            mf.write("| Field | Value |\n|---|---|\n")
            mf.write(f"| **RS Number** | {pid} |\n")
            mf.write(f"| **Observed Physical Volume** | {obs_ml:.2f} mL (`{canon_vox}` vox x `{voxel_vol_mm3:.4f}` mm3/vox) |\n")
            mf.write(f"| **Aligned Voxel Spacing (z', y', x')** | ({vz:.4f}, {vy:.4f}, {vx:.4f}) mm |\n")
            mf.write(f"| **Actual Volume** | {act_str} |\n")
            mf.write(f"| **Accuracy (Option 3: Ratio x Spatial Decay)** | {f'{acc:.4f}' if acc is not None else 'N/A'} |\n")
            mf.write(f"| **Volume Ratio min/max** | {vol_ratio_str} |\n")
            mf.write(f"| **Spatial Weight (sigma = {SPATIAL_TOL_MM:.0f} mm)** | {s_dist_str} |\n")
            mf.write(f"| **Combined Error (1 - Acc)** | {f'{total_err:.4f}' if total_err is not None else 'N/A'} |\n")
            mf.write(f"| **Absolute / Signed Volume Error** | {f'{abs_err_ml:.2f} mL ({signed_err_ml:+.2f} mL)' if abs_err_ml is not None else 'N/A'} |\n")
            mf.write(f"| **Relative Volume Error** | {f'{rel_err_pct:.2f}%' if rel_err_pct is not None else 'N/A'} (tol <= {VOL_REL_TOL_PCT:.0f}%: {within_vol_tol if within_vol_tol is not None else 'N/A'}) |\n")
            mf.write(f"| **Spatial Centroid Error** | {dist_str} (decay tol sigma = {SPATIAL_TOL_MM:.1f} mm) |\n")
            mf.write(f"| **If L/R Flipped — Accuracy / Combined Error** | {flip_acc_str} / {flip_err_str} |\n")
            mf.write(f"| **If L/R Flipped — Spatial Error / Match (H/L)** | {flip_dist_str} / {flip_hemi_match}/{flip_lobe_match} (ov={f'{flip_ov:.4f}' if flip_ov is not None else 'N/A'}) |\n")
            mf.write(f"| **Clinical Location** | {loc_str} |\n")
            mf.write(f"| **Pathology** | {path_str} |\n")
            mf.write(f"| **Detected Hemisphere** | {'Left' if side == 'L' else 'Right'} |\n")
            mf.write(f"| **Hemisphere Match** | {hemi_match if hemi_match is not None else 'N/A'} |\n")
            mf.write(f"| **Lobe Region Match (overlap >= {OVERLAP_TOL:.0%})** | {lobe_match if lobe_match is not None else 'N/A'} (ov={f'{ov:.4f}' if ov is not None else 'N/A'}) |\n")
            mf.write(f"| **Location Match** | {loc_match if loc_match is not None else 'N/A'} |\n")
            mf.write(f"| **Observed Tumor Centroid (z, y, x)** | ({obs_centroid[0]:.2f}, {obs_centroid[1]:.2f}, {obs_centroid[2]:.2f}) |\n")
            mf.write(f"| **Observed Peak Seed (z, y, x)** | ({pz}, {py}, {px}) |\n")
            mf.write("| **Actual Tumor Centroid** | Not explicitly available in dataset (no voxel-wise GT mask) |\n")
            mf.write(f"| **Reference Target Lobe Center (z, y, x)** | {ref_str} |\n")

        grid_y, grid_x = np.meshgrid(np.arange(224), np.arange(192), indexing="ij")
        ax_coords = np.array([
            np.full_like(grid_y, cz * scale[0] + offset[0], dtype=np.float32),
            grid_y * scale[1] + offset[1],
            grid_x * scale[2] + offset[2],
        ])
        sl_ax_raw = ndi.map_coordinates(raw_arr, ax_coords, order=1, mode="constant", cval=0.0)
        sl_ax_raw[sl_ax_raw > 42500.0] = 0.0

        grid_z_c, grid_x_c = np.meshgrid(np.arange(192), np.arange(192), indexing="ij")
        cor_coords = np.array([
            grid_z_c * scale[0] + offset[0],
            np.full_like(grid_z_c, cy * scale[1] + offset[1], dtype=np.float32),
            grid_x_c * scale[2] + offset[2],
        ])
        sl_cor_raw = ndi.map_coordinates(raw_arr, cor_coords, order=1, mode="constant", cval=0.0)
        sl_cor_raw[sl_cor_raw > 42500.0] = 0.0

        grid_z_s, grid_y_s = np.meshgrid(np.arange(192), np.arange(224), indexing="ij")
        sag_coords = np.array([
            grid_z_s * scale[0] + offset[0],
            grid_y_s * scale[1] + offset[1],
            np.full_like(grid_z_s, cx * scale[2] + offset[2], dtype=np.float32),
        ])
        sl_sag_raw = ndi.map_coordinates(raw_arr, sag_coords, order=1, mode="constant", cval=0.0)
        sl_sag_raw[sl_sag_raw > 42500.0] = 0.0

        core_mask = compute_inner_core_mask(sub_u16.astype(np.float32), mask)
        mir_mask = mask[:, :, ::-1] & (~mask)

        pos_vals = sl_ax_raw[sl_ax_raw > 500]
        p_lo = float(np.percentile(pos_vals, 2.0)) if pos_vals.size > 0 else 0.0
        p_hi = float(np.percentile(pos_vals, 99.2)) if pos_vals.size > 0 else 38000.0

        fig, axes = plt.subplots(1, 4, figsize=(17.2, 5.0), dpi=135, facecolor="#090d14")

        ax0 = axes[0]
        ax0.set_facecolor("#090d14")
        sl_ax_w = window_t1w(sl_ax_raw, p_lo, p_hi)
        ax0.imshow(sl_ax_w, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", aspect="equal", interpolation="bilinear")
        ax0.axvline(95.5, color="#58a6ff", linestyle=":", linewidth=0.85, alpha=0.55)
        ax0.plot(cx, cy, marker="+", color="#58a6ff", markersize=7, markeredgewidth=1.1, alpha=0.75)
        ax0.set_title("1. Canonical T1w Axial", color="#58a6ff", fontsize=10.0, fontweight="bold", pad=6)
        ax0.axis("off")

        draw_multilayer_slice(
            axes[1], sl_ax_w, mask[cz, :, :], core_mask[cz, :, :], mir_mask[cz, :, :], reg[cz, :, :],
            cx, cy, "2. Axial Overlay", show_midline=True,
        )
        sl_cor_w = window_t1w(sl_cor_raw, p_lo, p_hi)
        draw_multilayer_slice(
            axes[2], sl_cor_w, mask[:, cy, :], core_mask[:, cy, :], mir_mask[:, cy, :], reg[:, cy, :],
            cx, cz, "3. Coronal Overlay", show_midline=True,
        )
        sl_sag_w = window_t1w(sl_sag_raw, p_lo, p_hi)
        draw_multilayer_slice(
            axes[3], sl_sag_w, mask[:, :, cx], core_mask[:, :, cx], None, reg[:, :, cx],
            cy, cz, "4. Sagittal Overlay", show_midline=False,
        )

        fig.suptitle(
            f"{pid}  |  Observed: {obs_ml:.2f} mL  |  Actual: {act_str}  |  Accuracy: {acc_str}  |  Err: {err_str}",
            color="#ffffff", fontsize=11.5, fontweight="bold", y=0.965
        )

        legend_elems = [
            Line2D([0], [0], color="#ffeb3b", lw=2.0, label="Observed Tumor Boundary"),
            Line2D([0], [0], color="#00e676", lw=1.6, label="Inner High-Anomaly Core"),
            Line2D([0], [0], color="#00e5ff", lw=1.5, ls="--", label="Target Anatomical Lobe"),
            Line2D([0], [0], color="#e040fb", lw=1.4, ls=":", label="Contralateral Reference"),
            Line2D([0], [0], color="#ffffff", marker="+", lw=0, markersize=8, label="Observed Centroid"),
        ]
        fig.legend(
            handles=legend_elems, loc="lower center", ncol=5, frameon=True,
            facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3",
            fontsize=8.5, bbox_to_anchor=(0.5, 0.01)
        )

        plt.tight_layout(rect=[0, 0.07, 1, 0.91])
        png_name = f"cuda_segmentation_{pid}.png"
        out_p = os.path.join(VIS_DIR, png_name)
        save_png_atomic(fig, out_p)
        if os.path.abspath(VIS_DIR) != os.path.abspath(PAT_DIR):
            shutil.copy2(out_p, os.path.join(PAT_DIR, png_name))

        axial_thumbs.append((pid, obs_ml, act_str, acc_short, err_pct_short, sl_ax_w, mask[cz, :, :], core_mask[cz, :, :], reg[cz, :, :]))
        eval_records.append({
            "Index": idx,
            "RS_Number": pid,
            "Clinical_Location": loc_str,
            "Pathology": path_str,
            "Voxel_Spacing_ZYX_mm": f"({vz:.4f},{vy:.4f},{vx:.4f})",
            "Voxel_Vol_mm3": round(voxel_vol_mm3, 4),
            "Canonical_Voxels": canon_vox,
            "Observed_Volume_mL": obs_ml,
            "Actual_Volume_mL": act_ml if act_ml is not None else "N/A",
            "Volume_Ratio": round(vol_ratio, 6) if vol_ratio is not None else "N/A",
            "Spatial_Weight": round(s_dist, 6) if s_dist is not None else "N/A",
            "Accuracy": round(acc, 6) if acc is not None else "N/A",
            "Combined_Error": round(total_err, 6) if total_err is not None else "N/A",
            "Flip_Checked_Accuracy": round(flip_acc, 6) if flip_acc is not None else "N/A",
            "Flip_Checked_Combined_Error": round(flip_total_err, 6) if flip_total_err is not None else "N/A",
            "Flip_Centroid_Dist_mm": flip_dist_to_ref if flip_dist_to_ref is not None else "N/A",
            "Flip_Location_Match": bool(flip_hemi_match and flip_lobe_match) if flip_hemi_match is not None else "N/A",
            "Abs_Volume_Error_mL": abs_err_ml if abs_err_ml is not None else "N/A",
            "Signed_Volume_Error_mL": signed_err_ml if signed_err_ml is not None else "N/A",
            "Rel_Volume_Error_pct": rel_err_pct if rel_err_pct is not None else "N/A",
            "Centroid_Dist_mm": dist_to_ref if dist_to_ref is not None else "N/A",
            "Spatial_Tol_mm": SPATIAL_TOL_MM,
            "Overlap_Tol": OVERLAP_TOL,
            "Vol_Rel_Tol_pct": VOL_REL_TOL_PCT,
            "Within_Vol_Tol": within_vol_tol if within_vol_tol is not None else "N/A",
            "Observed_Centroid_ZYX": f"({obs_centroid[0]:.2f},{obs_centroid[1]:.2f},{obs_centroid[2]:.2f})",
            "Actual_Ref_Centroid_ZYX": f"({ref_center[0]:.2f},{ref_center[1]:.2f},{ref_center[2]:.2f})" if ref_center else "N/A",
            "Detected_Hemi": side,
            "Hemi_Match": hemi_match if hemi_match is not None else "N/A",
            "Lobe_Match": lobe_match if lobe_match is not None else "N/A",
            "Location_Match": loc_match if loc_match is not None else "N/A",
            "Region_Overlap": round(ov, 4) if ov is not None else "N/A",
            "Status": status,
        })

    for page_idx, chunk in enumerate([axial_thumbs[:21], axial_thumbs[21:]], 1):
        fig, axes = plt.subplots(3, 7, figsize=(21.5, 10.8), dpi=135, facecolor="#090d14")
        for ax, item in zip(axes.flat, chunk):
            pid, obs_ml, act_str, acc_str, err_pct_short, sl_w, sl_m, sl_core, sl_r = item
            ax.set_facecolor("#090d14")
            ax.imshow(sl_w, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", interpolation="bilinear")
            ax.axvline(95.5, color="#ffffff", linestyle=":", linewidth=0.5, alpha=0.3)
            if sl_r.any():
                ax.contour(sl_r.astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=0.9, linestyles="dashed", origin="lower")
            if sl_m.any():
                ov_rgba = np.zeros((*sl_m.shape, 4), dtype=np.float32)
                ov_rgba[sl_m] = [1.0, 0.22, 0.12, 0.36]
                if sl_core.any():
                    ov_rgba[sl_core] = [1.0, 0.58, 0.05, 0.48]
                ax.imshow(ov_rgba, origin="lower")
                ax.contour(sl_m.astype(float), levels=[0.5], colors=["#ffeb3b"], linewidths=1.25, origin="lower")
                if sl_core.any() and sl_core.sum() < sl_m.sum():
                    ax.contour(sl_core.astype(float), levels=[0.5], colors=["#00e676"], linewidths=0.85, origin="lower")
            ax.set_title(
                f"{pid}\nObs: {obs_ml:.2f} | Act: {act_str}\nAcc: {acc_str} | Err: {err_pct_short}",
                color="#f0f6fc", fontsize=8.0, fontweight="bold"
            )
            ax.axis("off")
        fig.suptitle(
            f"T1w Brain Tumor Segmentation Cohort Overview (v5, Page {page_idx}/2)",
            color="#ffffff", fontsize=14, fontweight="bold", y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        m_name = f"cohort_montage_v5_page{page_idx}.png"
        save_png_atomic(fig, os.path.join(VIS_DIR, m_name))

    eval_csv = os.path.join(V5_DIR, "CUDA_EVALUATION_REPORT_v5.csv")
    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(eval_records[0].keys()))
        w.writeheader()
        w.writerows(eval_records)

    aet_csv = os.path.join(V5_DIR, "ACCURACY_ERRORS_TOLERANCE_v5.csv")
    aet_fields = [
        "RS_Number",
        "Accuracy",
        "If_Flipped_Accuracy",
        "Combined_Error",
        "If_Flipped_Combined_Error",
        "Abs_Volume_Error_mL",
        "Signed_Volume_Error_mL",
        "Rel_Volume_Error_pct",
        "Centroid_Dist_mm",
        "If_Flipped_Centroid_Dist_mm",
        "Spatial_Tol_mm",
        "Overlap_Tol",
        "Vol_Rel_Tol_pct",
        "Within_Vol_Tol",
        "Align_Vol_Conservation_Tol_pct",
    ]
    with open(aet_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=aet_fields)
        w.writeheader()
        for r in eval_records:
            w.writerow({
                "RS_Number": r["RS_Number"],
                "Accuracy": r["Accuracy"],
                "If_Flipped_Accuracy": r["Flip_Checked_Accuracy"],
                "Combined_Error": r["Combined_Error"],
                "If_Flipped_Combined_Error": r["Flip_Checked_Combined_Error"],
                "Abs_Volume_Error_mL": r["Abs_Volume_Error_mL"],
                "Signed_Volume_Error_mL": r["Signed_Volume_Error_mL"],
                "Rel_Volume_Error_pct": r["Rel_Volume_Error_pct"],
                "Centroid_Dist_mm": r["Centroid_Dist_mm"],
                "If_Flipped_Centroid_Dist_mm": r["Flip_Centroid_Dist_mm"],
                "Spatial_Tol_mm": r["Spatial_Tol_mm"],
                "Overlap_Tol": r["Overlap_Tol"],
                "Vol_Rel_Tol_pct": r["Vol_Rel_Tol_pct"],
                "Within_Vol_Tol": r["Within_Vol_Tol"],
                "Align_Vol_Conservation_Tol_pct": 0.5,
            })

    unique_val = [r for r in eval_records if r["Status"] == "EVALUATED"]
    accs = [float(r["Accuracy"]) for r in unique_val]
    abs_errs = [float(r["Abs_Volume_Error_mL"]) for r in unique_val]
    signed_errs = [float(r["Signed_Volume_Error_mL"]) for r in unique_val]
    rel_errs = [float(r["Rel_Volume_Error_pct"]) for r in unique_val]
    dists = [float(r["Centroid_Dist_mm"]) for r in unique_val]
    vol_tol_ok = sum(1 for r in unique_val if r["Within_Vol_Tol"] is True)
    hemi_ok = sum(1 for r in unique_val if r["Hemi_Match"])
    lobe_ok = sum(1 for r in unique_val if r["Lobe_Match"])
    loc_ok = sum(1 for r in unique_val if r["Location_Match"])
    mean_acc = float(np.mean(accs))
    med_acc = float(np.median(accs))
    mae_ml = float(np.mean(abs_errs))
    med_ae_ml = float(np.median(abs_errs))
    mean_signed_ml = float(np.mean(signed_errs))
    mre_pct = float(np.mean(rel_errs))
    med_re_pct = float(np.median(rel_errs))
    mean_dist = float(np.mean(dists))
    med_dist = float(np.median(dists))

    eval_md = os.path.join(V5_DIR, "CUDA_EVALUATION_REPORT_v5.md")
    flip_aware_accs = [float(r["Flip_Checked_Accuracy"]) if r["Hemi_Match"] is False else float(r["Accuracy"]) for r in unique_val]
    flip_aware_mean = float(np.mean(flip_aware_accs))
    with open(eval_md, "w", encoding="utf-8") as f:
        f.write("# v5 Pure CUDA C/C++ Brain Tumor Segmentation Evaluation Report\n\n")
        f.write("**Physical volume reconstruction**: Python canonical pre-alignment (`align_canonical_v5.py`) stamps transformed physical voxel spacing `(vz', vy', vx') = (vz * s_z, vy * s_y, vx * s_x)` into each 18-byte binary header, so `observed_volume_ml = (canonical_voxels * vz' * vy' * vx') / 1000.0` preserves the original physical volume (alignment conservation error `< 0.5%` max across all brains).\n\n")
        f.write("**Accuracy formula (Option 3, continuous & flip-aware)**: `accuracy = (min(V_obs, V_act) / max(V_obs, V_act)) * S_dist`, where `S_dist = 1.0` if `Hemi_Match and Lobe_Match`, and `exp(-(dist_mm / 45.0)^2)` otherwise (never hard-zeroed even on `Hemi_Match == False`). For any scan with opposite laterality (`Hemi_Match == False`), the L/R-mirrored check (`Flip_Acc` / `Flip_Err`) is also reported alongside direct accuracy, errors, and tolerances.\n\n")
        f.write("**Tolerances**: Spatial Gaussian decay `sigma_tol = 45.0 mm`; Lobe region overlap `>= 0.15`; Volumetric agreement tolerance band `<= 35.0%` relative error; Alignment volume conservation `<= 0.50%`.\n\n")
        f.write("| Metric (39 Unique GT Scans) | v5 (Option 3 Accuracy + Errors & Tolerances) |\n")
        f.write("|---|---:|\n")
        f.write(f"| Mean Accuracy (Continuous Option 3) | {mean_acc:.4f} |\n")
        f.write(f"| Mean Accuracy (If L/R Flipped Checked on Mismatch) | {flip_aware_mean:.4f} |\n")
        f.write(f"| Median Accuracy | {med_acc:.4f} |\n")
        f.write(f"| Mean Combined Error (1 - Accuracy) | {1.0 - mean_acc:.4f} |\n")
        f.write(f"| Mean Absolute Volume Error (MAE) | {mae_ml:.2f} mL |\n")
        f.write(f"| Median Absolute Volume Error | {med_ae_ml:.2f} mL |\n")
        f.write(f"| Mean Signed Volume Bias (Obs - Act) | {mean_signed_ml:+.2f} mL |\n")
        f.write(f"| Mean Relative Volume Error (MRE) | {mre_pct:.2f}% |\n")
        f.write(f"| Median Relative Volume Error | {med_re_pct:.2f}% |\n")
        f.write(f"| Mean / Median Centroid Distance Error | {mean_dist:.2f} mm / {med_dist:.2f} mm |\n")
        f.write(f"| Within Volumetric Tolerance (<= 35% rel err) | {vol_tol_ok}/39 ({100.0*vol_tol_ok/39:.1f}%) |\n")
        f.write(f"| Hemisphere Hit Rate | {hemi_ok}/39 ({100.0*hemi_ok/39:.1f}%) |\n")
        f.write(f"| Anatomical Lobe Hit Rate (ov >= 15%) | {lobe_ok}/39 ({100.0*lobe_ok/39:.1f}%) |\n")
        f.write(f"| Combined Location Hit Rate (Hemi & Lobe) | {loc_ok}/39 ({100.0*loc_ok/39:.1f}%) |\n\n")
        f.write("## Per-Patient Results\n\n")
        f.write("| RS | Obs (mL) | Act (mL) | Accuracy | FlipAcc | Vol Err (mL) | Rel Err | Dist (mm) | VolTol<=35% | Loc (H/L) | Location | Status |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---|---|\n")
        for r in eval_records:
            act_cell = f"{r['Actual_Volume_mL']:.2f}" if isinstance(r["Actual_Volume_mL"], float) else "N/A"
            acc_cell = f"{r['Accuracy']:.4f}" if isinstance(r["Accuracy"], float) else "N/A"
            facc_cell = f"{r['Flip_Checked_Accuracy']:.4f}" if isinstance(r["Flip_Checked_Accuracy"], float) else "N/A"
            serr_cell = f"{r['Signed_Volume_Error_mL']:+.2f}" if isinstance(r["Signed_Volume_Error_mL"], float) else "N/A"
            rerr_cell = f"{r['Rel_Volume_Error_pct']:.1f}%" if isinstance(r["Rel_Volume_Error_pct"], float) else "N/A"
            dst_cell = f"{r['Centroid_Dist_mm']:.1f}" if isinstance(r["Centroid_Dist_mm"], float) else "N/A"
            vtol_cell = str(r["Within_Vol_Tol"])
            hl_cell = f"{r['Hemi_Match']}/{r['Lobe_Match']}" if r["Hemi_Match"] != "N/A" else "N/A"
            f.write(
                f"| {r['RS_Number']} | {r['Observed_Volume_mL']:.2f} | {act_cell} | {acc_cell} | {facc_cell} | {serr_cell} | {rerr_cell} | {dst_cell} | {vtol_cell} | {hl_cell} | "
                f"{r['Clinical_Location']} | {r['Status']} |\n"
            )


if __name__ == "__main__":
    main()
