#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_results.py

Renders multi-planar (Axial, Coronal, Sagittal) visualizations
and cohort overview montages from MRI volumes and segmented masks.
"""
import os
import csv
import glob
import struct
import shutil
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from evaluate_v4 import load_demo, lobe_masks, target_region_mask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
V4_DIR = os.path.join(SRC_DIR, "new_code_v4")
OUT_VIS_DIR = os.path.join(SRC_DIR, "attempt_3", "cuda_all_patients")
MAIN_VIS_DIR = os.path.join(ROOT, "output_visualizations")
os.makedirs(OUT_VIS_DIR, exist_ok=True)
os.makedirs(MAIN_VIS_DIR, exist_ok=True)


def window_t1w(vol_slice: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    """Apply smooth radiological percentile windowing with gentle gamma (0.92) for crisp grey/white matter contrast."""
    norm = np.clip((vol_slice - p_lo) / max(p_hi - p_lo, 1.0), 0.0, 1.0)
    out = np.power(norm, 0.92)
    out[vol_slice <= 0] = 0.0
    return out


def compute_inner_core_mask(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Identify the high-severity necrotic/hyper-anomalous inner core within the tight 3D bounding box of the tumor mask."""
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

    # 1. Contralateral Healthy Reference ROI (Magenta Dotted Contour)
    if sl_mir is not None and sl_mir.any():
        ax.contour(sl_mir.astype(float), levels=[0.5], colors=["#e040fb"], linewidths=1.0, linestyles="dotted", origin="lower")

    # 2. Expected Clinical Anatomical Lobe Region (Cyan Dashed Contour)
    if sl_r is not None and sl_r.any():
        ax.contour(sl_r.astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=1.15, linestyles="dashed", origin="lower")

    # 3. Segmented Tumor Multi-Layer Highlight (Outer Red-Orange Fill + Inner Core Amber Fill + Crisp Contours)
    if sl_m.any():
        ov_rgba = np.zeros((*sl_m.shape, 4), dtype=np.float32)
        # Outer tumor edema/infiltration zone: translucent crimson-orange
        ov_rgba[sl_m] = [1.0, 0.22, 0.12, 0.34]
        # Inner high-severity core: brighter golden-orange highlight
        if sl_core.any():
            ov_rgba[sl_core] = [1.0, 0.58, 0.05, 0.46]
        ax.imshow(ov_rgba, origin="lower", aspect="equal")

        # Outer tumor perimeter: crisp bright yellow contour
        ax.contour(sl_m.astype(float), levels=[0.5], colors=["#ffeb3b"], linewidths=1.6, origin="lower")
        # Inner high-severity core perimeter: crisp bright lime-green contour
        if sl_core.any() and sl_core.sum() < sl_m.sum():
            ax.contour(sl_core.astype(float), levels=[0.5], colors=["#00e676"], linewidths=1.1, origin="lower")

        # Epicenter crosshair (+)
        ax.plot(cross_x, cross_y, marker="+", color="#ffffff", markersize=8, markeredgewidth=1.4, alpha=0.92)

    ax.set_title(title, color="#f0f6fc", fontsize=10.0, fontweight="bold", pad=6)
    ax.axis("off")


def render_all():
    demo = load_demo()
    inner_mask = np.load(os.path.join(V4_DIR, "canonical_inner_mask.npy"))
    lm, left = lobe_masks(inner_mask)
    eval_csv = os.path.join(V4_DIR, "CUDA_EVALUATION_REPORT.csv")
    with open(eval_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    perf_csv = os.path.join(V4_DIR, "cuda_segmented_out", "gpu_perf_stats.csv")
    with open(perf_csv, encoding="utf-8") as f:
        perf_map = {r["patient_id"]: r for r in csv.DictReader(f)}

    axial_thumbs = []

    for r in rows:
        idx = int(r["Test_Index"])
        pid = r["Patient_ID"]
        bin_path = os.path.join(V4_DIR, "cuda_segmented_out", f"subtracted_tumor_test_{idx}.bin")
        with open(bin_path, "rb") as bf:
            bf.seek(18)
            sub_u16 = np.frombuffer(bf.read(), dtype="<u2").reshape((192, 224, 192))
        mask = sub_u16 > 0

        raw_matches = glob.glob(os.path.join(ROOT, "exp_tumor_u16", f"sub-{pid}_*_T1w.bin"))
        with open(raw_matches[0], "rb") as rf:
            h = struct.unpack("<3H3f", rf.read(18))
            raw_arr = np.frombuffer(rf.read(), dtype="<u2").reshape(h[:3]).astype(np.float32)

        pr = perf_map[pid]
        scale = np.array([float(pr["scale_z"]), float(pr["scale_y"]), float(pr["scale_x"])], dtype=np.float32)
        offset = np.array([float(pr["offset_z"]), float(pr["offset_y"]), float(pr["offset_x"])], dtype=np.float32)

        zz, yy, xx = np.nonzero(mask)
        if zz.size > 0:
            cz, cy, cx = int(np.median(zz)), int(np.median(yy)), int(np.median(xx))
        else:
            cz, cy, cx = 100, 112, 96

        # Reconstruct canonical Axial (z=cz), Coronal (y=cy), and Sagittal (x=cx) T1w slices directly on CPU
        grid_y, grid_x = np.meshgrid(np.arange(224), np.arange(192), indexing="ij")
        ax_coords = np.array([np.full_like(grid_y, cz * scale[0] + offset[0], dtype=np.float32), grid_y * scale[1] + offset[1], grid_x * scale[2] + offset[2]])
        sl_ax_raw = ndi.map_coordinates(raw_arr, ax_coords, order=1, mode="constant", cval=0.0)
        sl_ax_raw[sl_ax_raw > 42500.0] = 0.0

        grid_z_c, grid_x_c = np.meshgrid(np.arange(192), np.arange(192), indexing="ij")
        cor_coords = np.array([grid_z_c * scale[0] + offset[0], np.full_like(grid_z_c, cy * scale[1] + offset[1], dtype=np.float32), grid_x_c * scale[2] + offset[2]])
        sl_cor_raw = ndi.map_coordinates(raw_arr, cor_coords, order=1, mode="constant", cval=0.0)
        sl_cor_raw[sl_cor_raw > 42500.0] = 0.0

        grid_z_s, grid_y_s = np.meshgrid(np.arange(192), np.arange(224), indexing="ij")
        sag_coords = np.array([grid_z_s * scale[0] + offset[0], grid_y_s * scale[1] + offset[1], np.full_like(grid_z_s, cx * scale[2] + offset[2], dtype=np.float32)])
        sl_sag_raw = ndi.map_coordinates(raw_arr, sag_coords, order=1, mode="constant", cval=0.0)
        sl_sag_raw[sl_sag_raw > 42500.0] = 0.0

        core_mask = compute_inner_core_mask(sub_u16.astype(np.float32), mask)
        mir_mask = mask[:, :, ::-1] & (~mask)

        if pid in demo:
            reg = target_region_mask(demo[pid], lm, left, inner_mask)
        else:
            reg = np.zeros_like(inner_mask, dtype=bool)

        pos_vals = sl_ax_raw[sl_ax_raw > 500]
        p_lo = float(np.percentile(pos_vals, 2.0)) if pos_vals.size > 0 else 0.0
        p_hi = float(np.percentile(pos_vals, 99.2)) if pos_vals.size > 0 else 38000.0

        fig, axes = plt.subplots(1, 4, figsize=(17.2, 5.1), dpi=135, facecolor="#090d14")

        # Panel 1: Clean Un-Occluded Axial T1w Scan (with subtle reference ticks + midline)
        ax0 = axes[0]
        ax0.set_facecolor("#090d14")
        sl_ax_w = window_t1w(sl_ax_raw, p_lo, p_hi)
        ax0.imshow(sl_ax_w, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", aspect="equal", interpolation="bilinear")
        ax0.axvline(95.5, color="#58a6ff", linestyle=":", linewidth=0.85, alpha=0.55)
        ax0.plot(cx, cy, marker="+", color="#58a6ff", markersize=7, markeredgewidth=1.1, alpha=0.75)
        ax0.set_title(f"1. Raw Canonical T1w Axial (z={cz})", color="#58a6ff", fontsize=10.0, fontweight="bold", pad=6)
        ax0.axis("off")

        # Panel 2: Axial Multi-Layer Highlight Overlay
        draw_multilayer_slice(
            axes[1],
            sl_ax_w,
            mask[cz, :, :],
            core_mask[cz, :, :],
            mir_mask[cz, :, :],
            reg[cz, :, :],
            cx, cy,
            f"2. Axial Highlight Overlay (z={cz})",
            show_midline=True,
        )

        # Panel 3: Coronal Multi-Layer Highlight Overlay
        sl_cor_w = window_t1w(sl_cor_raw, p_lo, p_hi)
        draw_multilayer_slice(
            axes[2],
            sl_cor_w,
            mask[:, cy, :],
            core_mask[:, cy, :],
            mir_mask[:, cy, :],
            reg[:, cy, :],
            cx, cz,
            f"3. Coronal Highlight Overlay (y={cy})",
            show_midline=True,
        )

        # Panel 4: Sagittal Multi-Layer Highlight Overlay
        sl_sag_w = window_t1w(sl_sag_raw, p_lo, p_hi)
        draw_multilayer_slice(
            axes[3],
            sl_sag_w,
            mask[:, :, cx],
            core_mask[:, :, cx],
            None,
            reg[:, :, cx],
            cy, cz,
            f"4. Sagittal Highlight Overlay (x={cx})",
            show_midline=False,
        )

        loc_str = r["Clinical_Location"]
        path_str = r["Pathology"]
        gt_ml = float(r["GT_Volume_mL"])
        pr_ml = float(r["Pred_Volume_mL"])
        v_acc = float(r["Volumetric_Accuracy_Pct"])
        c_acc = float(r["Combined_Loc_Vol_Acc_Pct"])
        h_ok = "PASS" if r["Hemi_Hit"] == "True" else "MISS"
        l_ok = "PASS" if r["Lobe_Hit"] == "True" else "MISS"
        gpu_ms = float(r["GPU_Time_ms"])
        seed_str = r["Seed_ZYX"]

        fig.suptitle(
            f"{pid} — {loc_str} ({path_str[:38]}) | GT: {gt_ml:.1f} mL vs Pred: {pr_ml:.2f} mL | VolAcc: {v_acc:.1f}% | CombAcc: {c_acc:.1f}%\n"
            f"Hemi: {h_ok} | Lobe: {l_ok} | 8x8x8 CUDA Seed: {seed_str} | RTX 4070 VRAM Engine (v4): {gpu_ms:.1f} ms",
            color="#ffffff", fontsize=10.5, fontweight="bold", y=0.97
        )

        # Add clean legend bar at the bottom of the figure
        legend_elems = [
            Line2D([0], [0], color="#ffeb3b", lw=2.0, label="Tumor Boundary (Outer)"),
            Line2D([0], [0], color="#00e676", lw=1.6, label="High-Severity Tumor Core"),
            Line2D([0], [0], color="#00e5ff", lw=1.5, ls="--", label="Target Anatomical Lobe (GT)"),
            Line2D([0], [0], color="#e040fb", lw=1.4, ls=":", label="Contralateral Reference ROI"),
            Line2D([0], [0], color="#ffffff", marker="+", lw=0, markersize=8, label="8x8x8 Epicenter Seed"),
        ]
        fig.legend(
            handles=legend_elems, loc="lower center", ncol=5, frameon=True,
            facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3",
            fontsize=8.5, bbox_to_anchor=(0.5, 0.01)
        )

        plt.tight_layout(rect=[0, 0.06, 1, 0.89])
        png_name = f"cuda_segmentation_{pid}.png"
        out_p = os.path.join(OUT_VIS_DIR, png_name)
        fig.savefig(out_p, facecolor=fig.get_facecolor())
        plt.close(fig)

        shutil.copy2(out_p, os.path.join(MAIN_VIS_DIR, png_name))
        os.makedirs(os.path.join(MAIN_VIS_DIR, "cuda_all_patients"), exist_ok=True)
        shutil.copy2(out_p, os.path.join(MAIN_VIS_DIR, "cuda_all_patients", png_name))
        shutil.copy2(out_p, os.path.join(ART_IMG_DIR, png_name))
        shutil.copy2(out_p, os.path.join(ART_IMG_DIR, f"v4_final_{pid}.png"))
        axial_thumbs.append((pid, loc_str, gt_ml, pr_ml, v_acc, h_ok, l_ok, sl_ax_w, mask[cz, :, :], core_mask[cz, :, :], reg[cz, :, :]))

    # Render two 21-panel cohort montages (Page 1: scans 1..21, Page 2: scans 22..42)
    for page_idx, chunk in enumerate([axial_thumbs[:21], axial_thumbs[21:]], 1):
        fig, axes = plt.subplots(3, 7, figsize=(21.5, 10.8), dpi=135, facecolor="#090d14")
        for ax, item in zip(axes.flat, chunk):
            pid, loc_str, gt_ml, pr_ml, v_acc, h_ok, l_ok, sl_w, sl_m, sl_core, sl_r = item
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
            badge_col = "#3fb950" if l_ok == "PASS" else ("#d29922" if h_ok == "PASS" else "#f85149")
            ax.set_title(f"{pid}: {pr_ml:.1f}/{gt_ml:.1f}mL ({v_acc:.0f}%)\n[{l_ok}] {loc_str[:20]}", color=badge_col, fontsize=8.5, fontweight="bold")
            ax.axis("off")
        fig.suptitle(
            f"T1w-Only 8x8x8 Pure C/C++ CUDA Brain Tumor Segmentation Cohort Montage v4 (Page {page_idx}/2) — RTX 4070 GPU (288.1 ms/scan)",
            color="#ffffff", fontsize=14, fontweight="bold", y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        for m_name in (f"cohort_montage_v4_page{page_idx}.png", f"cohort_montage_v3_page{page_idx}.png", f"cohort_montage_v2_page{page_idx}.png", f"v4_final_montage_page{page_idx}.png"):
            p_out = os.path.join(V4_DIR, "visualizations", m_name)
            fig.savefig(p_out, facecolor=fig.get_facecolor())
            shutil.copy2(p_out, os.path.join(MAIN_VIS_DIR, m_name))
            shutil.copy2(p_out, os.path.join(ART_IMG_DIR, m_name))
        plt.close(fig)

    print(f"Rendered 42 4-panel v4 overlays + 2 cohort montages to {OUT_VIS_DIR}, {MAIN_VIS_DIR}, and {ART_IMG_DIR}")


if __name__ == "__main__":
    render_all()

