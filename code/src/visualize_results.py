#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_results.py

Renders multi-planar (Axial, Coronal, Sagittal) visualizations
and cohort overview montages from MRI volumes and segmented masks.
"""
import os
import csv
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluate_v3 import load_demo, lobe_masks, target_region_mask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.join(SRC_DIR, "new_code_v3")
OUT_VIS_DIR = os.path.join(SRC_DIR, "attempt_2", "cuda_all_patients")
MAIN_VIS_DIR = os.path.join(ROOT, "output_visualizations")
os.makedirs(OUT_VIS_DIR, exist_ok=True)
os.makedirs(MAIN_VIS_DIR, exist_ok=True)


def render_all():
    demo = load_demo()
    inner_mask = np.load(os.path.join(V3_DIR, "canonical_inner_mask.npy"))
    lm, left = lobe_masks(inner_mask)
    eval_csv = os.path.join(V3_DIR, "CUDA_EVALUATION_REPORT.csv")
    with open(eval_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    axial_thumbs = []

    for r in rows:
        idx = int(r["Test_Index"])
        pid = r["Patient_ID"]
        vol = np.load(os.path.join(V3_DIR, "cuda_segmented_out", f"{pid}_can_vol.npy")).astype(np.float32)
        mask = np.load(os.path.join(V3_DIR, "cuda_segmented_out", f"{pid}_can_mask.npy")) > 0

        if pid in demo:
            reg = target_region_mask(demo[pid], lm, left, inner_mask)
        else:
            reg = np.zeros_like(inner_mask, dtype=bool)

        zz, yy, xx = np.nonzero(mask)
        if zz.size > 0:
            cz, cy, cx = int(np.median(zz)), int(np.median(yy)), int(np.median(xx))
        else:
            cz, cy, cx = 100, 112, 96

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=130, facecolor="#0d1117")
        slices = [
            (vol[cz, :, :], mask[cz, :, :], reg[cz, :, :], f"Axial (z={cz})"),
            (vol[:, cy, :], mask[:, cy, :], reg[:, cy, :], f"Coronal (y={cy})"),
            (vol[:, :, cx], mask[:, :, cx], reg[:, :, cx], f"Sagittal (x={cx})"),
        ]
        vmax = max(float(np.percentile(vol[vol > 0], 99.0)) if (vol > 0).any() else 40000.0, 1.0)

        for ax, (sl_v, sl_m, sl_r, title) in zip(axes, slices):
            ax.set_facecolor("#0d1117")
            ax.imshow(sl_v, cmap="gray", vmin=0, vmax=vmax, origin="lower", aspect="equal")
            if sl_r.any():
                ax.contour(sl_r.astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=1.1, linestyles="dashed", origin="lower")
            if sl_m.any():
                ov_rgba = np.zeros((*sl_m.shape, 4), dtype=np.float32)
                ov_rgba[sl_m] = [1.0, 0.18, 0.18, 0.42]
                ax.imshow(ov_rgba, origin="lower", aspect="equal")
                ax.contour(sl_m.astype(float), levels=[0.5], colors=["#ffeb3b"], linewidths=1.5, origin="lower")
            ax.set_title(title, color="#e6edf3", fontsize=10.5, fontweight="bold")
            ax.axis("off")

        loc_str = r["Clinical_Location"]
        gt_ml = float(r["GT_Volume_mL"])
        pr_ml = float(r["Pred_Volume_mL"])
        v_acc = float(r["Volumetric_Accuracy_Pct"])
        c_acc = float(r["Combined_Loc_Vol_Acc_Pct"])
        h_ok = "PASS" if r["Hemi_Hit"] == "True" else "MISS"
        l_ok = "PASS" if r["Lobe_Hit"] == "True" else "MISS"
        gpu_ms = float(r["GPU_Time_ms"])

        fig.suptitle(
            f"{pid} ({loc_str}) | GT: {gt_ml:.1f} mL vs Pred: {pr_ml:.2f} mL | Hemi: {h_ok} | Lobe: {l_ok} | VolAcc: {v_acc:.1f}% | CombAcc: {c_acc:.1f}% | RTX 4070: {gpu_ms:.0f} ms\n"
            f"[Yellow/Red = T1w-Only CUDA Segmented Tumor | Cyan Dashed = Clinical Anatomical Lobe Region]",
            color="#ffffff", fontsize=10.5, fontweight="bold", y=0.97
        )
        plt.tight_layout(rect=[0, 0, 1, 0.89])
        png_name = f"cuda_segmentation_{pid}.png"
        out_p = os.path.join(OUT_VIS_DIR, png_name)
        fig.savefig(out_p, facecolor=fig.get_facecolor())
        plt.close(fig)

        shutil.copy2(out_p, os.path.join(MAIN_VIS_DIR, png_name))
        shutil.copy2(out_p, os.path.join(ART_IMG_DIR, png_name))
        axial_thumbs.append((pid, loc_str, gt_ml, pr_ml, v_acc, h_ok, l_ok, vol[cz, :, :], mask[cz, :, :], reg[cz, :, :], vmax))

    # Render two 21-panel cohort montages (Page 1: scans 1..21, Page 2: scans 22..42)
    for page_idx, chunk in enumerate([axial_thumbs[:21], axial_thumbs[21:]], 1):
        fig, axes = plt.subplots(3, 7, figsize=(21, 10.5), dpi=130, facecolor="#0d1117")
        for ax, item in zip(axes.flat, chunk):
            pid, loc_str, gt_ml, pr_ml, v_acc, h_ok, l_ok, sl_v, sl_m, sl_r, vmax = item
            ax.set_facecolor("#0d1117")
            ax.imshow(sl_v, cmap="gray", vmin=0, vmax=vmax, origin="lower")
            if sl_r.any():
                ax.contour(sl_r.astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=0.9, linestyles="dashed", origin="lower")
            if sl_m.any():
                ov_rgba = np.zeros((*sl_m.shape, 4), dtype=np.float32)
                ov_rgba[sl_m] = [1.0, 0.18, 0.18, 0.45]
                ax.imshow(ov_rgba, origin="lower")
                ax.contour(sl_m.astype(float), levels=[0.5], colors=["#ffeb3b"], linewidths=1.2, origin="lower")
            badge_col = "#3fb950" if l_ok == "PASS" else ("#d29922" if h_ok == "PASS" else "#f85149")
            ax.set_title(f"{pid}: {pr_ml:.1f}/{gt_ml:.1f}mL ({v_acc:.0f}%)\n[{l_ok}] {loc_str[:20]}", color=badge_col, fontsize=8.5, fontweight="bold")
            ax.axis("off")
        fig.suptitle(f"T1w-Only CUDA Brain Tumor Segmentation Cohort Montage (Page {page_idx}/2) — RTX 4070 GPU", color="#ffffff", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        for m_name in (f"cohort_montage_v3_page{page_idx}.png", f"cohort_montage_v2_page{page_idx}.png"):
            p_out = os.path.join(V3_DIR, "visualizations", m_name)
            fig.savefig(p_out, facecolor=fig.get_facecolor())
            shutil.copy2(p_out, os.path.join(MAIN_VIS_DIR, m_name))
            shutil.copy2(p_out, os.path.join(ART_IMG_DIR, m_name))
        plt.close(fig)

    print(f"Rendered 42 patient overlays + 2 cohort montages to {OUT_VIS_DIR}, {MAIN_VIS_DIR}, and {ART_IMG_DIR}")


if __name__ == "__main__":
    render_all()
