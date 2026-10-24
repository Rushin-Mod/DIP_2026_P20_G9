#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_results.py
====================
Deliverable 4: Verification & Multi-Planar Volumetric UI

Features:
- Prompts for or accepts via CLI:
    1. Preprocessed original test brain .bin file.
    2. Subtracted detected tumor .bin file (`subtracted_tumor_test_X.bin`).
- Reads & validates 18-byte packed binary header:
    struct VolumeHeader { uint16_t dim_z, dim_y, dim_x; float vz, vy, vx; };
- Supports both uint16 and float32 payloads dynamically based on file size.
- Strictly classical 3D 26-connected component analysis (scipy.ndimage) to filter out
  isolated single-voxel noise (clusters < 100 mm³).
- Computes and displays quantitative clinical metrics in terminal:
    * Total tumor voxel count
    * Total tumor physical volume in mm³ and cm³ (voxel_count * vz * vy * vx)
    * Spatial centroid / Center of Mass (CoM) coordinates in physical (Z, Y, X) mm
- Builds an interactive multi-planar (Axial, Coronal, Sagittal) slicer with interactive sliders:
    * Grayscale anatomical MRI with high-contrast red tumor overlay.
    * Accurately applies physical voxel spacing (eliminating spatial stretching).
- Optional 3D surface mesh rendering of brain and segmented tumor mass via Vedo.
"""

import os
import sys
import struct
import argparse
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap

try:
    import vedo
except ImportError:
    vedo = None

HEADER_FMT = "<3H3f"
HEADER_SIZE = 18

def read_bin_volume(bin_path):
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"File not found: {bin_path}")
    
    file_size = os.path.getsize(bin_path)
    if file_size < HEADER_SIZE:
        raise ValueError(f"File too small to contain 18-byte header: {bin_path}")

    with open(bin_path, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        dim_z, dim_y, dim_x, vz, vy, vx = struct.unpack(HEADER_FMT, header_bytes)
        payload_bytes = f.read()

    total_voxels = int(dim_z) * int(dim_y) * int(dim_x)
    actual_payload_len = len(payload_bytes)

    if actual_payload_len == total_voxels * 2:
        dtype_name = "uint16"
        data = np.frombuffer(payload_bytes, dtype="<u2").reshape((dim_z, dim_y, dim_x))
    elif actual_payload_len == total_voxels * 4:
        dtype_name = "float32"
        data = np.frombuffer(payload_bytes, dtype="<f4").reshape((dim_z, dim_y, dim_x))
    else:
        raise ValueError(
            f"Payload size mismatch for {bin_path}: expected {total_voxels*2} (uint16) "
            f"or {total_voxels*4} (float32), got {actual_payload_len}"
        )

    header = {
        "dim_z": dim_z, "dim_y": dim_y, "dim_x": dim_x,
        "vz": vz, "vy": vy, "vx": vx,
        "dtype": dtype_name
    }
    return data, header

def analyze_tumor(orig_data, subt_data, header, min_cluster_mm3=100.0):
    vz, vy, vx = header["vz"], header["vy"], header["vx"]
    voxel_vol_mm3 = float(vz * vy * vx)

    # Binary mask of non-zero detections
    raw_mask = subt_data > 0
    if not np.any(raw_mask):
        return np.zeros_like(raw_mask, dtype=bool), {
            "voxel_count": 0,
            "vol_mm3": 0.0,
            "vol_cm3": 0.0,
            "centroid_mm": (0.0, 0.0, 0.0),
            "centroid_idx": (0, 0, 0)
        }

    # 3D 26-connected component labeling (all face, edge, and corner diagonals)
    st26 = ndi.generate_binary_structure(3, 3)
    labeled_array, num_features = ndi.label(raw_mask, structure=st26)

    min_voxels = max(1, int(round(min_cluster_mm3 / voxel_vol_mm3)))

    # Filter out clusters smaller than min_cluster_mm3
    component_sizes = ndi.sum(raw_mask, labeled_array, index=np.arange(1, num_features + 1))
    valid_mask = component_sizes >= min_voxels
    valid_indices = np.where(valid_mask)[0] + 1

    clean_tumor_mask = np.isin(labeled_array, valid_indices)
    tumor_voxel_count = int(np.sum(clean_tumor_mask))

    if tumor_voxel_count == 0:
        return np.zeros_like(raw_mask, dtype=bool), {
            "voxel_count": 0,
            "vol_mm3": 0.0,
            "vol_cm3": 0.0,
            "centroid_mm": (0.0, 0.0, 0.0),
            "centroid_idx": (0, 0, 0)
        }

    # Total physical volumes
    total_vol_mm3 = tumor_voxel_count * voxel_vol_mm3
    total_vol_cm3 = total_vol_mm3 / 1000.0

    # Center of Mass (CoM) / Spatial Centroid
    com_idx = ndi.center_of_mass(clean_tumor_mask)
    centroid_mm = (
        float(com_idx[0] * vz),
        float(com_idx[1] * vy),
        float(com_idx[2] * vx)
    )

    metrics = {
        "voxel_count": tumor_voxel_count,
        "vol_mm3": total_vol_mm3,
        "vol_cm3": total_vol_cm3,
        "centroid_mm": centroid_mm,
        "centroid_idx": (float(com_idx[0]), float(com_idx[1]), float(com_idx[2]))
    }
    return clean_tumor_mask, metrics

def render_vedo_3d(orig_data, tumor_mask, header):
    """Optional 3D surface mesh rendering using Vedo (VTK backend)."""
    if vedo is None:
        print("[WARN] vedo not installed; skipping 3D mesh rendering.")
        return

    print("[INFO] Generating 3D anatomical & tumor isosurfaces in Vedo...")
    # Normalize original image to float [0, 1]
    norm = orig_data.astype(np.float32)
    mx = float(np.max(norm))
    if mx > 0:
        norm /= mx

    # 3D Volumes with correct physical spacing
    vol_brain = vedo.Volume(norm, spacing=(header['vx'], header['vy'], header['vz']))
    # Extract brain surface
    brain_mesh = vol_brain.isosurface(0.15).color('gray').alpha(0.18)

    actors = [brain_mesh]

    if tumor_mask is not None and np.any(tumor_mask):
        vol_tumor = vedo.Volume(tumor_mask.astype(np.uint8), spacing=(header['vx'], header['vy'], header['vz']))
        tumor_mesh = vol_tumor.isosurface(0.5).color('red').alpha(0.85)
        actors.append(tumor_mesh)

    plt_vedo = vedo.Plotter(title="3D Brain & Tumor Isosurface (Vedo)", axes=1, bg='black')
    plt_vedo.show(actors, viewup='z')

def visualize_interactive_slicer(orig_data, tumor_mask, header, metrics, no_gui=False, show_3d=False):
    print("\n=======================================================")
    print("      QUANTITATIVE TUMOR SEGMENTATION METRICS         ")
    print("=======================================================")
    print(f" Data Precision Mode     : {header['dtype']}")
    print(f" Matrix Geometry (Z,Y,X) : {header['dim_z']} x {header['dim_y']} x {header['dim_x']}")
    print(f" Voxel Spacing (mm)      : vz={header['vz']:.3f}, vy={header['vy']:.3f}, vx={header['vx']:.3f}")
    print(f" Total Tumor Voxel Count : {metrics['voxel_count']:,} voxels")
    print(f" Total Tumor Volume      : {metrics['vol_mm3']:.2f} mm³ ({metrics['vol_cm3']:.4f} cm³)")
    print(f" Centroid CoM (Index)    : Z={metrics['centroid_idx'][0]:.1f}, Y={metrics['centroid_idx'][1]:.1f}, X={metrics['centroid_idx'][2]:.1f}")
    print(f" Centroid CoM (Physical) : Z={metrics['centroid_mm'][0]:.2f} mm, Y={metrics['centroid_mm'][1]:.2f} mm, X={metrics['centroid_mm'][2]:.2f} mm")
    print("=======================================================\n")

    if no_gui:
        print("[INFO] Headless mode (--no-gui) enabled; skipping UI.")
        return

    # Normalize image for display
    norm_img = orig_data.astype(np.float32)
    max_val = np.percentile(norm_img[norm_img > 0], 99.8) if np.any(norm_img > 0) else 1.0
    norm_img = np.clip(norm_img / max_val, 0.0, 1.0)

    # Initial slice indices at tumor centroid or matrix center
    dim_z, dim_y, dim_x = header["dim_z"], header["dim_y"], header["dim_x"]
    vz, vy, vx = header["vz"], header["vy"], header["vx"]

    if metrics["voxel_count"] > 0:
        init_z = int(np.clip(round(metrics["centroid_idx"][0]), 0, dim_z - 1))
        init_y = int(np.clip(round(metrics["centroid_idx"][1]), 0, dim_y - 1))
        init_x = int(np.clip(round(metrics["centroid_idx"][2]), 0, dim_x - 1))
    else:
        init_z, init_y, init_x = dim_z // 2, dim_y // 2, dim_x // 2

    # High-contrast red colormap for tumor overlay (transparent at 0, pure red at 1)
    cmap_tumor = ListedColormap([(0, 0, 0, 0), (1.0, 0.1, 0.1, 0.75)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="#1e1e1e")
    plt.subplots_adjust(bottom=0.25, wspace=0.3)

    ax_axial, ax_coronal, ax_sagittal = axes

    # 1. Axial view (X-Y plane at Z slice)
    # aspect ratio = vy / vx
    im_ax = ax_axial.imshow(norm_img[init_z, :, :], cmap="gray", origin="lower", aspect=vy / vx)
    ov_ax = ax_axial.imshow(tumor_mask[init_z, :, :], cmap=cmap_tumor, vmin=0, vmax=1, origin="lower", aspect=vy / vx)
    ax_axial.set_title(f"Axial (Z = {init_z})", color="white", fontsize=12)
    ax_axial.set_xlabel("X (R -> L)", color="white")
    ax_axial.set_ylabel("Y (P -> A)", color="white")
    ax_axial.tick_params(colors="gray")

    # 2. Coronal view (X-Z plane at Y slice)
    # aspect ratio = vz / vx
    im_cor = ax_coronal.imshow(norm_img[:, init_y, :], cmap="gray", origin="lower", aspect=vz / vx)
    ov_cor = ax_coronal.imshow(tumor_mask[:, init_y, :], cmap=cmap_tumor, vmin=0, vmax=1, origin="lower", aspect=vz / vx)
    ax_coronal.set_title(f"Coronal (Y = {init_y})", color="white", fontsize=12)
    ax_coronal.set_xlabel("X (R -> L)", color="white")
    ax_coronal.set_ylabel("Z (I -> S)", color="white")
    ax_coronal.tick_params(colors="gray")

    # 3. Sagittal view (Y-Z plane at X slice)
    # aspect ratio = vz / vy
    im_sag = ax_sagittal.imshow(norm_img[:, :, init_x], cmap="gray", origin="lower", aspect=vz / vy)
    ov_sag = ax_sagittal.imshow(tumor_mask[:, :, init_x], cmap=cmap_tumor, vmin=0, vmax=1, origin="lower", aspect=vz / vy)
    ax_sagittal.set_title(f"Sagittal (X = {init_x})", color="white", fontsize=12)
    ax_sagittal.set_xlabel("Y (P -> A)", color="white")
    ax_sagittal.set_ylabel("Z (I -> S)", color="white")
    ax_sagittal.tick_params(colors="gray")

    fig.suptitle(
        f"3D Brain MRI Tumor Segmentation | Volume: {metrics['vol_cm3']:.2f} cm³ | Centroid: ({metrics['centroid_mm'][0]:.1f}, {metrics['centroid_mm'][1]:.1f}, {metrics['centroid_mm'][2]:.1f}) mm",
        color="white", fontsize=14, y=0.98
    )

    # UI Slider Controls
    slider_ax_z = plt.axes([0.15, 0.12, 0.7, 0.025], facecolor="#333333")
    slider_ax_y = plt.axes([0.15, 0.08, 0.7, 0.025], facecolor="#333333")
    slider_ax_x = plt.axes([0.15, 0.04, 0.7, 0.025], facecolor="#333333")

    s_z = Slider(slider_ax_z, "Axial (Z)", 0, dim_z - 1, valinit=init_z, valstep=1, color="#e63946")
    s_y = Slider(slider_ax_y, "Coronal (Y)", 0, dim_y - 1, valinit=init_y, valstep=1, color="#457b9d")
    s_x = Slider(slider_ax_x, "Sagittal (X)", 0, dim_x - 1, valinit=init_x, valstep=1, color="#2a9d8f")

    for s in [s_z, s_y, s_x]:
        s.label.set_color("white")
        s.valtext.set_color("white")

    def update(val):
        cz = int(s_z.val)
        cy = int(s_y.val)
        cx = int(s_x.val)

        im_ax.set_data(norm_img[cz, :, :])
        ov_ax.set_data(tumor_mask[cz, :, :])
        ax_axial.set_title(f"Axial (Z = {cz})", color="white")

        im_cor.set_data(norm_img[:, cy, :])
        ov_cor.set_data(tumor_mask[:, cy, :])
        ax_coronal.set_title(f"Coronal (Y = {cy})", color="white")

        im_sag.set_data(norm_img[:, :, cx])
        ov_sag.set_data(tumor_mask[:, :, cx])
        ax_sagittal.set_title(f"Sagittal (X = {cx})", color="white")

        fig.canvas.draw_idle()

    s_z.on_changed(update)
    s_y.on_changed(update)
    s_x.on_changed(update)

    if show_3d and vedo is not None:
        render_vedo_3d(orig_data, tumor_mask, header)

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="3D Brain MRI Tumor Segmentation Visualizer (Classical DIP)")
    parser.add_argument("--orig", help="Path to original preprocessed .bin file")
    parser.add_argument("--subt", help="Path to subtracted detected tumor .bin file")
    parser.add_argument("--min-cluster", type=float, default=150.0, help="Minimum cluster size in mm3 (default: 150.0)")
    parser.add_argument("--no-gui", action="store_true", help="Run without opening GUI window")
    parser.add_argument("--3d", dest="show_3d", action="store_true", help="Also render 3D isosurface mesh in Vedo")
    args = parser.parse_args()

    orig_path = args.orig or input("Enter path to original preprocessed .bin file: ").strip().strip('"')
    subt_path = args.subt or input("Enter path to subtracted tumor .bin file: ").strip().strip('"')

    print(f"\n[INFO] Loading original volume: {orig_path}")
    orig_data, orig_hdr = read_bin_volume(orig_path)
    print(f"  Shape: {orig_data.shape}, Voxel type: {orig_hdr['dtype']}")

    print(f"[INFO] Loading subtracted tumor volume: {subt_path}")
    subt_data, subt_hdr = read_bin_volume(subt_path)
    print(f"  Shape: {subt_data.shape}, Voxel type: {subt_hdr['dtype']}")

    # Verification of headers
    if (orig_hdr['dim_z'] != subt_hdr['dim_z'] or
        orig_hdr['dim_y'] != subt_hdr['dim_y'] or
        orig_hdr['dim_x'] != subt_hdr['dim_x']):
        print("[ERROR] Header dimensions do not match between original and subtracted volume!", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Performing 3D 26-Connected Component Analysis & Noise Gating...")
    tumor_mask, metrics = analyze_tumor(orig_data, subt_data, subt_hdr, min_cluster_mm3=args.min_cluster)

    visualize_interactive_slicer(orig_data, tumor_mask, subt_hdr, metrics, no_gui=args.no_gui, show_3d=args.show_3d)

if __name__ == "__main__":
    main()
