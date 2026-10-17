import os
import struct
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_opening, generate_binary_structure
from nilearn.image import resample_img

def isolate_brain_and_normalize(data):
    """
    Strips the skull/scalp via adaptive thresholding and 3D morphological opening,
    then normalizes the brain tissue intensity to uint16 (0 - 65535).
    """
    # Background thresholding
    non_zero = data[data > 0]
    if len(non_zero) == 0:
        return np.zeros(data.shape, dtype=np.uint16)
    
    thresh = np.percentile(non_zero, 15)
    binary_mask = data > thresh

    # 3D morphological opening using a 6-face connected connectivity
    struct_elem = generate_binary_structure(3, 1)
    cleaned_mask = binary_opening(binary_mask, structure=struct_elem, iterations=2)

    brain_voxels = np.where(cleaned_mask, data, 0)

    # Intensity scaling to full uint16 depth
    max_val = np.max(brain_voxels)
    min_val = np.min(brain_voxels[brain_voxels > 0]) if np.any(brain_voxels > 0) else 0

    if max_val > min_val:
        norm = (brain_voxels - min_val) / (max_val - min_val)
        norm = np.clip(norm * 65535.0, 0, 65535).astype(np.uint16)
    else:
        norm = brain_voxels.astype(np.uint16)

    return np.where(cleaned_mask, norm, 0).astype(np.uint16)

def main():
    input_list_file = input("Enter path to input NIfTI file list (.txt): ").strip()
    output_dir = input("Enter target directory for exported raw binaries: ").strip()
    os.makedirs(output_dir, exist_ok=True)

    with open(input_list_file, "r") as f:
        file_paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not file_paths:
        print("[ERROR] File list is empty.")
        return

    # Target volume geometry (176, 288, 288) standard from project dataset
    target_shape = (176, 288, 288)
    
    # Load first image as canonical orientation reference
    ref_img = nib.load(file_paths[0])
    ref_img = nib.as_closest_canonical(ref_img)
    target_affine = ref_img.affine.copy()
    
    # Preserve voxel resolutions (e.g., 1.0mm x 0.972mm x 0.972mm)
    zooms = ref_img.header.get_zooms()[:3]
    voxel_size_z, voxel_size_y, voxel_size_x = float(zooms[0]), float(zooms[1]), float(zooms[2])

    print(f"[INFO] Reference Voxel Spacing: ({voxel_size_z:.4f}, {voxel_size_y:.4f}, {voxel_size_x:.4f}) mm")
    print(f"[INFO] Standardizing {len(file_paths)} scans to shape: {target_shape}...")

    manifest_lines = []

    for idx, path in enumerate(file_paths):
        if not os.path.exists(path):
            print(f"[WARN] Skipping missing file: {path}")
            continue

        base_name = os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")
        out_bin_path = os.path.join(output_dir, f"{base_name}.bin")

        # Load and align to RAS canonical orientation
        img = nib.load(path)
        img = nib.as_closest_canonical(img)

        # Spatial resampling to standardized grid shape
        resampled_img = resample_img(
            img,
            target_affine=target_affine,
            target_shape=target_shape,
            interpolation="continuous"
        )

        data = resampled_img.get_fdata(dtype=np.float32)
        processed_data = isolate_brain_and_normalize(data).astype(np.uint16)

        dim_z, dim_y, dim_x = processed_data.shape

        # Pack header: 3x uint16 (dims) + 3x float32 (voxel spacing identity)
        header_bytes = struct.pack("<HHHfff", dim_z, dim_y, dim_x, voxel_size_z, voxel_size_y, voxel_size_x)

        with open(out_bin_path, "wb") as bin_f:
            bin_f.write(header_bytes)
            bin_f.write(processed_data.tobytes())

        manifest_lines.append(out_bin_path)
        print(f"[{idx+1}/{len(file_paths)}] Processed -> {out_bin_path}")

    # Export a manifest for C++ execution
    manifest_path = os.path.join(output_dir, "binaries_manifest.txt")
    with open(manifest_path, "w") as mf:
        mf.write("\n".join(manifest_lines) + "\n")

    print(f"[SUCCESS] Export complete. Manifest written to: {manifest_path}")

if __name__ == "__main__":
    main()