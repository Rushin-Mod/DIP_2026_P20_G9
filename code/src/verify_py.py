import os
import struct
import numpy as np
import vedo
from vedo import Volume
from vedo.applications import Slicer3DPlotter


def load_raw_u16_volume(bin_path: str):
    """Loads a raw binary file containing:

    - 18-byte packed header: 3x uint16 (dim_z, dim_y, dim_x) + 3x float32 (vz,
    vy, vx)
    - uint16 voxel payload (Z x Y x X)
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"File not found: {bin_path}")

    with open(bin_path, "rb") as f:
        # Read the 18-byte header
        header_bytes = f.read(18)
        dim_z, dim_y, dim_x, vz, vy, vx = struct.unpack("<HHHfff", header_bytes)

        # Total expected voxels and payload size in bytes
        total_voxels = int(dim_z) * int(dim_y) * int(dim_x)
        expected_bytes = total_voxels * 2  # 2 bytes per uint16

        # Read the raw voxel data directly into a NumPy uint16 buffer
        raw_buffer = f.read(expected_bytes)
        if len(raw_buffer) != expected_bytes:
            raise ValueError(
                f"File corrupted or incomplete: expected {expected_bytes} bytes, read {len(raw_buffer)} bytes."
            )

        voxel_data = np.frombuffer(raw_buffer, dtype=np.uint16).reshape(
            (dim_z, dim_y, dim_x)
        )

    return voxel_data, (dim_z, dim_y, dim_x), (vz, vy, vx)


def launch_mricrogl_viewer(bin_path: str):
    """Loads the binary image and opens the interactive 3-plane slicer (MRIcroGL

    style).
    """
    data, dims, spacing = load_raw_u16_volume(bin_path)
    dim_z, dim_y, dim_x = dims
    vz, vy, vx = spacing

    print(f"\n[INFO] Loaded Image: {bin_path}")
    print(f"[INFO] Dimensions (Z, Y, X): ({dim_z}, {dim_y}, {dim_x})")
    print(
        f"[INFO] Voxel Spacing (Z, Y, X): ({vz:.4f} mm, {vy:.4f} mm, {vx:.4f} mm)"
    )
    print(
        f"[INFO] Data Type: {data.dtype} | Min: {data.min()}, Max: {data.max()}"
    )

    # Reorder dimensions and voxel spacing to match VTK/Vedo conventions: (X, Y, Z)
    transposed_data = np.transpose(data, (2, 1, 0))
    transposed_spacing = (vx, vy, vz)

    # Set backend to VTK native window
    vedo.settings.default_backend = "vtk"

    # Wrap as a 3D Volume
    vol = Volume(transposed_data, spacing=transposed_spacing)

    # Launch the interactive MRIcroGL-style 3-plane orthographic slicer
    plt = Slicer3DPlotter(
        vol,
        cmaps=["gray"],
        bg="dark grey",
        axes=1,
    )

    print("[INFO] Launching viewer window. Close the window to exit.")
    plt.show(title=f"MRIcroGL Clone Viewer - {os.path.basename(bin_path)}")


if __name__ == "__main__":
    path_to_bin = input("Enter path to exported .bin file: ").strip()
    launch_mricrogl_viewer(path_to_bin)