import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt

def check_affine_alignment(file_paths):
    """Check if all NIfTI volumes have the same affine transformation."""
    affines = []
    
    for file in file_paths:
        if not os.path.exists(file):
            print(f"Warning: File {file} not found.")
            continue
        affines.append((file, nib.load(file).affine))

    if not affines:
        print("Error: No valid NIfTI files found.")
        return False

    reference_affine = affines[0][1]
    mismatched_files = [file for file, affine in affines if not np.allclose(affine, reference_affine)]

    if mismatched_files:
        print("Affine mismatch found in the following files:")
        for file in mismatched_files:
            print(f"- {file}")
        return False

    print("All volumes are affine aligned.")
    return True

def extract_middle_slice(file_path):
    """Extract the middle axial slice from a 3D NIfTI volume and print metadata."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    
    img = nib.load(file_path)
    data = img.get_fdata()
    middle_slice_index = data.shape[2] // 2
    middle_slice = data[:, :, middle_slice_index]

    plt.imshow(middle_slice, cmap='gray')
    plt.axis('off')
    plt.show()

    voxel_spacing = img.header.get_zooms()
    
    print(f"Archivo: {file_path}")
    print(f"Tamaño del volumen: {data.shape}")  # Dimensiones (X, Y, Z)
    print(f"Espaciado de los voxeles (mm): {voxel_spacing}")  # mm/voxel
    print(f"Tamaño del corte axial: {middle_slice.shape}")  # Dimensiones del slice

    return middle_slice

# Example usage
volume_dir = './nirep/nifti'
volume_files = [os.path.join(volume_dir, f'na{num:02d}.nii.gz') for num in range(1, 17)]

if check_affine_alignment(volume_files):
    for file in volume_files:
        extract_middle_slice(file)
        print()