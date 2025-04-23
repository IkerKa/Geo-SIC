import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


import matplotlib.pyplot as plt

def load_nifti(file_path):
    """
    Load a NIfTI file as a PyTorch tensor.
    """
    nifti_image = nib.load(file_path)
    image_np = nifti_image.get_fdata()
    image_tensor = torch.tensor(image_np, dtype=torch.float32)
    return image_tensor.unsqueeze(0)  # Add batch dimension

def load_grid(file_path):
    """
    Load a saved numpy grid file as a PyTorch tensor.
    """
    grid_np = np.load(file_path)
    grid_tensor = torch.tensor(grid_np, dtype=torch.float32)
    return grid_tensor


# Path to the folder containing .nii.gz files
folder_path = './storage/'  # Replace with your folder path

# List all .nii.gz files in the folder
nii_files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]
numpy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# Function to plot a slice from a 3D volume
def plot_slice(slice, title):
    slice = slice.squeeze().cpu().detach().numpy()  # Remove batch dimension and convert to numpy
    plt.imshow(slice, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_grid(phi_inv, interval, title):
    """
    Plot a diffeomorphic deformation grid.
    """
    fig, ax = plt.subplots()
    for row in range(0, phi_inv.shape[0], interval):
        ax.plot(phi_inv[row, :, 0],
            phi_inv[row, :, 1],
            'm')

    for col in range(0, phi_inv.shape[1], interval):
        ax.plot(phi_inv[:, col, 0],
            phi_inv[:, col, 1],
            'm')

    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# Read and plot each .nii.gz file
for nii_file in nii_files:
    
    file_path = os.path.join(folder_path, nii_file)
    # Load the NIfTI file
    data = load_nifti(file_path)

    # Plot the 2D slice directly
    plot_slice(data, f"Slice of {nii_file}")


# Read and plot each .npy file
for numpy_file in numpy_files:
    
    file_path = os.path.join(folder_path, numpy_file)
    # Load the numpy grid file
    grid = load_grid(file_path)

    # Plot the diffeomorphic deformation grid
    plot_grid(grid, interval=5, title=f"Grid of {numpy_file}")