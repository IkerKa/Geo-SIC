import os
import torch
import nibabel as nib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import random

def load_atlas(file_path):
    """
    Load an atlas image from file.
    
    If the file is a NIfTI (.nii or .nii.gz), use nibabel.
    If the file is a PNG, use PIL.
    
    Returns:
        atlas (numpy.ndarray): The atlas image as a 2D array.
    """
    ext = os.path.splitext(file_path)[1]
    if ext in ['.nii', '.nii.gz']:
        # Load using nibabel (assumes 3D data)
        img = nib.load(file_path)
        atlas = img.get_fdata()
        # For 3D images, take the middle slice along the first dimension
        if atlas.ndim == 3:
            slice_idx = atlas.shape[0] // 2
            atlas = atlas[slice_idx, :, :]
        # Normalize if needed (assuming intensities are not already in [0,1])
        atlas = atlas.astype(np.float32)
        atlas = (atlas - atlas.min()) / (atlas.max() - atlas.min() + 1e-8)
        return atlas
    elif ext == '.png':
        # Load using PIL and convert to grayscale
        image = Image.open(file_path).convert('L')
        atlas = np.array(image, dtype=np.float32) / 255.0
        return atlas
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def load_training_dataset(dataset_file):
    """
    Load the training dataset saved in a .pt file.
    """
    dataset = torch.load(dataset_file)
    return dataset

def mse_metric(image1, image2):
    """
    Compute Mean Squared Error (MSE) between two images.
    """
    return np.mean((image1 - image2) ** 2)

def compare_images(image1, image2):
    """
    Compare two 2D images by calculating MSE and SSIM.
        
    Returns:
        mse_val (float): Mean Squared Error.
        ssim_val (float): Structural Similarity Index.
    """
    mse_val = mse_metric(image1, image2)
    ssim_val = ssim(image1, image2, data_range=image2.max()-image2.min())
    return mse_val, ssim_val

def visualize_comparison(image1, image2, title1="Atlas", title2="Training Sample"):
    """
    Display two images side-by-side for visual comparison.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    plt.show()

def main():
    # Define paths
    atlas_folder = "atlas_snapshots"   # Folder where atlas images (png or nii.gz) are stored.
    dataset_file = "dataloaderCIRCLES.pt"          # Path to the .pt file with training images.
    
    # List all atlas files (supporting nii.gz and png)
    atlas_files = [os.path.join(atlas_folder, f) for f in os.listdir(atlas_folder) if f.endswith('.png')]

    print(atlas_files)
    atlas_files.sort()  # Order the files (e.g., by epoch)
    
    # Load the training dataset
    training_data_loader = load_training_dataset(dataset_file)
    # Assume training_data_loader is a DataLoader object
    sample_idx = random.randint(0, len(training_data_loader.dataset) - 1)
    sample_image = training_data_loader.dataset[sample_idx][0].squeeze().cpu().numpy()
    print(f"Loaded training dataset with {len(training_data_loader.dataset)} samples. Random sample index: {sample_idx}")
    
    # For each atlas file, load the atlas and compare it with the sample training image.
    for atlas_file in atlas_files:
        atlas = load_atlas(atlas_file)
        mse_val, ssim_val = compare_images(atlas, sample_image)
        print(f"Comparing atlas '{os.path.basename(atlas_file)}': MSE = {mse_val:.4f}, SSIM = {ssim_val:.4f}")
        visualize_comparison(atlas, sample_image, title1=os.path.basename(atlas_file), title2="Training Sample")
    
if __name__ == "__main__":
    main()
