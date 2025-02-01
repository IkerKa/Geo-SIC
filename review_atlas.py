import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def load_nifti(file_path):
    """Loads a NIfTI file and returns a NumPy array."""
    image = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(image)

def compare_atlases(atlas1_path, atlas2_path):
    """Compares two atlases using mean difference, histogram, and SSIM."""
    
    atlas1 = load_nifti(atlas1_path)
    atlas2 = load_nifti(atlas2_path)

    # Compute differences
    diff = atlas2 - atlas1
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    ssim_index = ssim(atlas1, atlas2, data_range=atlas2.max() - atlas2.min())

    print(f"Mean Difference: {mean_diff:.4f}")
    print(f"Standard Deviation of Difference: {std_diff:.4f}")
    print(f"SSIM (Structural Similarity Index): {ssim_index:.4f}")

    # Plot differences
    plt.figure(figsize=(15, 5))

    # Middle slice comparison
    slice_idx = atlas1.shape[0] // 2
    
    plt.subplot(1, 3, 1)
    plt.imshow(atlas1[slice_idx], cmap="gray")
    plt.title("Atlas 1 (Before)")

    plt.subplot(1, 3, 2)
    plt.imshow(atlas2[slice_idx], cmap="gray")
    plt.title("Atlas 2 (After)")

    plt.subplot(1, 3, 3)
    plt.imshow(diff[slice_idx], cmap="hot")
    plt.title("Difference (After - Before)")

    plt.tight_layout()
    plt.show()


def see_atlas(atlas_path):
    """Visualizes the atlas."""
    atlas = load_nifti(atlas_path)
    plt.figure(figsize=(15, 5))
    if len(atlas.shape) == 4:
        slice_idx = atlas.shape[2] // 2
        plt.imshow(atlas[0, 0, slice_idx, :, :], cmap='gray')
    else:
        slice_idx = atlas.shape[0] // 2
        plt.imshow(atlas[slice_idx], cmap='gray')
    plt.title("Atlas")
    plt.show()


if __name__ == "__main__":
    path = "atlas_snapshots/"
    #names are epoch_0, epoch_1, epoch_2, epoch_3, epoch_4
    atlas1_path = path + "atlas_epoch_0.nii.gz"
    atlas2_path = path + "atlas_epoch_2.nii.gz"

    see_atlas(atlas1_path)
    see_atlas(atlas2_path)
    compare_atlases(atlas1_path, atlas2_path)
