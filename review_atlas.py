import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from sklearn.metrics import mutual_info_score

def load_nifti(file_path):
    """Loads a NIfTI file and returns a NumPy array."""
    image = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(image)

def compare_atlases(atlas1_path, atlas2_path):
    """Compares two atlases using multiple metrics and visualizations."""
    
    atlas1 = load_nifti(atlas1_path)
    atlas2 = load_nifti(atlas2_path)

    # Compute global intensity range for metrics
    global_min = min(atlas1.min(), atlas2.min())
    global_max = max(atlas1.max(), atlas2.max())
    data_range = global_max - global_min

    # Compute differences
    diff = atlas2 - atlas1
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    # Calculate similarity metrics
    ssim_index = ssim(atlas1, atlas2, data_range=data_range)
    mse = np.mean((atlas1 - atlas2) ** 2)
    psnr_value = peak_signal_noise_ratio(atlas1, atlas2, data_range=data_range)
    corr_coef = np.corrcoef(atlas1.ravel(), atlas2.ravel())[0, 1]

    print(f"Mean Difference: {mean_diff:.4f}")
    print(f"Standard Deviation of Difference: {std_diff:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"PSNR: {psnr_value:.4f} dB")
    print(f"SSIM: {ssim_index:.4f}")
    print(f"Pearson's r: {corr_coef:.4f}")

    hist_2d, _, _ = np.histogram2d(atlas1.ravel(), atlas2.ravel(), bins=20)
    mi = mutual_info_score(None, None, contingency=hist_2d)
    print(f"Mutual Information: {mi:.4f}")

    # Plot comparisons
    plt.figure(figsize=(15, 5))
    slice_idx = atlas1.shape[0] // 2
    
    plt.subplot(1, 4, 1)
    plt.imshow(atlas1[slice_idx], cmap="gray")
    plt.title("Atlas 1 (Before)")

    plt.subplot(1, 4, 2)
    plt.imshow(atlas2[slice_idx], cmap="gray")
    plt.title("Atlas 2 (After)")

    plt.subplot(1, 4, 3)
    plt.imshow(diff[slice_idx], cmap="hot")
    plt.title("Difference (After - Before)")

    
    plt.tight_layout()
    plt.show()

    # Histogram comparison
    plt.figure()
    plt.hist(atlas1.ravel(), bins=256, alpha=0.5, label='Atlas 1')
    plt.hist(atlas2.ravel(), bins=256, alpha=0.5, label='Atlas 2')
    plt.legend()
    plt.title("Intensity Histograms")
    plt.show()

    # Zoomed difference view
    plt.figure()
    plt.imshow(diff[slice_idx], cmap="hot", vmin=-0.005, vmax=0.005)
    plt.colorbar()
    plt.title("Difference (Enhanced Contrast)")
    plt.show()

    if np.issubdtype(atlas1.dtype, np.integer):
        mismatch_percent = 100 * np.sum(atlas1 != atlas2) / atlas1.size
        print(f"Label mismatch: {mismatch_percent:.6f}%")

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
    atlas1_path = path + "atlas_epoch_0.nii.gz"
    atlas2_path = path + "atlas_epoch_12.nii.gz"

    see_atlas(atlas1_path)
    see_atlas(atlas2_path)
    compare_atlases(atlas1_path, atlas2_path)