import nibabel as nib
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2

def load_nifti_image(file_path):

    img = nib.load(file_path)
    return img.get_fdata()

def get_middle_slice(volume):

    z_idx = volume.shape[2] // 2
    slice_2d = volume[:, :, z_idx]

    slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)

    slice_2d = cv2.resize(slice_2d, (256, 256), interpolation=cv2.INTER_LINEAR)
    
    slice_2d = cv2.rotate(slice_2d, cv2.ROTATE_90_CLOCKWISE)
    
    return slice_2d

def compute_average_image():
    data_path = './nirep/nifti/na*.nii.gz'
    nii_files = glob.glob(data_path)
    nii_files = [file for file in nii_files if not file.endswith('_seg.nii.gz')]

    print(f"Found {len(nii_files)} files")

    data_arrays = []
    for file in nii_files:
        img = load_nifti_image(file)
        data_arrays.append(img)

    average_volume = np.mean(data_arrays, axis=0)
    print(f"Average volume shape: {average_volume.shape}")
    return get_middle_slice(average_volume) 

def compare_images(image1, image2):
    difference = np.abs(image1 - image2)
    difference = (difference - np.min(difference)) / (np.max(difference) - np.min(difference) + 1e-8)
    return difference

def main():
    atlas_path = './backup/b14/atlas_epoch_500.nii.gz'
    atlas_slice = load_nifti_image(atlas_path)


    avg_slice = compute_average_image()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(atlas_slice, cmap='gray')
    plt.title('Atlas Image')

    plt.subplot(1, 3, 2)
    plt.imshow(avg_slice, cmap='gray')
    plt.title('Average Image')

    difference_image = compare_images(atlas_slice, avg_slice)

    plt.subplot(1, 3, 3)
    plt.imshow(difference_image, cmap='hot') 
    plt.title('Difference Image')

    plt.show()

if __name__ == "__main__":
    main()
