import nibabel as nib
import numpy as np
import glob

import matplotlib.pyplot as plt

# Path to the directory containing the .nii.gz files
data_path = './nirep/nifti/na*.nii.gz'
# Load all .nii.gz files only that starts wit na_{number}.ni.gz
# filter eliminating the ones that ends with _seg.nii.gz
nii_files = glob.glob(data_path)
nii_files = [file for file in nii_files if not file.endswith('_seg.nii.gz')]



print(len(nii_files))

#filter files
# nii_files = [file for file in nii_files if np.basename(file).startswith('na_') and not file.endswith('_seg.nii.gz')]
data_arrays = []

for file in nii_files:
    img = nib.load(file)
    data = img.get_fdata()
    data_arrays.append(data)

# Calculate the average of the arrays
average = np.mean(data_arrays, axis=0)

# Plot the average (selecting the middle slice of the 3D array)
middle_slice = average[:, :, average.shape[2] // 2]
plt.imshow(middle_slice, cmap='gray')

plt.show()
