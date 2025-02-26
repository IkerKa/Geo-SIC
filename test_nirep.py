import nibabel as nib               # type: ignore
import numpy as np
import os
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt     # type: ignore

path = "./atlas_snapshots/atlas_epoch_1.nii.gz"

#load the image from the path
image = nib.load(path)
image_data = image.get_fdata()
image_data = np.array(image_data)
print(image_data.shape)

#convert the image to a PIL image
image_data = np.array(image_data)
image_data = image_data.astype(np.uint8)
image_data = Image.fromarray(image_data)
plt.imshow(image_data)
plt.show()