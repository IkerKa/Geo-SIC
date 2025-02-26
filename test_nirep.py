import nibabel as nib               # type: ignore
import numpy as np
import os
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt     # type: ignore

path = "nirep/nifti/"
def load_images(path):
    images = []
    for i in range(1, 17):
        filename = os.path.join(path, f"na{i:02d}.nii.gz")
        if os.path.exists(filename):
            img = nib.load(filename)
            images.append(img)
        else:
            print(f"File {filename} does not exist.")
    return images

images = load_images(path)

random_image = np.random.randint(0, len(images))
data = images[random_image].get_fdata()
print(f"Image shape: {data.shape}")

#middle slice
middle = data.shape[2] // 2
plt.imshow(data[:, :, middle], cmap="gray")
plt.axis("off")
plt.show()
