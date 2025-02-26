import nibabel as nib
import numpy as np
from vedo import Volume
from vedo.applications import Slicer2DPlotter

nii_img = nib.load("./nirep/nifti/na15.nii.gz")
data = nii_img.get_fdata()

data = np.array(data, dtype=np.float32)
# data = np.flip(data, axis=0).transpose(2, 1, 0)
vol = Volume(data, spacing=(1, 1, 1))

slicer = Slicer2DPlotter(vol, bg='white')
slicer.show()
