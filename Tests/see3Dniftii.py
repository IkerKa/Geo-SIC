import nibabel as nib
import numpy as np
from vedo import Volume, Text2D
from vedo.applications import Slicer2DPlotter

# Load NIfTI image
nii_img = nib.load("./nirep/nifti/na15.nii.gz")
data = nii_img.get_fdata()
data = np.array(data, dtype=np.float32)

# Create a volume
vol = Volume(data, spacing=(1, 1, 1))

# Create Slicer2DPlotter
slicer = Slicer2DPlotter(vol, bg='white')

# Add a text label
text_actor = Text2D("Slice: 0", pos="top-left", c="black")
slicer.add(text_actor)

# Function to update text when slider moves
def update_text(widget, event):
    slice_idx = int(widget.GetRepresentation().GetValue())
    text_actor.text(f"Slice: {slice_idx}")

# Manually add a slider to control the slice
slider = slicer.add_slider(
    update_text,
    xmin=0, xmax=data.shape[2] - 1,  # Assuming slices are along the z-axis
    value=0,
    pos=[(0.1, 0.05), (0.9, 0.05)],
    title="Slice Index"
)

slicer.show()
