import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import io
import SimpleITK as sitk

class MHD2DDataset(Dataset):
    """
    Dataset to load 2D images from .mhd files.
    
    Each image is resized (optional) and converted to a tensor.
    """
    def __init__(self, mhd_folder, resize=None, transform=None):
        """
        Args:
            mhd_file (str): Path to the .mhd file.
            samples (int): Number of slices to process.
            resize (tuple or None): Desired size (width, height) to resize the image (e.g., (128,128)).
            transform (callable, optional): Additional transformation to apply on the tensor.
        """
        # self.samples = samples
        self.resize = resize
        self.transform = transform
        self.images = []  # PIL image set save
        
        # count the number of files in the folder
        mhd_files = [f for f in os.listdir(mhd_folder) if f.endswith('.dcm')]
        print(f"Number of .mhd files: {len(mhd_files)}")
        if len(mhd_files) == 0:
            print("No files found, you sure you are in the right folder? or the extension is .mhd, if not change the line 31")
        # Read the .mhd files
        for mhd_file in mhd_files:
            # Read the .mhd file
            itk_image = sitk.ReadImage(os.path.join(mhd_folder, mhd_file))
            image_np = sitk.GetArrayFromImage(itk_image)
            # Per each slice...
            for i, slice_np in enumerate(image_np):
                # --image PIL object--
                image = Image.fromarray(slice_np).convert("L")
                if self.resize is not None:
                    image = image.resize(self.resize)
                self.images.append(image)


    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # From PIL to Numpy to Tensor (and normalize)
        image_np = np.array(self.images[idx], dtype=np.float32) / 255.0 #[1, H, W]
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        if self.transform:  #optional
            image_tensor = self.transform(image_tensor)
        return image_tensor

class DataLoaderHandler:
    def __init__(self, mhd_folder, resize=128, batch_size=16):
        self.mhd_folder = mhd_folder
        self.resize = resize
        self.batch_size = batch_size
        self.dataset = MHD2DDataset(mhd_folder)
        print(f"Dataset length: {len(self.dataset)}")
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    
    def save_dataloader(self, file_path='dataloader.pt'):
        torch.save(self.dataloader, file_path)
    
    def show_example(self):
        for batch in self.dataloader:
            # log.info(message=f"Batch shape: {batch.shape}")
            img = batch[0].squeeze().numpy()
            plt.imshow(img, cmap="gray")
            plt.title("Example 2D Image")
            plt.axis("off")
            plt.show()
            break
