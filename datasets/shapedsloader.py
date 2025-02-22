import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates
from torchvision import transforms 
import torch.nn.functional as F

class ShapesDataset(Dataset):
    """
    Dataset to load shape images from a folder containing files with the format {name}-{number}.gif.
    Allows filtering by the name of the object.
    """
    def __init__(self, shapes_folder, object_name=None, transform=None, resize=128, samples=None):
        """
        Args:
            shapes_folder (str): Path to the folder containing the images.
            object_name (str, optional): Name of the object to filter (e.g., "circle").
                                         If None, all images are loaded.
            transform (callable, optional): Additional transformation to apply to the image.
            resize (int or tuple, optional): Desired size to resize the image.
                                              If an integer, a square size is assumed.
        """
        self.shapes_folder = shapes_folder
        self.transform = transform
        self.resize = resize
        self.samples = samples

        # List all .gif files in the folder
        all_files = [f for f in os.listdir(shapes_folder) if f.endswith('.gif')]
        # Filter by the object name if specified
        if object_name is not None:
            self.files = [f for f in all_files if f.startswith(f"{object_name}-")]
        else:
            self.files = all_files
        self.files.sort()  # To ensure consistent order
        

        

        self.num_original = len(self.files)
        self.augmented_images = []
        self.samples = samples if samples is not None else self.num_original
        if self.samples > 0 and self.samples > self.num_original:
            num_augmented = self.samples - self.num_original
            for i in range(num_augmented):
                idx = i % self.num_original
                file_name = self.files[idx]
                file_path = os.path.join(self.shapes_folder, file_name)
                image = Image.open(file_path).convert("L")
                deformed = self.elastic_deformation(image)
                if self.transform:
                    deformed = self.transform(deformed)
                image_np = np.array(deformed, dtype=np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np).unsqueeze(0)
                self.augmented_images.append(image_tensor)  # Add the augmented image to the list
    
    def __len__(self):
        return self.samples if self.samples is not None else self.num_original
    
    def __getitem__(self, idx):
        if idx < self.num_original:
            file_name = self.files[idx]
            file_path = os.path.join(self.shapes_folder, file_name)
            image = Image.open(file_path).convert("L")
            if self.resize is not None:
                if isinstance(self.resize, int):
                    image = image.resize((self.resize, self.resize))
                elif isinstance(self.resize, tuple):
                    image = image.resize(self.resize)
            if self.transform:
                image = self.transform(image)
            image_np = np.array(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            label = file_name.split("-")[0]
            return image_tensor, label
        else:
            augmented_idx = idx - self.num_original
            label = self.files[augmented_idx % self.num_original].split("-")[0]
            augmented_tensor = self.augmented_images[augmented_idx % len(self.augmented_images)]
            if self.resize is not None:
                if isinstance(self.resize, int):
                    augmented_tensor = F.interpolate(augmented_tensor.unsqueeze(0), size=(self.resize, self.resize), mode='bilinear', align_corners=False).squeeze(0)
                elif isinstance(self.resize, tuple):
                    augmented_tensor = F.interpolate(augmented_tensor.unsqueeze(0), size=self.resize, mode='bilinear', align_corners=False).squeeze(0)
            return augmented_tensor, label


    def elastic_deformation(self, image, alpha_range=(20, 55), sigma_range=(2, 8)):

        image_np = np.array(image)
        shape = image_np.shape[:2]
        # Take random parameters
        alpha = np.random.uniform(*alpha_range)
        sigma = np.random.uniform(*sigma_range)
        # Generate random deformation fields
        random_state = np.random.RandomState(None)
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        if image_np.ndim == 3:
            deformed = np.zeros_like(image_np)
            for c in range(image_np.shape[2]):
                deformed[:, :, c] = map_coordinates(image_np[:, :, c], indices, order=1, mode='reflect').reshape(shape)
        else:
            deformed = map_coordinates(image_np, indices, order=1, mode='reflect').reshape(shape)
        # Convertir de nuevo a imagen PIL
        return Image.fromarray(deformed.astype(np.uint8))
class ShapesDataLoaderHandler:
    """
    Class to handle the DataLoader for the shapes dataset.
    """
    def __init__(self, shapes_folder, object_name, batch_size=16, resize=128, transform=None, samples=None):
        self.dataset = ShapesDataset(shapes_folder, object_name=object_name, transform=transform, resize=resize, samples=samples)
        if len(self.dataset) == 0:
            print(f"No images found for object name '{object_name}'")
            print(f"Available objects: {', '.join(set(f.split('-')[0] for f in os.listdir(shapes_folder) if f.endswith('.gif')))}")
            sys.exit(1)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    
    def show_example(self):
        for batch in self.dataloader:
            images, labels = batch
            # Show the first image of the batch along with its label
            img = images[0].squeeze().numpy()
            plt.imshow(img, cmap="gray")
            plt.title(f"Example: {labels[0]}")
            plt.axis("off")
            plt.show()
            break

    def get_all_images_tensor(self):
        """
        Returns a tensor containing all images in the dataset.
        """
        images = [self.dataset[i][0] for i in range(len(self.dataset))]
        size = images[0].shape[-1]
        print(f"Size of the images: {size}")
        return torch.stack(images)

