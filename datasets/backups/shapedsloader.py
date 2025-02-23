import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates
from torchvision import transforms

def elastic_deformation(image, alpha_range=(20, 55), sigma_range=(2, 8)):
    image_np = np.array(image)
    shape = image_np.shape[:2]
    # Parámetros aleatorios
    alpha = np.random.uniform(*alpha_range)
    sigma = np.random.uniform(*sigma_range)
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
    return Image.fromarray(deformed.astype(np.uint8))

class ShapeDataset(Dataset):
    """
    Dataset to load images from a directory and convert them to tensors.
    """
    def __init__(self, directory, object_name, transform=None, size=None, nAugment=0):
        """
        Args:
            directory (str): Path to the directory containing the images.
            object_name (str): Name of the object to filter (e.g., "circle").
            transform (callable, optional): Additional transformation to apply on the tensor.
        """
        self.size = size
        self.transform = transform

        # -- Get all files in the directory --
        all_files = [f for f in os.listdir(directory) if f.endswith('.gif')]
        self.files = [f for f in all_files if f.startswith(f"{object_name}-")]
        self.files.sort()

        # -- Load images nd convert to grayscale --
        self.images = [Image.open(os.path.join(directory, f)).convert("RGB") for f in self.files]
        self.images = [ImageOps.grayscale(img) for img in self.images]

        # RESIZE.
        if size:
            self.images = [img.resize((size, size)) for img in self.images]


        # self.labels = [f.split("-")[1].split(".")[0] for f in self.files]
        augmentations = []
        #select a random image and add n random agumentations to the image set
        for i in range(nAugment):
            idx = np.random.randint(len(self.images))
            image = self.images[idx]
            augmentations.append(elastic_deformation(image))

        self.images += augmentations
        # self.labels += [f.split("-")[1].split(".")[0] for f in self.files]



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert the PIL image to tensor: [C, H, W] and normalize to [0,1]
        image_np = np.array(self.images[idx], dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor

class ShapesDataLoaderHandler:
    def __init__(self, directory, object_name, batch_size=16, size=None, n_aug = 0):
        self.directory = directory
        self.batch_size = batch_size
        self.dataset = ShapeDataset(directory, object_name, size=size, nAugment=n_aug)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    
    def save_dataloader(self, file_path='dataloader.pt'):
        torch.save(self.dataloader, file_path)
    
    def show_example(self):
        print(f"Number of images: {len(self.dataset)}")
        for batch in self.dataloader:
            img = batch[0].squeeze().numpy()
            #gray scale
            plt.imshow(img, cmap="gray")
            plt.title("Example of image")
            plt.axis("off")
            plt.show()
            break

    def plot_average_image(self):
        """
        Muestra la imagen promedio de todas las imágenes.
        """
        avg_image = np.zeros_like(np.array(self.dataset.images[0], dtype=np.float32))
        for img in self.dataset.images:
            avg_image += np.array(img)
        avg_image /= len(self.dataset.images)
        plt.imshow(avg_image, cmap='gray')
        plt.title("Average Image")
        plt.axis('off')
        plt.show()
