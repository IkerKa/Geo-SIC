import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

class ImageTransformDataset(Dataset):
    """
    Dataset to create multiple distorted (elastic deformed) versions of an input image.
    """
    def __init__(self, image_path, samples=100, transform=None, size=None):
        """
        Args:
            image_path (str): Path to the input image.
            samples (int): Number of distorted samples to generate.
            transform (callable, optional): Additional transformation to apply on the tensor.
        """
        self.samples = samples
        self.transform = transform
        self.image = Image.open(image_path).convert("RGB")
        #get the mean of channels to get a gray scale image
        self.image = ImageOps.grayscale(self.image)
        self.distorted_images = self.generate_distorted_images()

    def generate_distorted_images(self):
        distorted_images = []
        for _ in range(self.samples):
            img = self.image.copy()
            #resize to get a squared image
            if self.size is not None:
                new_size = (self.size, self.size)
            else:
                width, height = img.size
                new_size = 2 ** int(np.log2(min(width, height) // 2))
                
            img = ImageOps.fit(img, (new_size, new_size), method=0, bleed=0.0, centering=(0.5, 0.5))
            #gray scale
            img = self.apply_random_transformations(img)
            distorted_images.append(img)
        return distorted_images

    def elastic_deformation(self, image, alpha_range=(20, 55), sigma_range=(2, 8)):
        """
        Aplica una deformación elástica a la imagen.
        
        Args:
            image (PIL.Image): Imagen de entrada.
            alpha_range (tuple): Rango para el parámetro alpha (intensidad de la deformación).
            sigma_range (tuple): Rango para el parámetro sigma (suavizado de la deformación).
        
        Returns:
            PIL.Image: Imagen deformada.
        """
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
    
    
    def apply_random_transformations(self, img):
        """
        Aplica una transformación aleatoria a la imagen.
        En este caso, se aplica la deformación elástica con una probabilidad del 50%.
        """
        if random.random() > 0.5:
            img = self.elastic_deformation(img)
        # Aquí podrías añadir más transformaciones si lo deseas.
        return img
    
    def __len__(self):
        return len(self.distorted_images)

    def __getitem__(self, idx):
        # Convertir la imagen PIL a tensor: [C, H, W] y normalizar a [0,1]
        image_np = np.array(self.distorted_images[idx], dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor

class DataLoaderHandler:
    def __init__(self, image_path, samples=100, batch_size=16):
        self.image_path = image_path
        self.samples = samples
        self.batch_size = batch_size
        self.dataset = ImageTransformDataset(image_path, samples=samples)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    
    def save_dataloader(self, file_path='dataloader.pt'):
        torch.save(self.dataloader, file_path)
    
    def show_example(self):
        for batch in self.dataloader:
            img = batch[0].squeeze().numpy()
            plt.imshow(img)
            plt.title("Example of distorted image")
            plt.axis("off")
            plt.show()
            break
