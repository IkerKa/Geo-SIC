import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import random
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates

class ImageTransformDataset(Dataset):
    """
    Dataset to create multiple distorted (elastic deformed) versions of an input image.
    """
    def __init__(self, image_path, samples=100, transform=None, size=None, shape_seg=None):
        """
        Args:
            image_path (str): Path to the input image.
            samples (int): Number of distorted samples to generate.
            transform (callable, optional): Additional transformation to apply on the tensor.
        """
        self.size = size
        self.shape_seg = shape_seg
        self.samples = samples
        self.transform = transform
        self.image = Image.open(image_path).convert("RGB")
        #get the mean of channels to get a gray scale image
        self.image = ImageOps.grayscale(self.image)
        self.distorted_images = self.generate_distorted_images()
 
       

    def apply_segmentation(self):
        # Convertir la imagen a numpy y asegurarse de que es en escala de grises
        image_np = np.array(self.image)
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(image_np, (3, 3), 0)
        
        _, mask = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=10)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            biggest_contour = max(contours, key=cv2.contourArea)
            new_mask = np.zeros_like(mask)
            cv2.drawContours(new_mask, [biggest_contour], -1, 255, thickness=cv2.FILLED)
        else:
            new_mask = mask
        
        self.image = Image.fromarray(new_mask)

    def generate_distorted_images(self):

        if self.shape_seg:
            #apply the segmentation to work in shape space
            self.apply_segmentation()

        distorted_images = []
        for _ in range(self.samples):
            img = self.image.copy()
            #resize to get a squared image
            if self.size is not None:
                new_size = (self.size, self.size)
            else:
                width, height = img.size
                new_size = 2 ** int(np.log2(min(width, height) // 2))
                # print(f"Resizing image to {new_size}x{new_size}")
                
            img = ImageOps.fit(img, (new_size, new_size), method=0, bleed=0.0, centering=(0.5, 0.5))
            #gray scale
            img = self.apply_random_transformations(img)
            distorted_images.append(img)
        return distorted_images

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
    def __init__(self, image_path, samples=100, batch_size=16, size=None, shape_seg=None):
        self.image_path = image_path
        self.samples = samples
        self.batch_size = batch_size
        self.dataset = ImageTransformDataset(image_path, samples=samples, size=size, shape_seg=shape_seg)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    
    def save_dataloader(self, file_path='dataloader.pt'):
        torch.save(self.dataloader, file_path)
    
    def show_example(self):
        for batch in self.dataloader:
            img = batch[0].squeeze().numpy()
            #gray scale
            plt.imshow(img, cmap="gray")
            plt.title("Example of distorted image")
            plt.axis("off")
            plt.show()
            break

    def plot_average_image(self):
        """
        Muestra la imagen promedio de todas las imágenes distorsionadas.
        """
        avg_image = np.zeros_like(np.array(self.dataset.distorted_images[0], dtype=np.float32))
        for img in self.dataset.distorted_images:
            # print(len(self.dataset.distorted_images))
            avg_image += np.array(img)
        avg_image /= len(self.dataset.distorted_images)
        plt.imshow(avg_image, cmap='gray')
        plt.title("Average Distorted Image")
        plt.axis('off')
        plt.show()