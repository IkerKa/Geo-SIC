

#--Imports--
import json
import numpy as np
import torch                                                        #type: ignore            
from torch.utils.data import Dataset, DataLoader                    #type: ignore
import matplotlib.pyplot as plt                                     #type: ignore
from PIL import Image
import io
import os
import SimpleITK as sitk                                            #type: ignore
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates          #type: ignore
from PIL import ImageOps
import random
#------------

#--Shared functions--
def elastic_deformation(image, alpha_range=(20, 55), sigma_range=(2, 8)):

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
    
#------------------------------------------------------------   

#--Parameters: ndjson_file, samples=100, resize=None, transform=None
class GoogleDrawDataset2d(Dataset):
    """
    Dataset para cargar imágenes 2D a partir de un archivo .ndjson de Google QuickDraw.
    
    Cada dibujo se dibuja en blanco y negro, se redimensiona (opcional) y se convierte en un tensor.
    """
    def __init__(self, ndjson_file, samples=100, resize=None, transform=None):
        """
        Args:
            ndjson_file (str): Ruta al archivo .ndjson.
            samples (int): Número de dibujos a procesar.
            resize (tuple or None): Tamaño deseado (ancho, alto) para redimensionar la imagen (por ejemplo, (128,128)).
            transform (callable, optional): Transformación adicional a aplicar sobre el tensor.
        """
        self.samples = samples
        self.resize = resize
        self.transform = transform
        self.images = []  # PIL image set save
        
        # Leer el archivo .ndjson
        with open(ndjson_file, 'r') as f:
            drawings = [json.loads(line) for line in f]
        
        # Per each drawing...
        for i, drawing in enumerate(drawings[:samples]):
            fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)  #~256x256 píxeles
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.axis('off')
            ax.set_facecolor("white")  #White background

            # Plot each stroke
            for stroke in drawing["drawing"]:
                x, y = stroke[0], stroke[1]
                #invert y axis
                ax.plot(x, 255 - np.array(y), color="black", linewidth=2)

            # --image PIL object--
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, pad_inches=0)
            plt.close(fig)
            buf.seek(0)

            image = Image.open(buf).convert("L")
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
#--Parameters: mhd_folder, resize=None, transform=None
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
#--Parameters: image_path, samples=100, transform=None, size=None, shape_seg=None
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
        image = np.array(self.image)

        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        edges = cv2.Canny(blurred, 20, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(image)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        
        result = cv2.bitwise_and(image, image, mask=mask)

        self.image = Image.fromarray(result)


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
                new_size = 2 ** int(np.log2(min(width, height) // 4))
                # print(f"Resizing image to {new_size}x{new_size}")
                
            img = ImageOps.fit(img, (new_size, new_size), method=0, bleed=0.0, centering=(0.5, 0.5))
            #gray scale
            img = self.apply_random_transformations(img)
            distorted_images.append(img)
        return distorted_images

   
    
    def apply_random_transformations(self, img):
        """
        Aplica una transformación aleatoria a la imagen.
        En este caso, se aplica la deformación elástica con una probabilidad del 50%.
        """
        if random.random() > 0.5:
            img = elastic_deformation(img)
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
#--Parameters: directory, object_name, transform=None, size=None, nAugment=0
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


class DataHandler:
    def __init__(self, dataset_type, **kwargs):
        """
        Args:
            dataset_type (str): Dataset type. i.e, 'google_draw', 'mhd', 'image_transform' o 'shape'.
            **kwargs: Additional arguments for the dataset.
        """
        self.dataset = self._load_dataset(dataset_type, **kwargs)
    
    def _load_dataset(self, dataset_type, **kwargs):
        if dataset_type == 'google_draw':
            return GoogleDrawDataset2d(**kwargs)
        elif dataset_type == 'medical':
            return MHD2DDataset(**kwargs)
        elif dataset_type == 'image_transform':
            return ImageTransformDataset(**kwargs)
        elif dataset_type == 'shape':
            return ShapeDataset(**kwargs)
        else:
            raise ValueError(f"Dataloader type '{dataset_type}' not recognized.")
    
    def get_dataset(self):
        return self.dataset
