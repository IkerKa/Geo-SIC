import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import io

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
        self.images = []  # Aquí se almacenarán las imágenes procesadas (tipo PIL)
        
        # Leer el archivo .ndjson
        with open(ndjson_file, 'r') as f:
            drawings = [json.loads(line) for line in f]
        
        # Procesar cada dibujo
        for i, drawing in enumerate(drawings[:samples]):
            # Crear figura y ejes para dibujar
            fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)  # Esto crea una imagen de ~256x256 píxeles
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.axis('off')
            ax.set_facecolor("white")  # Fondo blanco

            # Dibujar cada trazo
            for stroke in drawing["drawing"]:
                x, y = stroke[0], stroke[1]
                # Se invierte el eje Y para que el dibujo quede orientado correctamente
                ax.plot(x, 255 - np.array(y), color="black", linewidth=2)

            # Guardar la figura en un buffer de memoria en lugar de en disco
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, pad_inches=0)
            plt.close(fig)
            buf.seek(0)

            # Abrir la imagen con PIL y convertirla a escala de grises
            image = Image.open(buf).convert("L")
            if self.resize is not None:
                image = image.resize(self.resize)
            self.images.append(image)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Convertir la imagen PIL a un array NumPy y normalizar los valores [0,1]
        image_np = np.array(self.images[idx], dtype=np.float32) / 255.0
        # Agregar la dimensión del canal para obtener [1, H, W]
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor

class DataLoaderHandler:
    def __init__(self, ndjson_file, samples=100, resize=128, batch_size=16):
        self.ndjson_file = ndjson_file
        self.samples = samples
        self.resize = resize
        self.batch_size = batch_size
        self.dataset = GoogleDrawDataset2d(ndjson_file, samples=samples, resize=(resize, resize))
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    
    def save_dataloader(self, file_path='dataloader.pt'):
        torch.save(self.dataloader, file_path)
        print(f"Dataloader guardado en '{file_path}'")
    
    def show_example(self):
        for batch in self.dataloader:
            print("Tamaño del batch:", batch.shape)  # Debería mostrar: [batch_size, 1, H, W]
            # Mostrar la primera imagen del batch usando matplotlib
            img = batch[0].squeeze().numpy()
            plt.imshow(img, cmap="gray")
            plt.title("Ejemplo de imagen 2D")
            plt.axis("off")
            plt.show()
            break
