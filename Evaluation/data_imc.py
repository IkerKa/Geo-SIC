import os
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat

class NiftiMatDataset(Dataset):
    """
    Dataset para cargar imágenes y segmentaciones desde archivos .mat
    Asume nombres: NIREP_XX-Sub.mat y NIREP_XX-Seg.mat, con XX de 01 a 16.
    """
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.img_paths = [f"{dataset_path}NIREP_{i:02d}-Sub.mat" for i in range(1, 17)]
        self.seg_paths = [f"{dataset_path}NIREP_{i:02d}-Seg.mat" for i in range(1, 17)]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_data = loadmat(self.img_paths[idx])['im']
        seg_data = loadmat(self.seg_paths[idx])['seg']

        # Normalización a [0, 1]
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data) + 1e-8)

        # Convertir a tensor y añadir canal
        img_tensor = torch.from_numpy(img_data).float().unsqueeze(0)  # [1, D, H, W]
        seg_tensor = torch.from_numpy(seg_data).long().unsqueeze(0)   # [1, D, H, W]

        img_tensor = img_tensor[:, 0:176, 0:200, 0:176]
        seg_tensor = seg_tensor[:, 0:176, 0:200, 0:176]

        if self.transform:
            img_tensor = self.transform(img_tensor)
            seg_tensor = self.transform(seg_tensor)

        return img_tensor, 0

# Ejemplo de uso:
# dataset = NiftiMatDataset('./Baseline/NIREP_Matlab/')
# trainloader = DataLoader(dataset, batch_size=2, shuffle=True)