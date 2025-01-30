import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import TensorDataset, Dataset
import json

class SitkDataset(Dataset):
    def __init__(self, json_file, keyword, path = None, transform=None):
        self.keyword = keyword
        self.path = None
        if path is not None:
            self.path = path
        with open(json_file, 'r') as f:
            self.data_info = json.load(f)


    def __len__(self):
        return len(self.data_info[self.keyword])
    def __getitem__(self, idx):
        src = self.data_info[self.keyword][idx]['image']
        # Load the .nii.gz file using SimpleITK 
        if self.path is not None:
            src = self.path + src

        # print(src)
        src_img = sitk.ReadImage(src)
        src_data = torch.from_numpy(sitk.GetArrayFromImage(src_img)).unsqueeze(0)

        lbl = self.data_info[self.keyword][idx]['label']

        return src_data, lbl
