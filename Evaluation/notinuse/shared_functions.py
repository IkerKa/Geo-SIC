import json
import subprocess
import sys
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR # type: ignore
from torch.utils.data import TensorDataset, DataLoader # type: ignore
import torch # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import torch.optim as optim      # type: ignore
from easydict import EasyDict as edict  # type: ignore
import nibabel as nib #type: ignore
import random 
import yaml
import torchvision.transforms as transforms
import numpy as np # type: ignore
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(parent_dir)

from losses import NCC, MSE, Grad
from networks import UnetDense  
from SitkDataSet import SitkDataset as SData
from uEpdiff import Epdiff
from uEpdiff2D import Epdiff2D
from networks import *
import argparse
import matplotlib.pyplot as plt # type: ignore
# from datasets.datasetloader import GoogleDrawDataset2d, DataLoaderHandler
# from datasets.datasetloader3d import MHD2DDataset
# from datasets.datasetloader3d import DataLoaderHandler as d3d
# from datasets.createDataset import DataLoaderHandler as d2d
# from datasets.createDataset import ImageTransformDataset
# from datasets.shapedsloader import ShapesDataLoaderHandler as sdh
from datasets.datasethandler import DataHandler as dh
import tqdm
import SimpleITK as sitk # type: ignore

from Run_Atlas_trainer import initialize_network_optimizer2D, read_yaml

from skimage.metrics import structural_similarity as ssim # type: ignore
from medpy.metric.binary import dc

#debug argument
import pdb


#--Shared functions between the evaluation scripts--#


# Load Data
def load_data(nifti_datadir='nirep/nifti/', size=128, slice_index=149, tgt_index=5, src_index=7):
    datahandler = dh(dataset_type='nifti', directory=nifti_datadir, size=size, slice_index=slice_index, seg=True)
    return datahandler.get_image(src_index), datahandler.get_image(tgt_index)

def load_all_data(nifti_datadir='nirep/nifti/', size=128, slice_index=149):
    datahandler = dh(dataset_type='nifti', directory=nifti_datadir, size=size, slice_index=slice_index, seg=True)
    return datahandler.get_all_images()

def get_tensor_dataset(dataset, device):
    return [convert_to_tensor(image, device) for image in dataset]

def get_tensor_dataset(dataset, device):
    return [convert_to_tensor(image, device) for image in dataset]



# Convert images to tensors
def convert_to_tensor(image, device):
    return torch.tensor(image, dtype=torch.float32).to(device).unsqueeze(0)


####  DICE CALCULATION FUNCTIONS ####

def compute_dice(warped_moving, fixed, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    assert warped_moving is not None 
    assert fixed is not None 

    dicem = np.zeros(len(labels))

    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(warped_moving == label, fixed == label))
        bottom = np.sum(warped_moving == label) + np.sum(fixed == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
        
    return dicem

def compute_segmentation(I1_seg, phi_inv, I2_seg, dev):

    #transpose phi_inv
    assert I1_seg.shape == I1_seg.shape, "Image and segmentation must have the same shape!"
    assert I2_seg.shape == I2_seg.shape, "Target image and segmentation must have the same shape!"

    print(f'Phi_inv shape: {phi_inv.shape}')
    phi_inv = phi_inv.permute(2,0,1).unsqueeze(0)
    print(f'Phi_inv shape: {phi_inv.shape}')
    st_seg = SpatialTransformer(size=I1_seg.shape[2:],  mode='nearest').to(dev)
    warped_seg = st_seg(I1_seg, phi_inv)

    warped_seg_np = warped_seg.squeeze().cpu().detach().numpy()
    fixed_seg_np = I2_seg.squeeze().cpu().detach().numpy()
    # dice_score = dc(warped_seg_np, fixed_seg_np)

    #take the labels
    labels = np.unique(fixed_seg_np) 
    labels = labels[labels != 0] 

    #compute dice score for each label
    dice_scores = compute_dice(warped_seg_np, fixed_seg_np, labels)
    dice = np.mean(dice_scores)



    return warped_seg_np, fixed_seg_np, dice

######################################
