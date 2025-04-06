
from os import PathLike
from pathlib import Path
from signal import pause
import time
import numpy as np
import SimpleITK as sitk # type: ignore
import os, glob
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
import SimpleITK as sitk # type: ignore

from Run_Atlas_trainer import initialize_network_optimizer,  read_yaml

from skimage.metrics import structural_similarity as ssim # type: ignore
from medpy.metric.binary import dc

import tqdm

#debug argument
import pdb


# Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Atlas Registration')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--pretrain', type=int, default=100, help='Number of pre-training epochs')
    return parser.parse_args()

# Load Parameters
def load_parameters():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    para = read_yaml('parameters.yml')
    return para, device

# Load Data
def load_data(nifti_datadir='nirep/nifti/', size=128, tgt_index=5, src_index=7):
    datahandler = dh(dataset_type='nifti3d', directory=nifti_datadir, size=size, seg=True)

    return datahandler.get_image(src_index), datahandler.get_image(tgt_index)


def load_debug_data(image_datadir = 'datasets/images/circle.png'):
    datahandler = dh(
            dataset_type='image_transform',
            image_path=image_datadir,
            samples=1,
            # size=(128, 128),
            shape_seg=False
        )
    
    return datahandler.get_image(0)

# Convert images to tensors
def convert_to_tensor(image, device):
    return torch.tensor(image, dtype=torch.float32).to(device).unsqueeze(0)

# Train the model
def train_model(net, optimizer, I1, I2, I1_seg, I2_seg, para, num_epochs, device):
    loss_total = 0
    phi_inv = None
    
    print('Pre-training for', num_epochs, 'epochs')

    ssim_per_epoch = []
    loss_per_epoch = []
    
    I1 = I1.to(device).float()
    I2 = I2.to(device).float()


    for epoch in tqdm.tqdm(range(num_epochs), desc="Training Epochs"):
        net.eval()
        optimizer.zero_grad()

        y_src, momentum, _, new_locs = net(I1, I2, registration=True, shooting='SVF', return_phi=True)

        dist_loss = NCC().loss(y_src, I2)
        reg_loss = Grad(penalty='l2').loss(momentum)
        
        loss_total = 10 * dist_loss + 0.001 * reg_loss
        loss_total.backward()
        optimizer.step()

        loss_per_epoch.append(loss_total.item())
        ssim_per_epoch.append(ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(),
                                   data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min()))
        
        with torch.no_grad():
            phi_inv = new_locs[0, ...]
            


    #plot graph
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(loss_per_epoch)
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].plot(ssim_per_epoch)
    ax[1].set_title('SSIM')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('SSIM')
    plt.show()
    
    return phi_inv, y_src

# Save metrics
def save_metrics(output_path, I2, y_src, mean_dice_score):
    ssim_score = ssim(I2.squeeze().cpu().detach().numpy(), y_src.squeeze().cpu().detach().numpy(),
                       data_range=y_src.squeeze().cpu().detach().numpy().max() - y_src.squeeze().cpu().detach().numpy().min())
    rmse_score = np.sqrt(np.mean((I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy())**2))

    # for slice_idx in range(0, I2.shape[2], 10):
    #     # Calculate RMSE for each slice
    #     rmse_slice = np.sqrt(np.mean((I2.squeeze().cpu().detach().numpy()[:, :, slice_idx] - y_src.squeeze().cpu().detach().numpy()[:, :, slice_idx]) ** 2))
    #     print(f'RMSE en slice {slice_idx}: {rmse_slice}')

    metrics = {'ssim': float(ssim_score), 'rmse': float(rmse_score), 'dice': float(mean_dice_score)}

    os.makedirs(output_path, exist_ok=True)

    # If there already exists a metrics.json, append to it with the next index of the execution
    # i.e
    # { idx: 0, ssim: 0.5, rmse: 0.5, dice: 0.5}
    # { idx: 1, ssim: 0.5, rmse: 0.5, dice: 0.5}
    # ...
    # { idx: n, ssim: 0.5, rmse: 0.5, dice: 0.5}

    if os.path.exists(os.path.join(output_path, 'metrics.json')):
        with open(os.path.join(output_path, 'metrics.json'), 'r') as f:
            metrics_data = json.load(f)
        idx = len(metrics_data)
    else:
        metrics_data = {}
        idx = 0

    metrics_data[idx] = metrics

    with open(os.path.join(output_path, 'metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)


#Function taken from NODEO
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


# Compute and visualize segmentation
def compute_segmentation(I1_seg, phi_inv, I2_seg, dev):

    #transpose phi_inv
    assert I1_seg.shape == I1_seg.shape, "Image and segmentation must have the same shape!"
    assert I2_seg.shape == I2_seg.shape, "Target image and segmentation must have the same shape!"

    
    
    # phi_inv = phi_inv.permute(0, 2, 3, 1)
    # phi_inv = phi_inv.permute(0, 3, 2, 1) 

    #transpose the content of phi_inv
    # phi_inv = phi_inv.permute(1, 2, 0).unsqueeze(0)

    # print(f'Phi_inv shape: {phi_inv.shape}')

    # phi_inv = phi_inv.permute(0, 3, 1, 2)  # Ensure the last dimension is 2

    # print(f'Phi_inv shape: {phi_inv.shape}')

    # phi_inv = phi_inv.permute(1, 2, 0).unsqueeze(0)
    # phi_inv = phi_inv.permute(0, 1, 3, 2)
    # phi_inv = phi_inv[..., [1, 0]]  # Swap the last two dimensions

    # print(f'Phi_inv shape: {phi_inv.shape}')    

    phi_inv = phi_inv.permute(3, 0, 1, 2).unsqueeze(0)
    print(f'Phi_inv shape: {phi_inv.shape}')
    st_seg = SpatialTransformer(size=I1_seg.shape[2:], mode='nearest').to(dev)
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

    # print(f'Mean Dice score: {dice}')


    #compute dice score for each label
    # dice_scores = {}
    # for label in labels:
    #     dice_scores[label] = dc(warped_seg_np == label, fixed_seg_np == label)
    #     # print(f'Dice score for label {label}: {dice_scores[label]}')


    # #Mean dice score
    # mean_dice_score = np.mean(list(dice_scores.values()))
    # print(f'Mean Dice score: {mean_dice_score}')


    return warped_seg_np, fixed_seg_np, dice


# Main execution block
def main():
    args = parse_arguments()
    para, device = load_parameters()

    if not os.path.exists(args.output) and args.output != None:
        os.makedirs(args.output)


    target_index = 1
    source_index = 2

    (input_image, input_segmentation), (target_image, target_segmentation) = load_data(tgt_index=target_index, src_index=source_index)

    #normalize images
    input_image = (input_image - torch.min(input_image)) / (torch.max(input_image) - torch.min(input_image))
    target_image = (target_image - torch.min(target_image)) / (torch.max(target_image) - torch.min(target_image))
    input_segmentation = (input_segmentation - torch.min(input_segmentation)) / (torch.max(input_segmentation) - torch.min(input_segmentation))
    target_segmentation = (target_segmentation - torch.min(target_segmentation)) / (torch.max(target_segmentation) - torch.min(target_segmentation))
    

    #check if segmentations has same size as images
    if input_image.shape != input_segmentation.shape:
        raise ValueError('Input image and segmentation must have the same size')
    
    I1 = convert_to_tensor(input_image, device)
    I2 = convert_to_tensor(target_image, device)
    I1_seg = convert_to_tensor(input_segmentation, device)
    I2_seg = convert_to_tensor(target_segmentation, device)

    #we have 256 silces, we have volume of 128x128x256, we take the first 128 slices

    # shape form: [B, C, H, W, D] # B=1, C=1, H=128, W=128, D=256
        
    net, _, optimizer = initialize_network_optimizer(128, 128, 256, para, device)
    phi_inv, y_src = train_model(net, optimizer, I1, I2, I1_seg, I2_seg, para, args.pretrain, device)

    slice_idx = 149

    mean_dice_score = 0 # in case we don't have segmentation
    if I1_seg is not None and I2_seg is not None:
        warped_seg_np, fixed_seg_np, mean_dice_score = compute_segmentation(I1_seg, phi_inv, I2_seg, device)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # slice_idx = warped_seg_np.shape[2] // 2  # Assuming shape [B, C, H, W, D]
        warped_seg_slice = warped_seg_np[:, :, slice_idx]  # (H, W)
        fixed_seg_slice = fixed_seg_np[:, :, slice_idx]  # (H, W)
        overlay_slice = np.maximum(warped_seg_slice, fixed_seg_slice)

        axes[0].imshow(warped_seg_slice, cmap='gray')
        axes[0].set_title('Registered Segmentation (I1 → I2)')
        axes[1].imshow(fixed_seg_slice, cmap='gray')
        axes[1].set_title('Target Segmentation (I2)')
        axes[2].imshow(overlay_slice, cmap='gray')
        axes[2].set_title('Overlay of Segmentations')

        plt.show()

    
    if args.output:
        save_metrics(args.output, I2, y_src, mean_dice_score)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    #For sagital views: [slice_idx, :, :], for coronal views: [:, slice_idx, :], for axial views: [:, :, slice_idx]
    #Take the grid from the same slice!

    # Get middle slice index (D // 2)
    # slice_idx = I1.shape[4] // 2  

    # Extract the same slice from each volume
    I1_slice = I1.squeeze().cpu().detach().numpy()[:, :, slice_idx]  # (H, W)
    I2_slice = I2.squeeze().cpu().detach().numpy()[:, :, slice_idx]  # (H, W)
    y_src_slice = y_src.squeeze().cpu().detach().numpy()[:, :, slice_idx]  # (H, W)

    # Plot each image
    axes[0].imshow(I1_slice, cmap='gray')
    axes[0].set_title('Source Image (I1)')
    axes[1].imshow(I2_slice, cmap='gray')
    axes[1].set_title('Target Image (I2)')
    axes[2].imshow(y_src_slice, cmap='gray')
    axes[2].set_title('Registered Image (y_src)')

    plt.show()

    # Error map for the selected slice
    error_map = np.abs(I2_slice - y_src_slice)
    plt.imshow(error_map, cmap='gray')
    plt.colorbar()
    plt.title("Error Map")
    plt.show()

if __name__ == '__main__':
    main()

