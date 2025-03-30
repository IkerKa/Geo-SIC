
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

from Run_Atlas_trainer import initialize_network_optimizer2D, read_yaml

from skimage.metrics import structural_similarity as ssim # type: ignore
from medpy.metric.binary import dc

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
def load_data(nifti_datadir='nirep/nifti/', size=128, slice_index=149, tgt_index=5, src_index=7):
    datahandler = dh(dataset_type='nifti', directory=nifti_datadir, size=size, slice_index=slice_index, seg=True)
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
def train_model(net, optimizer, I1, I2, I1_seg, I2_seg, para, num_epochs, output_path):
    loss_total = 0
    phi_inv = None
    
    print('Pre-training for', num_epochs, 'epochs')

    ssim_per_epoch = []
    loss_per_epoch = []
    
    for epoch in range(num_epochs):
        net.eval()
        optimizer.zero_grad()
        y_src, momentum, _, new_locs = net(I1, I2, registration=True, shooting='SVF', return_phi=True)
        
        dist_loss = NCC(win=[21,21]).loss(y_src, I2)
        reg_loss = Grad(penalty='l2').loss2D(momentum)
        
        loss_total = dist_loss + reg_loss
        loss_total.backward()
        optimizer.step()

        loss_per_epoch.append(loss_total.item())
        ssim_per_epoch.append(ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(),
                                   data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min()))
        
        
        with torch.no_grad():
            # print(f'New locations shape: {new_locs.shape}')
            phi_inv = new_locs[0, ...]
            final_new_locs = new_locs
            # print(f'Phi_inv shape: {phi_inv.shape}')
            # print(f'Final new locations shape: {final_new_locs.shape}')

            


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
    metrics = {'ssim': float(ssim_score), 'rmse': float(rmse_score), 'dice': float(mean_dice_score)}

    print(f'SSIM: {ssim_score}')
    print(f'RMSE: {rmse_score}')
    print(f'Dice: {mean_dice_score}')

    with open(os.path.join(output_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)


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

    # phi_inv = phi_inv.unsqueeze(0)
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

    _debug = False

    if _debug:
        # circle_path = 'datasets/images/circle.png'
        # input_image = load_debug_data(circle_path)
        # #read image as target
        # target_image = Image.open(circle_path).convert('L')
        # transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        # target_image = transform(target_image).unsqueeze(0).to(device)
        # target_image = target_image.squeeze(0)
        # #resize input image to 128x128
        # input_image = transforms.Resize((128, 128))(input_image)

        #read circle.png and circle_2.png and convert them to tensors
        target_image = Image.open('datasets/images/circle.png').convert('L')
        input_image = Image.open('datasets/images/circle_2.png').convert('L')
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        input_image = transform(input_image).to(device)
        target_image = transform(target_image).to(device)


        #convert both to tensor
        I1 = convert_to_tensor(input_image, device)
        I2 = convert_to_tensor(target_image, device)
        I1_seg = None
        I2_seg = None

        print(f'Input image shape: {I1.shape}')
        print(f'Target image shape: {I2.shape}')


        #plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].imshow(I1.squeeze().cpu().detach().numpy(), cmap='gray')
        axes[0].set_title('Source Image (I1)')
        axes[1].imshow(I2.squeeze().cpu().detach().numpy(), cmap='gray')
        axes[1].set_title('Target Image (I2)')
        plt.show()


    else:
        #if there isnt an output directory, create it
        if not os.path.exists(args.output) and args.output != None:
            os.makedirs(args.output)



        target_index = 1
        source_index = 2

        (input_image, input_segmentation), (target_image, target_segmentation) = load_data(tgt_index=target_index, src_index=source_index)
        
        print(f'Input image shape: {input_image.shape}')
        print(f'Target image shape: {target_image.shape}')

        #check if segmentations has same size as images
        if input_image.shape != input_segmentation.shape:
            raise ValueError('Input image and segmentation must have the same size')
        
        I1 = convert_to_tensor(input_image, device)
        I2 = convert_to_tensor(target_image, device)
        I1_seg = convert_to_tensor(input_segmentation, device)
        I2_seg = convert_to_tensor(target_segmentation, device)
        
    net, _, optimizer = initialize_network_optimizer2D(128, 128, para, device)
    phi_inv, y_src = train_model(net, optimizer, I1, I2, I1_seg, I2_seg, para, args.pretrain, args.output)

    mean_dice_score = 0

    
    if I1_seg is not None and I2_seg is not None:
        warped_seg_np, fixed_seg_np, mean_dice_score = compute_segmentation(I1_seg, phi_inv, I2_seg, device)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(warped_seg_np, cmap='gray')
        axes[0].set_title('Registered Segmentation (I1 → I2)')
        axes[1].imshow(fixed_seg_np, cmap='gray')
        axes[1].set_title('Target Segmentation (I2)')
        overlay = np.maximum(warped_seg_np, fixed_seg_np)
        axes[2].imshow(overlay, cmap='gray')
        axes[2].set_title('Overlay of Segmentations')
        plt.show()

    
    if args.output:
        save_metrics(args.output, I2, y_src, mean_dice_score)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(I1.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[0].set_title('Source Image (I1)')
    axes[1].imshow(I2.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[1].set_title('Target Image (I2)')
    axes[2].imshow(y_src.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[2].set_title('Registered Image (y_src)')
    plt.show()
    
    plt.imshow(np.abs(I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()), cmap='gray')
    plt.colorbar()
    plt.title("Error Map")
    plt.show()
    

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(I1.squeeze().cpu().detach().numpy(), cmap='gray')

    H, W = I1.shape[2], I1.shape[3]  # Asumiendo formato [B, C, H, W]

    #convert phi from -1,1 to image 
    phi_inv_np = phi_inv.cpu().detach().numpy()
    phi_inv_x = (phi_inv_np[:, :, 1] + 1) * (W - 1) / 2  # X coordinates
    phi_inv_y = (phi_inv_np[:, :, 0] + 1) * (H - 1) / 2  # Y coordinates

    #transpose the phi_inv
    phi_inv_x = np.transpose(phi_inv_x)
    phi_inv_y = np.transpose(phi_inv_y)


    
    interval = 2
    for row in range(0, H, interval):
        ax.plot(phi_inv_x[row, :], phi_inv_y[row, :], 'm')  
    for col in range(0, W, interval):
        ax.plot(phi_inv_x[:, col], phi_inv_y[:, col], 'm')  

    plt.title("Diffeomorphic deformation grid overlaid on Source Image")
    plt.show()


    # Plot deformation

    fig, ax = plt.subplots(figsize=(6, 6))
    interval = 2

    for row in range(0, phi_inv.shape[0], interval):
        ax.plot(phi_inv[row, :, 0].cpu().detach().numpy(),  
                phi_inv[row, :, 1].cpu().detach().numpy(),  
                'm')

    for col in range(0, phi_inv.shape[1], interval):
        ax.plot(phi_inv[:, col, 0].cpu().detach().numpy(),  
                phi_inv[:, col, 1].cpu().detach().numpy(),  
                'm')

    plt.title("Diffeomorphic deformation grid")

    plt.show()

if __name__ == '__main__':
    main()


#--EXTRA CONTENT


# I1_np = I1.squeeze().cpu().detach().numpy()
# I2_np = I2.squeeze().cpu().detach().numpy()
# y_src_np = y_src.squeeze().cpu().detach().numpy()

# sitk.WriteImage(sitk.GetImageFromArray(I1_np), output_path + '/I1.nii')
# sitk.WriteImage(sitk.GetImageFromArray(I2_np), output_path + '/I2.nii')
# sitk.WriteImage(sitk.GetImageFromArray(y_src_np), output_path + '/y_src.nii')




# X, Y = np.meshgrid(np.arange(phi_inv.shape[1]), np.arange(phi_inv.shape[0]))

# # Extraer desplazamientos
# U = phi_inv[:, :, 1].cpu().detach().numpy() - X  # Diferencia en X
# V = phi_inv[:, :, 0].cpu().detach().numpy() - Y  # Diferencia en Y

# # Visualizar
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.imshow(I1.squeeze().cpu().detach().numpy(), cmap='gray')  # Imagen base
# ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='m')

# plt.title("Mapa de flujo de deformación")
# plt.show()


# I1_seg_warped = F.grid_sample(I1_seg, phi_inv.unsqueeze(0), mode='nearest')

# S1_warped_np = I1_seg_warped.squeeze().cpu().detach().numpy()
# S2_np = I2_seg.squeeze().cpu().detach().numpy()

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0].imshow(S1_warped_np, cmap='gray')
# axes[0].set_title('Segmentación registrada (I1 → I2)')

# axes[1].imshow(S2_np, cmap='gray')
# axes[1].set_title('Segmentación real (I2)')

# overlay = np.maximum(S1_warped_np, S2_np)
# axes[2].imshow(overlay, cmap='gray')
# axes[2].set_title('Superposición de segmentaciones')

# plt.show()



#-Preguntas:
# 1. Tengo alguna manera de observar los v(x,t) en cada iteracion? para ver si al usar SVF no cambian.
# 2. El momentum es lo que transforma la imagen I1 a I2
# 3. Otra manera de evaluar el regsitro es usar NODEO?