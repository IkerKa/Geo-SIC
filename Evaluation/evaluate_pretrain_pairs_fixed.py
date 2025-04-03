
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
import tqdm

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

def load_all_data(nifti_datadir='nirep/nifti/', size=128, slice_index=149):
    datahandler = dh(dataset_type='nifti', directory=nifti_datadir, size=size, slice_index=slice_index, seg=True)
    return datahandler.get_all_images()

def get_tensor_dataset(dataset, device):
    return [convert_to_tensor(image, device) for image in dataset]


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


def plot_pairs(image1, image2, title1='Image 1', title2='Image 2'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(image1.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[0].set_title(title1)
    axes[1].imshow(image2.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[1].set_title(title2)
    plt.show()



# Train the model
def train_model(net, optimizer, num_epochs, train_dataset, test_dataset, train_shapes, test_shapes, weight_dist, weight_reg,device):
    
    loss_total = 0
    phi_inv = None
    
    print('Pre-training for', num_epochs, 'epochs')

    ssim_per_epoch = []
    loss_per_epoch = []
    dices_per_epoch = []

    I1_fixed = test_dataset[0]
    I1_fixed_shape = test_shapes[0]

    phi_inv, y_src = None, None

    tqdm_epochs = tqdm.tqdm(range(num_epochs))

    #possible pairs with respect to the fixed image
    pairs = [(I1_fixed, I2) for I2 in train_dataset]
    shape_pairs = [(I1_fixed_shape, I2) for I2 in train_shapes]
    for epoch in tqdm_epochs:
        tqdm_epochs.set_description(f'Epoch {epoch + 1}/{num_epochs}')
        net.train()

        pairs_loss = []
        pairs_ssim = []
        pairs_dice = []

        for pair_idx, (I1, I2) in enumerate(pairs):
            I1 = I1.to(device).float()
            I2 = I2.to(device).float()
            I1_shape = shape_pairs[pair_idx][0].to(device).float()
            I2_shape = shape_pairs[pair_idx][1].to(device).float()


            y_src, momentum, _, new_locs = net(I1, I2, registration=True, shooting='SVF', return_phi=True)
            momentum = momentum.permute(0, 3, 1, 2) 
            # dist_loss = NCC(win=[9,9]).loss(I2, y_src)
            dist_loss = MSE().loss(I2, y_src) #MSE loss
            reg_loss = Grad(penalty='l2').loss2D(momentum)
            
            loss_total = (1 * dist_loss +  1 * reg_loss) / 2.0
            loss_total.backward(retain_graph=True)

            #compute the DICE metric
            _,_, dice_score = compute_segmentation(I1_shape, new_locs[0,...], I2_shape, device)
            pairs_dice.append(dice_score)
          
            if (pair_idx + 1) % 2 == 0 or pair_idx == len(pairs) - 1: 
                optimizer.step()
                optimizer.zero_grad()

            pairs_loss.append(loss_total.item())
            pairs_ssim.append(ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(),
                                    data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min()))
            
            with torch.no_grad():
                phi_inv = new_locs[0, ...]

        loss_per_epoch.append(np.mean(pairs_loss))
        ssim_per_epoch.append(np.mean(pairs_ssim))
        dices_per_epoch.append(np.mean(pairs_dice))

        tqdm_epochs.set_postfix(loss=np.mean(pairs_loss), ssim=np.mean(pairs_ssim), dice=np.mean(pairs_dice))
                


    #plot graph
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(loss_per_epoch)
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].plot(ssim_per_epoch)
    ax[1].set_title('SSIM')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('SSIM')
    ax[2].plot(dices_per_epoch)
    ax[2].set_title('DICE')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('DICE')
    plt.tight_layout()
    plt.show()
    
    return phi_inv, y_src

def compute_segmentation(I1_seg, phi_inv, I2_seg, dev):


    phi_inv = phi_inv.permute(2,0,1).unsqueeze(0)
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



def net_test_model(net, test_images, test_segs, train_dataset, train_shapes, device, args):
    # with the trained model, test the images (phi inverted)

    # can be also use the net?
    pairs = [(i, j) for i in range(len(test_images)) for j in range(len(test_images))]
    print(f'Testing {len(pairs)} pairs of images...')
    phiinvs = []
    y_srcs = []

    with torch.no_grad():
        net.eval()
        for i, j in pairs:
            I1 = test_images[i].to(device).float()
            I2 = test_images[j].to(device).float()
            I1_seg = test_segs[i]
            I2_seg = test_segs[j]

            net.eval()  # Modo evaluación
            y_src, _, _, new_locs = net(I1, I2, registration=True, shooting="SVF", return_phi=True)
            _, _, dice_score = compute_segmentation(I1_seg, new_locs[0,...], I2_seg, device)

            #obtain the metrics and save them
            ssim_score = ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(), data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min())
            rmse_score = np.sqrt(np.mean((I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()) ** 2))
            mean_dice_score = np.mean(dice_score)

            phiinvs.append(new_locs[0, ...])
            y_srcs.append(y_src)

            print(f'Test - SSIM: {ssim_score:.4f}, RMSE: {rmse_score:.4f}, Dice: {mean_dice_score:.4f}')

        plot_results(test_images, test_segs, phiinvs, y_srcs)


def plot_results(test_images, test_shapes, phiinvs, y_srcs):
    
    #for every possible pair
    pairs = [(i, j) for i in range(len(test_images)) for j in range(len(test_images))]
    shapes_pairs = [(i, j) for i in range(len(test_shapes)) for j in range(len(test_shapes))]

    for pair_idx, (pair) in enumerate(pairs):
        I1 = test_images[pair[0]]
        I2 = test_images[pair[1]]
        I1_seg = test_shapes[shapes_pairs[pair_idx][0]]
        I2_seg = test_shapes[shapes_pairs[pair_idx][1]]

        phi_inv = phiinvs[pair_idx]
        y_src = y_srcs[pair_idx]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(I1.squeeze().cpu().detach().numpy(), cmap='gray')
        axes[0].set_title('Source Image (I1)')
        axes[1].imshow(I2.squeeze().cpu().detach().numpy(), cmap='gray')
        axes[1].set_title('Target Image (I2)')
        axes[2].imshow(y_src.squeeze().cpu().detach().numpy(), cmap='gray')
        axes[2].set_title('Registered Image (y_src)')
        plt.show()

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






            
def evaluate_model(phi_inv, test_dataset, test_shapes, train_dataset, train_shapes, device, args):


    #Take the test images
    I1 = convert_to_tensor(test_dataset[0], device)
    I2 = convert_to_tensor(test_dataset[1], device)

    I1_seg = convert_to_tensor(test_shapes[0], device)
    I2_seg = convert_to_tensor(test_shapes[1], device)

    # I1 composed with phi_inv should be close to I2 -> warping I1 with phi_inv should give I2
    y_src = F.grid_sample(I1, phi_inv.unsqueeze(0), align_corners=True, mode='bilinear')

    warped_seg_np, fixed_seg_np, dice_score = compute_segmentation(I1_seg, phi_inv, I2_seg, device)

    # Save metrics
    save_metrics(args.output, I2, y_src, dice_score)


    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(I1.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[0].set_title('Source Image (I1)')
    axes[1].imshow(I2.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[1].set_title('Target Image (I2)')
    axes[2].imshow(y_src.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[2].set_title('Registered Image (y_src)')
    plt.show()


    fig, ax = plt.subplots(figsize=(6, 6))
    mappable = ax.imshow(phi_inv.cpu().detach().numpy()[:, :, 0], cmap='jet')  
    plt.title("Phi_inv X")
    fig.colorbar(mappable, ax=ax)  
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    mappable = ax.imshow(phi_inv.cpu().detach().numpy()[:, :, 1], cmap='jet')  
    plt.title("Phi_inv Y")
    fig.colorbar(mappable, ax=ax)  
    plt.show()

    
    #Collage for the absolute error
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[0].set_title('Error')
    axes[1].imshow(I2.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[1].set_title('Target Image (I2)')
    axes[2].imshow(y_src.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[2].set_title('Registered Image (y_src)')
    
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(I1.squeeze().cpu().detach().numpy(), cmap='gray')

    H, W = I1.shape[2], I1.shape[3]  # Asumiendo formato [B, C, H, W]

    #convert phi from -1,1 to image 
    phi_inv_np = phi_inv.cpu().detach().numpy()
    phi_inv_x = (phi_inv_np[:, :, 0] + 1) * (W - 1) / 2  # X coordinates
    phi_inv_y = (phi_inv_np[:, :, 1] + 1) * (H - 1) / 2  # Y coordinates

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

    # Plot the segmentation error between warped and I2_Seg
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(warped_seg_np, cmap='gray')
    axes[0].set_title('Warped Segmentation')
    axes[1].imshow(fixed_seg_np, cmap='gray')
    axes[1].set_title('Fixed Segmentation')
    axes[2].imshow((warped_seg_np - fixed_seg_np), cmap='gray')
    axes[2].set_title('Segmentation Error')
    plt.show()




# Save metrics
def save_metrics(output_path, I2, y_src, mean_dice_score):
    ssim_score = ssim(I2.squeeze().cpu().detach().numpy(), y_src.squeeze().cpu().detach().numpy(),
                       data_range=y_src.squeeze().cpu().detach().numpy().max() - y_src.squeeze().cpu().detach().numpy().min())
    rmse_score = np.sqrt(np.mean((I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy())**2))
    metrics = {'ssim': float(ssim_score), 'rmse': float(rmse_score), 'dice': float(mean_dice_score)}

    print(f'SSIM: {ssim_score}')
    print(f'RMSE: {rmse_score}')
    print(f'Dice: {mean_dice_score}')

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # If there already exists a metrics.json, append to it with the next index of the execution
    # i.e
    # { idx: 0, ssim: 0.5, rmse: 0.5, dice: 0.5}
    # { idx: 1, ssim: 0.5, rmse: 0.5, dice: 0.5}
    # ...
    # { idx: n, ssim: 0.5, rmse: 0.5, dice: 0.5}

    metrics_file = os.path.join(output_path, 'metrics.json')
    if os.path.exists(metrics_file) and os.path.getsize(metrics_file) > 0:
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        idx = len(metrics_data)
    else:
        metrics_data = {}
        idx = 0

    metrics_data[idx] = metrics

    with open(os.path.join(output_path, 'metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)

   


# Main execution block
def main():
    args = parse_arguments()
    para, device = load_parameters()


    all_dataset = load_all_data()

    #all_dataset = (image, shape) 16 times


    #--save 2 images for testing

    train_dataset = [data[0] for data in all_dataset[2:]]
    train_shapes = [data[1] for data in all_dataset[2:]]
    test_dataset = [data[0] for data in all_dataset[:2]]
    test_shapes = [data[1] for data in all_dataset[:2]]

    print(f'Getting the tensor files...')
    train_tensor_dataset = get_tensor_dataset(train_dataset, device)
    test_tensor_dataset = get_tensor_dataset(test_dataset, device)
    train_tensor_shapes = get_tensor_dataset(train_shapes, device)
    test_tensor_shapes = get_tensor_dataset(test_shapes, device)
    print(f'Got the tensor files...')

    net, _, optimizer = initialize_network_optimizer2D(128, 128, para, device)
    phi_inv, _ = train_model(net, optimizer, args.pretrain, train_tensor_dataset, test_tensor_dataset, train_tensor_shapes, test_tensor_shapes,  10, 0.001, device)

    net_test_model(net, test_tensor_dataset, test_tensor_shapes, train_tensor_dataset, train_shapes, device, args)
 
    print('Done!')

if __name__ == '__main__':
    main()

