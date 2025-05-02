
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
from mpl_toolkits.mplot3d import Axes3D
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

from Run_Atlas_trainer import initialize_network_optimizer2D, initialize_network_optimizer, read_yaml


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
    datahandler = dh(dataset_type='nifti3d', directory=nifti_datadir, size=size, seg=True)
    return datahandler.get_image(src_index), datahandler.get_image(tgt_index)

def load_all_data(nifti_datadir='nirep/nifti/', size=128, slice_index=149):
    datahandler = dh(dataset_type='nifti3d', directory=nifti_datadir, size=size, seg=True)
    return datahandler.get_all_images()

def get_tensor_dataset(dataset, device):
    return [convert_to_tensor(image, device) for image in dataset]

# Convert images to tensors
def convert_to_tensor(image, device):
    return torch.tensor(image, dtype=torch.float32).to(device).unsqueeze(0)

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

def validate_model(net, val_images, val_segs, device):
    net.eval()  # Modo evaluación
    ssim_scores, rmse_scores, dice_scores = [], [], []

    with torch.no_grad():
        net.eval()
        #all possible pairs of images (val images)
        pairs = [(i, j) for i in range(len(val_images)) for j in range(len(val_images))]
        print(f'Validating {len(pairs)} pairs of images...')
        #for-loop for every pair of images
        for i, j in pairs:
            I1 = val_images[i]
            I2 = val_images[j]
            I1_seg = val_segs[i]
            I2_seg = val_segs[j]

            I1 = I1.to(device).float()
            I2 = I2.to(device).float()
            y_src, _, _, new_locs = net(I1, I2, registration=True, shooting='SVF', return_phi=True)

            ssim_score = ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(), data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min())
            rmse_score = np.sqrt(np.mean((I2.squeeze().cpu().numpy() - y_src.squeeze().cpu().numpy()) ** 2))

            _, _, dice_score = compute_segmentation(I1_seg, new_locs[0,...], I2_seg, device)

            ssim_scores.append(ssim_score)
            rmse_scores.append(rmse_score)
            dice_scores.append(dice_score)

    return np.mean(ssim_scores), np.mean(rmse_scores), np.mean(dice_scores)


def train_model_with_validation(net, optimizer, num_epochs, train_dataset, train_seg, weight_dist, weight_reg, device, criterion=None, flag = 'SVF', val_every=5):
    loss_total = 0
    phi_inv = None
    ssim_per_epoch, loss_per_epoch, dice_per_epoch = [], [], []


    # #Combine train and val datasets
    # train_dataset = train_dataset + val_dataset
    pairs = [(i, j) for i in range(len(train_dataset)) for j in range(len(train_dataset))] 
    with tqdm.tqdm(total=num_epochs, desc='Training', unit='epoch', leave=True) as tqdm_epochs:
        net.train()
        for epoch in range(num_epochs):
            tqdm_epochs.update(1)
   
            optimizer.zero_grad()
            phiinv = None
            # Get two random indices
            idx, idx2 = random.sample(range(len(train_dataset)), 2)
            tqdm_epochs.set_description(f'Training epoch {epoch} with pair {idx} and {idx2}')

            ## TAKE THE IMAGES AND SEGMENTATIONS ##
            I1, I2 = train_dataset[idx], train_dataset[(idx + 1) % len(train_dataset)]
            I1_seg, I2_seg = train_seg[idx], train_seg[(idx + 1) % len(train_seg)]
            I1, I2 = I1.to(device).float(), I2.to(device).float()

            
            y_src, momentum, _, new_locs = net(I1, I2, registration=True, shooting=flag, return_phi=True)
            _,_, dice_score = compute_segmentation(I1_seg, new_locs[0,...], I2_seg, device)

            if flag == 'SVF':
                Dist = NCC().loss(I2, y_src)  # Compute the NCC loss
                Reg = Grad(penalty='l2')
                Reg_loss = Reg.loss(momentum)

                loss_total = weight_dist * Dist + weight_reg * Reg_loss
                loss_total.backward()
            
            #Update the network parameters
            optimizer.step()

            # Update the loss and ssim values
            loss_per_epoch.append(loss_total.item())
            loss_total = 0.0
            ssim_per_epoch.append(ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(),
                                        data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min()))
            
            dice_per_epoch.append(dice_score)

            tqdm_epochs.set_postfix(loss=loss_per_epoch[-1], ssim=ssim_per_epoch[-1], dice=dice_score)

            with torch.no_grad():
                phi_inv = new_locs[0,...]

     #plot the metrics
    # Plot Loss
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_per_epoch, label='Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot SSIM
    plt.figure(figsize=(10, 5))
    plt.plot(ssim_per_epoch, label='SSIM', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.title('Training SSIM Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Dice
    plt.figure(figsize=(10, 5))
    plt.plot(dice_per_epoch, label='Dice', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.title('Training Dice Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    return phi_inv, y_src, loss_per_epoch, ssim_per_epoch


def compute_segmentation(I1_seg, phi_inv, I2_seg, dev):

    phi_inv = phi_inv.permute(3, 0, 1, 2).unsqueeze(0)  # Adjust for 3D volumes
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



    return warped_seg_np, fixed_seg_np, dice
def net_test_model(net, test_images, test_segs, flag, device):
    # with the trained model, test the images (phi inverted)

    # can be also use the net?
    pairs = [(i, j) for i in range(len(test_images)) for j in range(len(test_images)) if i != j]
    print(f'Testing {len(pairs)} pairs of images...')

    #for-loop for every pair of images
    rmses, ssims, dices = [], [], []
    with torch.no_grad():
        net.eval()
        for i, j in pairs:
            I1 = test_images[i]
            I2 = test_images[j]
            I1_seg = test_segs[i]
            I2_seg = test_segs[j]

            I1 = I1.to(device).float()
            I2 = I2.to(device).float()
            y_src, _, _, new_locs = net(I1, I2, registration=True, shooting=flag, return_phi=True)
            _, _, dice_score = compute_segmentation(I1_seg, new_locs[0,...], I2_seg, device)

            #obtain the metrics and save them
            ssim_score = ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(), data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min())
            rmse_score = np.sqrt(np.mean((I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()) ** 2))
            mean_dice_score = np.mean(dice_score)

            # Append the scores to the lists
            ssims.append(ssim_score)
            rmses.append(rmse_score)
            dices.append(mean_dice_score)

            print(f'Test with pair {i} and {j} - SSIM: {ssim_score:.4f}, RMSE: {rmse_score:.4f}, Dice: {mean_dice_score:.4f}')

             ### PLOT THE RESULTS ###
            slice_idx = 149

            #PRO TIP! For sagital views: [slice_idx, :, :], for coronal views: [:, slice_idx, :], for axial views: [:, :, slice_idx]

            I1_slice = I1.squeeze().cpu().detach().numpy()[:, :, slice_idx]  # (H, W)
            I2_slice = I2.squeeze().cpu().detach().numpy()[:, :, slice_idx]  # (H, W)
            y_src_slice = y_src.squeeze().cpu().detach().numpy()[:, :, slice_idx]  # (H, W)

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(I1_slice, cmap='gray')
            ax[0].set_title('Source Image')
            ax[0].axis('off')
            ax[1].imshow(I2_slice, cmap='gray')
            ax[1].set_title('Target Image')
            ax[1].axis('off')
            ax[2].imshow(y_src_slice, cmap='gray')
            ax[2].set_title('Warped Image')
            ax[2].axis('off')
            plt.show()


            # save_metrics('output', I2, y_src, ssim_score, rmse_score, mean_dice_score)

    # Calculate the average scores
    avg_ssim = np.mean(ssims)
    avg_rmse = np.mean(rmses)
    avg_dice = np.mean(dices)

    print(f'Average SSIM: {avg_ssim:.4f}, Average RMSE: {avg_rmse:.4f}, Average Dice: {avg_dice:.4f}')

def main():
    args = parse_arguments()
    para, device = load_parameters()


    ## ONLY 2 IMAGES ##
    # target_index = 1
    # source_index = 2

    # (input_image, input_segmentation), (target_image, target_segmentation) = load_data(tgt_index=target_index, src_index=source_index)

    # input_image = (input_image - torch.min(input_image)) / (torch.max(input_image) - torch.min(input_image))
    # target_image = (target_image - torch.min(target_image)) / (torch.max(target_image) - torch.min(target_image))
    # input_segmentation = (input_segmentation - torch.min(input_segmentation)) / (torch.max(input_segmentation) - torch.min(input_segmentation))
    # target_segmentation = (target_segmentation - torch.min(target_segmentation)) / (torch.max(target_segmentation) - torch.min(target_segmentation))

    # I1 = convert_to_tensor(input_image, device)
    # I2 = convert_to_tensor(target_image, device)
    # I1_seg = convert_to_tensor(input_segmentation, device)
    # I2_seg = convert_to_tensor(target_segmentation, device)

    ### LOAD ALL IMAGES ###
    all_dataset = load_all_data()
    shuffled_dataset = all_dataset.copy()

    # We have 256 silces, we have volume of 128x128x256, we take the first 128 slices
    # Shape form: [B, C, H, W, D] # B=1, C=1, H=128, W=128, D=256
    num_total_images = len(all_dataset)
    num_train = int(num_total_images * 0.8)
    num_val = num_total_images - num_train

    assert num_train + num_val == num_total_images, "Train and test split does not match total images"


    ### PREPARE DATASET ###
    context = 1 

    if context == 1:
        train_dataset = shuffled_dataset[:num_train]
        val_dataset = shuffled_dataset[num_train:num_train + num_val]
        test_dataset = shuffled_dataset[num_train + num_val:]             #0!

    elif context == 2:
        train_dataset = shuffled_dataset[:num_train]
        val_dataset = shuffled_dataset[num_train:num_train + num_val]
        #add one image and segmentation pair to the train dataset
        train_dataset.append(val_dataset[0])
        #test is empty
        test_dataset = shuffled_dataset[num_train + num_val:]

    elif context == 3:
        #All of the test images are visible during training
        train_dataset = shuffled_dataset[:num_train]
        val_dataset = shuffled_dataset[num_train:num_train + num_val]
        #add all images and segmentations to the train dataset
        train_dataset.extend(val_dataset)
        #test is empty
        test_dataset = shuffled_dataset[num_train + num_val:]



    ### CONVERT TO TENSORS ###
    train_images = [data[0] for data in train_dataset]
    train_segmentations = [data[1] for data in train_dataset]
    train_images_tensor = get_tensor_dataset(train_images, device)
    train_segmentations_tensor = get_tensor_dataset(train_segmentations, device)
    #----------------------------------------------------------
    val_images = [data[0] for data in val_dataset]
    val_segmentations = [data[1] for data in val_dataset]
    val_images_tensor = get_tensor_dataset(val_images, device)
    val_segmentations_tensor = get_tensor_dataset(val_segmentations, device)
    #----------------------------------------------------------
    test_images = [data[0] for data in test_dataset]
    test_segmentations = [data[1] for data in test_dataset]
    test_images_tensor = get_tensor_dataset(test_images, device)
    test_segmentations_tensor = get_tensor_dataset(test_segmentations, device)
    #----------------------------------------------------------

            

    net, criterion, optimizer = initialize_network_optimizer(128, 128, 256, para, device)


    time_init = time.time()
    ### TRAINING (ONLY PAIRWISE) ###
    phi_inv, y_src, _, _ = train_model_with_validation(net, optimizer, args.pretrain, train_images_tensor, train_segmentations_tensor, 1, 0.001, device, criterion, 'SVF', val_every=10)
    
    torch.save(net.state_dict(), os.path.join(args.output, 'model_trained.pth'))
    time_end = time.time()
    print(f'Training time: {time_end - time_init} seconds')
    net_test_model(net, val_images_tensor, val_segmentations_tensor, 'SVF', device)
    
   

if __name__ == '__main__':
    main()




    
