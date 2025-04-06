
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
import torchvision.transforms as T
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(parent_dir)

from losses import NCC, MSE, Grad, DiceLoss
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

def load_all_data(nifti_datadir='nirep/nifti/', size=128, slice_index=135,view=2):
    datahandler = dh(dataset_type='nifti', directory=nifti_datadir, size=size, slice_index=slice_index, seg=True, view=view)
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

#calculate DICE loss
def dice_loss(I1_seg, I2_seg, phi_inv, device):
    phi_inv = phi_inv.permute(2,0,1).unsqueeze(0)
    I1_seg_warped = F.grid_sample(I1_seg.float(), phi_inv, mode='nearest', align_corners=True)
    dice_loss_fn = DiceLoss()
    seg_loss = dice_loss_fn(I1_seg_warped, I2_seg)

    return seg_loss

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

def exhaustive_train_model_with_validation(net, optimizer, num_epochs, train_dataset, train_segmentations, weight_dist, weight_reg, device):
    loss_total = 0
    phi_inv = None
    ssim_per_epoch, loss_per_epoch, val_dice_scores = [], [], []

    pairs = [(i, j) for i in range(len(train_dataset)) for j in range(i + 1, len(train_dataset))]

    batch_size = 2
    acc_loss = 0
    for epoch in range(num_epochs):
        print(f'\rEpoch {epoch + 1}/{num_epochs}', end='', flush=True)

        net.train()
        pair_loss = []
        pair_ssim = []
        pair_dice = []

        for pair_idx, (i, j) in enumerate(pairs):
            I1 = train_dataset[i]
            I2 = train_dataset[j]
            I1_seg = train_segmentations[i]
            I2_seg = train_segmentations[j]

            I1, I2 = I1.to(device).float(), I2.to(device).float()
            y_src, y_tgt, momentum, _, _, new_locs, _ = net(I1, I2, registration=True, shooting="SVF", return_phi=True)
            _, _, dice_score = compute_segmentation(I1_seg, new_locs[0, ...], I2_seg, device)

            Dist = (NCC().loss(I2, y_src) + NCC().loss(I1, y_tgt))
            Reg = Grad(penalty='l2')
            Reg_loss = Reg.loss2D(momentum)

            loss_total = (weight_dist * Dist + weight_reg * Reg_loss)
            acc_loss += loss_total

            if (pair_idx + 1) % batch_size == 0 or pair_idx == len(pairs) - 1:
                avg_loss = acc_loss / batch_size
                avg_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                acc_loss = 0

            pair_loss.append(loss_total.item())
            pair_ssim.append(ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(),
                                  data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min()))
            pair_dice.append(dice_score)

            with torch.no_grad():
                phi_inv = new_locs[0, ...]

        loss_per_epoch.append(np.mean(pair_loss))
        ssim_per_epoch.append(np.mean(pair_ssim))
        val_dice_scores.append(np.mean(pair_dice))

    print()  # To move to the next line after the last epoch
    return phi_inv, y_src, loss_per_epoch, ssim_per_epoch


def compute_segmentation(I1_seg, phi_inv, I2_seg, dev):

    phi_inv = phi_inv.permute(2,0,1).unsqueeze(0)
    spat_trans = SpatialTransformer(size=I1_seg.shape[2:], mode='nearest').to(dev)
    warped_seg = spat_trans(I1_seg, phi_inv)
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

    phiinvs = []
    y_srcs = []

    ssims = []
    rmse = []
    dices = []

    with torch.no_grad():
        net.eval()
        for i, j in pairs:
            I1 = test_images[i]
            I2 = test_images[j]
            I1_seg = test_segs[i]
            I2_seg = test_segs[j]

            I1 = I1.to(device).float()
            I2 = I2.to(device).float()
            net.eval()  # Modo evaluación
            y_src, _, _, _, _, new_locs, _ = net(I1, I2, registration=True, shooting=flag, return_phi=True)
            _, _, dice_score = compute_segmentation(I1_seg, new_locs[0,...], I2_seg, device)

            phiinvs.append(new_locs[0,...])
            y_srcs.append(y_src)



            #obtain the metrics and save them
            ssim_score = ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(), data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min())
            rmse_score = np.sqrt(np.mean((I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()) ** 2))
            mean_dice_score = np.mean(dice_score)

            ssims.append(ssim_score)
            rmse.append(rmse_score)
            dices.append(mean_dice_score)

            # print(f'Test - SSIM: {ssim_score:.4f}, RMSE: {rmse_score:.4f}, Dice: {mean_dice_score:.4f}')

            # save_metrics('output', I2, y_src, ssim_score, rmse_score, mean_dice_score)

        #print the average
        # print(f'Average from run - SSIM: {np.mean(ssims):.4f}, RMSE: {np.mean(rmse):.4f}, Dice: {np.mean(dices):.4f}')

    return np.mean(ssims), np.mean(rmse), np.mean(dices)




def plot_results(test_images, test_shapes, phiinvs, y_srcs):
    
    #for every possible pair
    pairs = [(i, j) for i in range(len(test_images)) for j in range(len(test_images)) if i != j]
    shapes_pairs = [(i, j) for i in range(len(test_shapes)) for j in range(len(test_shapes)) if i != j]

    for pair_idx, (pair) in enumerate(pairs):
        #if the I1 == I2, skip it
        if pair[0] == pair[1]:
            continue

        I1 = test_images[pair[0]]
        I2 = test_images[pair[1]]
        I1_seg = test_shapes[shapes_pairs[pair_idx][0]]
        I2_seg = test_shapes[shapes_pairs[pair_idx][1]]

        phi_inv = phiinvs[pair_idx]
        y_src = y_srcs[pair_idx]

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(I1.squeeze().cpu().detach().numpy(), cmap='gray')
        axes[0].set_title('Source Image (I1)')
        axes[1].imshow(I2.squeeze().cpu().detach().numpy(), cmap='gray')
        axes[1].set_title('Target Image (I2)')
        axes[2].imshow(y_src.squeeze().cpu().detach().numpy(), cmap='gray')
        axes[2].set_title('Registered Image (y_src)')
        error = I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()
        axes[3].imshow(error, cmap='gray')
        axes[3].set_title('Error (I2 - y_src)')

        # Plotting the deformation grid for phi_inv
        ax = axes[4]
        interval = 2

        for row in range(0, phi_inv.shape[0], interval):
            ax.plot(phi_inv[row, :, 0].cpu().detach().numpy(),
                phi_inv[row, :, 1].cpu().detach().numpy(),
                'm')

        for col in range(0, phi_inv.shape[1], interval):
            ax.plot(phi_inv[:, col, 0].cpu().detach().numpy(),
                phi_inv[:, col, 1].cpu().detach().numpy(),
                'm')

        ax.set_title("Diffeomorphic deformation grid")
        plt.tight_layout()
        plt.show()




# Main execution block
def main():
    args = parse_arguments()
    para, device = load_parameters()



    #if there isnt an output directory, create it
    if not os.path.exists(args.output) and args.output != None:
        os.makedirs(args.output)


    #if there isnt a model directory, create it
    if not os.path.exists('models') and args.output != None:
        os.makedirs('models')

    # Load data
    all_dataset = load_all_data()

    # shuffled_dataset = random.sample(all_dataset, len(all_dataset))
    shuffled_dataset = all_dataset.copy()

    num_total = len(shuffled_dataset)
    num_train = int(num_total * 0.8)  # 80% for training
    num_val = num_total - num_train  # Remaining for validation

    # Ensure no leftover by checking the sum matches the total
    assert num_train + num_val == num_total, "The split sizes do not match the total dataset size."


    #atm we exclude the test images from the training set

    #--Dataset preparation so we have 3 different contexts
    # 1. No test images visible during training
    # 2. ONE test image visible during training
    # 3. HALF of the test images visible during training
    # 4. ALL test images visible during training

    context = 1  # Cambia a 2, 3 o 4 según el experimento

    if context == 1:
        train_dataset = shuffled_dataset[:num_train]
        val_dataset = shuffled_dataset[num_train:num_train + num_val]
        test_dataset = shuffled_dataset[num_train + num_val:]   #will be 0

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

        

    print("---" * 20)
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')
    print("---" * 20)


    train_images = [data[0] for data in train_dataset]
    train_seg = [data[1] for data in train_dataset]

    val_images = [data[0] for data in val_dataset]
    val_seg = [data[1] for data in val_dataset]

    test_images = [data[0] for data in test_dataset]
    test_seg = [data[1] for data in test_dataset]

    # Convert to tensors
    train_images = get_tensor_dataset(train_images, device)
    train_seg = get_tensor_dataset(train_seg, device)
    #--
    val_images = get_tensor_dataset(val_images, device)
    val_seg = get_tensor_dataset(val_seg, device)
    #--
    test_images = get_tensor_dataset(test_images, device)
    test_seg = get_tensor_dataset(test_seg, device)

    #plot an image to see if it works
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(train_images[0].squeeze().cpu().detach().numpy(), cmap='gray')
    axes[0].set_title('Train Image (I1)')
    axes[1].imshow(train_seg[0].squeeze().cpu().detach().numpy(), cmap='gray')
    axes[1].set_title('Train Segmentation (I1)')
    plt.show()


    net, _, optimizer = initialize_network_optimizer2D(128, 128, para, device)

    shooting_flag = 'SVF'
    # weight_regs = [0.001, 0.01, 0.05, 0.1, 0.5, 1]
    weight_regs = [0.001, 0.005, 0.01, 0.05]
    results = []
    for weight_reg in weight_regs:
        print(f"Training with weight_reg: {weight_reg}")
        # Train the model with validation
        phi_inv, y_src, loss, ssim = exhaustive_train_model_with_validation(net, optimizer, args.pretrain, train_images, train_seg, 1, weight_reg, device)
        ssim, rmse, dices = net_test_model(net, val_images, val_seg, shooting_flag, device)
        results.append({
            'weight_reg': weight_reg,
            'ssim': ssim,
            'rmse': rmse,
            'dice': dices
        })

    
    print(f'Ended executions with every weight_reg')
    #print the results
    for result in results:
        print(f"Weight reg: {result['weight_reg']}, SSIM: {result['ssim']}, RMSE: {result['rmse']}, DICE: {result['dice']}")


if __name__ == '__main__':
    main()
