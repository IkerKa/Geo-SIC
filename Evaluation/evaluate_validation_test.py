
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


def exhaustive_train_model_with_validation(net, optimizer, num_epochs, train_dataset, val_dataset, val_segmentations, weight_dist, weight_reg, device, val_every=5):
    loss_total = 0
    phi_inv = None
    ssim_per_epoch, loss_per_epoch = [], []
    val_ssim_scores, val_rmse_scores, val_dice_scores = [], [], []

    tqdm_epochs = tqdm.tqdm(range(num_epochs))

    pairs = [(i, j) for i in range(len(train_dataset)) for j in range(len(train_dataset))]
    print(f'Training with {len(pairs)} pairs of images...')
    for epoch in tqdm_epochs:
        tqdm_epochs.set_description(f'Epoch {epoch + 1}/{num_epochs}')
        for i, j in pairs:
            # idx = random.randint(0, len(train_dataset) - 1)
            # I1, I2 = train_dataset[idx], train_dataset[(idx + 1) % len(train_dataset)]
            I1, I2 = train_dataset[i], train_dataset[j]
            # I1_seg, I2_seg = train_seg[i], train_seg[j]


            # Per each epoch, train with all possible pairs of images? can be that done? i think it would improve

            net.train()
            optimizer.zero_grad()



            I1, I2 = I1.to(device).float(), I2.to(device).float()
            y_src, momentum, _, new_locs = net(I1, I2, registration=True, shooting='SVF', return_phi=True)

            dist_loss = NCC(win=[9,9]).loss(y_src, I2)
            reg_loss = Grad(penalty='l2').loss2D(momentum)
            loss_total = weight_dist * dist_loss + weight_reg * reg_loss

            loss_total.backward()
            optimizer.step()

            loss_per_epoch.append(loss_total.item())
            ssim_per_epoch.append(ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(),
                                    data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min()))


            if epoch % val_every == 0:
                val_ssim, val_rmse, val_dice = validate_model(net, val_dataset, val_segmentations, device)
                val_ssim_scores.append(val_ssim)
                val_rmse_scores.append(val_rmse)
                val_dice_scores.append(val_dice)
                print(f'Validation - Epoch {epoch}: SSIM={val_ssim:.4f}, RMSE={val_rmse:.4f}, Dice={val_dice:.4f}')

        with torch.no_grad():
            phi_inv = new_locs[0,...]

    return phi_inv, y_src, loss_per_epoch, ssim_per_epoch

def train_model_with_validation(net, optimizer, num_epochs, train_dataset, val_dataset, val_segmentations, weight_dist, weight_reg, device, criterion=None, flag = 'SVF', val_every=5):
    loss_total = 0
    phi_inv = None
    ssim_per_epoch, loss_per_epoch = [], []
    val_ssim_scores, val_rmse_scores, val_dice_scores = [], [], []

    tqdm_epochs = tqdm.tqdm(range(num_epochs))

    pairs = [(i, j) for i in range(len(train_dataset)) for j in range(len(train_dataset))]
    print(f'Training with {len(pairs)} pairs of images...')
    for epoch in tqdm_epochs:

        net.train()

        phiinv = None
        optimizer.zero_grad()

        idx = random.randint(0, len(train_dataset) - 1)
        I1, I2 = train_dataset[idx], train_dataset[(idx + 1) % len(train_dataset)]

        b, c, w, h = I1.shape
        phiinv_bch = torch.zeros(b, w, h, 2).to(device)
        reg_save = torch.zeros(b, w, h, 2).to(device)


        # Per each epoch, train with all possible pairs of images? can be that done? i think it would improve
        tqdm_epochs.set_description(f'Epoch {epoch + 1}/{num_epochs}')
        net.train()
        optimizer.zero_grad()



        I1, I2 = I1.to(device).float(), I2.to(device).float()
        y_src, momentum, _, new_locs = net(I1, I2, registration=True, shooting=flag, return_phi=True)

        if flag == 'SVF':
            Dist = NCC().loss(y_src, I2)
            Reg = Grad(penalty='l2')
            Reg_loss = Reg.loss2D(momentum)

            loss_total = weight_dist * Dist + weight_reg * Reg_loss
            loss_total.backward(retain_graph=True)
        #TODO revisar el bloque de EPD
        else:
            momentum = momentum.permute(0, 3, 2, 1) # ? ARE THE SIZES CORRECT?

            #MATHS things
            img_size = 128
            identity = get_grid2D(img_size, device).permute([0, 3, 2, 1])
            epd = Epdiff2D(device, (16, 16), (128, 128), 5, 0.5, 2)
            # logger.divider("Math part")

            for b_id in range(b):
                v_fourier = epd.spatial2fourier(momentum[b_id,...].reshape(128, 128, 2))
                velocity = epd.fourier2spatial(epd.Kcoeff * v_fourier).reshape(128, 128, 2)
                reg_temp = epd.fourier2spatial(epd.Lcoeff * v_fourier * v_fourier)
                num_steps = 12
                v_seq, displacement = epd.forward_shooting_v_and_phiinv(velocity, num_steps)    # ! Bottleneck for complexity
                phiinv = displacement.unsqueeze(0) + identity
                phiinv_bch[b_id,...] = phiinv
                reg_save[b_id,...] = reg_temp

            dfm = Torchinterp2D(I1,phiinv_bch)
            Dist = criterion(dfm, I2)
            Reg_loss =  reg_save.sum()
            loss_total =  weight_dist * Dist + weight_reg * Reg_loss
            loss_total.backward(retain_graph=True)

        #Update the network parameters
        optimizer.step()

        # Update the loss and ssim values
        loss_per_epoch.append(loss_total.item())
        loss_total = 0.0
        ssim_per_epoch.append(ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(),
                                    data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min()))

        #Validation step
        if epoch % val_every == 0:
            val_ssim, val_rmse, val_dice = validate_model(net, val_dataset, val_segmentations, device)
            val_ssim_scores.append(val_ssim)
            val_rmse_scores.append(val_rmse)
            val_dice_scores.append(val_dice)
            print(f'Validation - Epoch {epoch}: SSIM={val_ssim:.4f}, RMSE={val_rmse:.4f}, Dice={val_dice:.4f}')

        with torch.no_grad():
            phi_inv = new_locs[0,...]

    return phi_inv, y_src, loss_per_epoch, ssim_per_epoch

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
def compute_segmentationFT(I1_seg, phi_inv, I2_seg, dev):


    # phi_inv = phi_inv.permute(2,0,1).unsqueeze(0)
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
def net_test_model(net, test_images, test_segs, flag, device):
    # with the trained model, test the images (phi inverted)

    # can be also use the net?
    pairs = [(i, j) for i in range(len(test_images)) for j in range(len(test_images))]
    print(f'Testing {len(pairs)} pairs of images...')

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
            y_src, _, _, new_locs = net(I1, I2, registration=True, shooting=flag, return_phi=True)
            _, _, dice_score = compute_segmentation(I1_seg, new_locs[0,...], I2_seg, device)

            #obtain the metrics and save them
            ssim_score = ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(), data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min())
            rmse_score = np.sqrt(np.mean((I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()) ** 2))
            mean_dice_score = np.mean(dice_score)

            print(f'Test - SSIM: {ssim_score:.4f}, RMSE: {rmse_score:.4f}, Dice: {mean_dice_score:.4f}')

            # save_metrics('output', I2, y_src, ssim_score, rmse_score, mean_dice_score)

def test_model(phi_inv, test_images, test_segs, device):
    # with the trained model, test the images (phi inverted)

    # can be also use the net?
    I1 = test_images[0]
    I2 = test_images[1]
    I1_seg = test_segs[0]
    I2_seg = test_segs[1]

    y_src = F.grid_sample(I1, phi_inv.unsqueeze(0), align_corners=True, mode='bilinear')
    warped_seg_np, fixed_seg_np, dice_score = compute_segmentation(I1_seg, phi_inv, I2_seg, device)

    #obtain the metrics and save them
    ssim_score = ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(), data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min())
    rmse_score = np.sqrt(np.mean((I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()) ** 2))
    mean_dice_score = np.mean(dice_score)

    print(f'Test - SSIM: {ssim_score:.4f}, RMSE: {rmse_score:.4f}, Dice: {mean_dice_score:.4f}')

    save_metrics('output', I2, y_src, ssim_score, rmse_score, mean_dice_score)

def fine_tune_deformation(net, test_dataset, test_segmentations, device, criterion, num_steps=100, lr=0.01):

    #after the inference we are going to fine tune the model with the test images
    #we are going to use the same pairs of images as in the inference
    pairs = [(i, j) for i in range(len(test_dataset)) for j in range(len(test_dataset))]
    print(f'Fine tunning {len(pairs)} pairs of images...')

    all_pairs_loss = []

    for i, j in pairs:
        I1 = test_dataset[i]
        I2 = test_dataset[j]
        I1_seg = test_segmentations[i]
        I2_seg = test_segmentations[j]

        I1 = I1.to(device).float()
        I2 = I2.to(device).float()
        # Set the network to training mode
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

        loss_per_step = []
        for step in range(num_steps):
            optimizer.zero_grad()
            # Forward pass through the network
            y_src, momentum, _, _ = net(I1, I2, registration=True, shooting='SVF', return_phi=True)

            # Compute the loss for the image deformation
            dist_loss = criterion(y_src, I2)
            reg_loss = Grad(penalty='l2').loss2D(momentum)
            loss_total = 10 * dist_loss + 0.001 * reg_loss

            # Backpropagation
            loss_total.backward()
            loss_per_step.append(loss_total.item())
            # Update the network parameters
            optimizer.step()

            # # Print the loss every 10 steps
            # if step % 10 == 0:
            #     print(f'Step {step}/{num_steps}, Loss: {loss_total.item():.4f}')


        #save current loss vector
        all_pairs_loss.append(([i, j], loss_per_step))

        

        #-Evaluate the fine-tuned model
        with torch.no_grad():
            net.eval()
            y_src, _, _, new_locs = net(I1, I2, registration=True, shooting='SVF', return_phi=True)
            _, _, dice_score = compute_segmentation(I1_seg, new_locs[0,...], I2_seg, device)

            #obtain the metrics and save them
            ssim_score = ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(), data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min())
            rmse_score = np.sqrt(np.mean((I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()) ** 2))
            mean_dice_score = dice_score

            #plot the results
            # fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            # axes[0].imshow(I1.squeeze().cpu().detach().numpy(), cmap='gray')
            # axes[0].set_title('Source Image (I1)')
            # axes[1].imshow(I2.squeeze().cpu().detach().numpy(), cmap='gray')
            # axes[1].set_title('Target Image (I2)')
            # axes[2].imshow(y_src.squeeze().cpu().detach().numpy(), cmap='gray')
            # axes[2].set_title('Warped Image (y_src)')

            # # Plotting the deformation grid for phi_inv
            # ax = axes[3]
            # interval = 2

            # for row in range(0, new_locs[0,...].shape[0], interval):
            #     ax.plot(new_locs[0,...][row, :, 0].cpu().detach().numpy(),
            #             new_locs[0,...][row, :, 1].cpu().detach().numpy(),
            #             'm')

            # for col in range(0, new_locs[0,...].shape[1], interval):
            #     ax.plot(new_locs[0,...][:, col, 0].cpu().detach().numpy(),
            #             new_locs[0,...][:, col, 1].cpu().detach().numpy(),
            #             'm')

            # plt.show()

            
            print(f'Fine tuning - SSIM: {ssim_score:.4f}, RMSE: {rmse_score:.4f}, Dice: {mean_dice_score:.4f}')


    #plot all the losses
    fig, ax = plt.subplots(figsize=(10, 5))
    for pair, loss in all_pairs_loss:
        ax.plot(loss, label=f'Pair {pair}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs Step')
    ax.legend()
    plt.show()

# Save metrics
def save_metrics(output_path, I2, y_src, ssim_score, rmse_score, mean_dice_score):
    metrics = {
        'ssim': float(ssim_score),
        'rmse': float(rmse_score),
        'dice': float(mean_dice_score)
    }


    # Ensure the output directory exists
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


def plot_results(loss, ssim, test_images, test_segs, phi_inv, device):
    #1st plot: loss vs epoch
    plt.figure(figsize=(10, 5))
    plt.plot(loss, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    # plt.savefig('output/loss_vs_epoch.png')
    plt.show()

    #2nd plot: ssim vs epoch
    plt.figure(figsize=(10, 5))
    plt.plot(ssim, label='SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM vs Epoch')
    plt.legend()

    # plt.savefig('output/ssim_vs_epoch.png')
    plt.show()

    #3rd plot: target vs warped vs phi_inv
    I1 = test_images[0]
    I2 = test_images[1]

    y_src = F.grid_sample(I1, phi_inv.unsqueeze(0), align_corners=True, mode='bilinear')
    I2 = I2.squeeze().cpu().detach().numpy()
    I1 = I1.squeeze().cpu().detach().numpy()
    y_src = y_src.squeeze().cpu().detach().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(I1, cmap='gray')
    axes[0].set_title('Source Image (I1)')
    axes[1].imshow(I2, cmap='gray')
    axes[1].set_title('Target Image (I2)')
    axes[2].imshow(y_src, cmap='gray')
    axes[2].set_title('Warped Image (y_src)')

    # Plotting the deformation grid for phi_inv
    ax = axes[3]
    interval = 2

    for row in range(0, phi_inv.shape[0], interval):
        ax.plot(phi_inv[row, :, 0].cpu().detach().numpy(),
                phi_inv[row, :, 1].cpu().detach().numpy(),
                'm')

    for col in range(0, phi_inv.shape[1], interval):
        ax.plot(phi_inv[:, col, 0].cpu().detach().numpy(),
                phi_inv[:, col, 1].cpu().detach().numpy(),
                'm')

    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(I1, cmap='gray')

    H, W = I1.shape

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




# Main execution block
def main():
    args = parse_arguments()
    para, device = load_parameters()

    _debug = False

    if _debug:
        circle_path = 'datasets/images/circle.png'
        input_image = load_debug_data(circle_path)
        #read image as target
        target_image = Image.open(circle_path).convert('L')
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        target_image = transform(target_image).unsqueeze(0).to(device)
        target_image = target_image.squeeze(0)
        #resize input image to 128x128
        input_image = transforms.Resize((128, 128))(input_image)

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


        #if there isnt a model directory, create it
        if not os.path.exists('models') and args.output != None:
            os.makedirs('models')

        # Load data
        all_dataset = load_all_data()

        shuffled_dataset = random.sample(all_dataset, len(all_dataset))

        num_total = len(shuffled_dataset)
        num_train = 12
        num_val = 2
        num_test = 2

        #--Dataset preparation so we have 3 different contexts
        # 1. No test images visible during training
        # 2. ONE test image visible during training
        # 3. HALF of the test images visible during training
        # 4. ALL test images visible during training

        context = 2  # Cambia a 2, 3 o 4 según el experimento

        if context == 1:
            train_dataset = shuffled_dataset[:num_train]
            val_dataset = shuffled_dataset[num_train:num_train + num_val]
            test_dataset = shuffled_dataset[num_train + num_val:]

        elif context == 2:
            train_dataset = shuffled_dataset[:num_train]
            val_dataset = shuffled_dataset[num_train:num_train + num_val]
            test_dataset = shuffled_dataset[num_train + num_val:]
            train_dataset.append(test_dataset[0])  # Adding one test image

        elif context == 3:
            train_dataset = shuffled_dataset[:num_train]
            val_dataset = shuffled_dataset[num_train:num_train + num_val]
            test_dataset = shuffled_dataset[num_train + num_val:]
            train_dataset.extend(test_dataset[:len(test_dataset)//2])  # Adding half of test images
        elif context == 4:
            train_dataset = shuffled_dataset[:num_train]
            val_dataset = shuffled_dataset[num_train:num_train + num_val]
            test_dataset = shuffled_dataset[num_train + num_val:]
            train_dataset.extend(test_dataset)  # Adding all test images

        print(f'Context {context}')
        print(f'Total dataset size: {num_total}')
        print(f'Train dataset size: {len(train_dataset)}')
        print(f'Validation dataset size: {len(val_dataset)}')
        print(f'Test dataset size: {len(test_dataset)}')

        print(f'Train dataset size: {len(train_dataset)}')
        print(f'Validation dataset size: {len(val_dataset)}')
        print(f'Test dataset size: {len(test_dataset)}')



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

        net, criterion, optimizer = initialize_network_optimizer2D(128, 128, para, device)

        shooting_flag = 'LDDMM'

        time_init = time.time()
        # ef exhaustive_train_model_with_validation(net, optimizer, num_epochs, train_dataset, val_dataset, val_segmentations, weight_dist, weight_reg, device, val_every=5)
        # phi_inv, _, loss, ssim = exhaustive_train_model_with_validation(net, optimizer, args.pretrain, train_images, val_images, val_seg, 10, 0.001, device, val_every=20) #, flag = 'SVF', val_every=20)
        phi_inv, _, loss, ssim = train_model_with_validation(net, optimizer, args.pretrain, train_images, val_images, val_seg, 10, 0.001, device, criterion, shooting_flag, val_every=20)
        plot_results(loss, ssim, test_images, test_seg, phi_inv, device)
        torch.save(net.state_dict(), os.path.join(args.output, 'model_trained.pth'))
        time_end = time.time()
        print(f'Training time: {time_end - time_init} seconds')

        #load the model in eval mode
        net.load_state_dict(torch.load(os.path.join(args.output, 'model_trained.pth')))

        net_test_model(net, test_images, test_seg, shooting_flag,device)

        if shooting_flag == 'SVF':
            fine_tune_deformation(net, test_images, test_seg, device, criterion, num_steps=80, lr=0.01)


        

        



if __name__ == '__main__':
    main()

