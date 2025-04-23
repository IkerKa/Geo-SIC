
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

def load_all_data(nifti_datadir='nirep/nifti/', size=128, slice_index=90, view=2):
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

def exhaustive_train_model_with_validation(net, optimizer, num_epochs, train_dataset, train_segmentations, device, flag = 'SVF'):
    loss_total = 0
    phi_inv = None
    ssim_per_epoch, loss_per_epoch, val_dice_scores, val_dice_medians = [], [], [], []

    # #Combine train and val datasets
    # train_dataset = train_dataset + val_dataset
    # pairs = [(i, j) for i in range(len(train_dataset)) for j in range(len(train_dataset))]
    # pairs = [(i, j) for i in range(len(train_dataset)) for j in range(i + 1, len(train_dataset))]
    # print(f'Training with {len(pairs)} pairs of images...')

    batch_size = 4
    acc_loss = 0
    
    plot_phis = []

    tqdm_epochs = tqdm.tqdm(total=num_epochs, desc="Training Progress", leave=False)
    for epoch in range(num_epochs):
        tqdm_epochs.update(1)

        net.train()
        
        # Per each epoch, train with all possible pairs of images? can be that done? i think it would improve
        tqdm_epochs.set_description(f'Epoch {epoch + 1}/{num_epochs}')

        pair_loss = []
        pair_ssim = []
        pair_dice = []
        pair_dice_median = []

        pairs = [(i, j) for i in range(len(train_dataset)) for j in range(i + 1, len(train_dataset))]
        pairs = random.sample(pairs, k=min(len(pairs), 100))  # Limit to 100 pairs for training

        for pair_idx, (i,j) in enumerate(pairs):

            I1 = train_dataset[i]
            I2 = train_dataset[j]

            I1_seg = train_segmentations[i]
            I2_seg = train_segmentations[j]

            b, c, w, h = I1.shape
            phiinv_bch = torch.zeros(b, w, h, 2).to(device)
            reg_save = torch.zeros(b, w, h, 2).to(device)

            I1, I2 = I1.to(device).float(), I2.to(device).float()
            y_src, momentum,  _, new_locs = net(I1, I2, registration=True, shooting="SVF", return_phi=True)
            warped_img, _, dice_mean, dice_median = compute_segmentation(I1_seg, new_locs[0, ...], I2_seg, device)
            # momentum = momentum.permute(0, 3, 1, 2)  # Permute to [batch, 2, height, width]
            # momentum_neg = momentum_neg.permute(0, 3, 1, 2)  # Permute to [batch, 2, height, width]


            if flag == 'SVF':
                Dist = (NCC().loss(I2, y_src)) # + NCC().loss(y_tgt, I1))
                # Dist = MSE().loss(I2, y_src)
                Reg = Grad(penalty='l2')
                Reg_loss = Reg.loss2D(momentum)
                # Dice_loss = DiceLoss(device=device)

                loss_total = (1 * Dist + 0.1 * Reg_loss)
                acc_loss += loss_total

            if (pair_idx + 1) % batch_size == 0 or pair_idx == len(pairs) - 1:
                avg_loss = acc_loss / batch_size
                avg_loss.backward()
                optimizer.step()      # Update model parameters
                optimizer.zero_grad() # Reset gradients
                acc_loss = 0

            # Update the loss and ssim values
            pair_loss.append(loss_total.item())
            pair_ssim.append(ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(),
                                    data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min()))
            pair_dice.append(dice_mean)
            pair_dice_median.append(dice_median)

            with torch.no_grad():
                phi_inv = new_locs[0,...]

        
        if epoch % 10 == 0:
            plot_phis.append(phi_inv.cpu().detach().numpy())
            # plt.imshow(phi_inv[:, :, 0].cpu().detach().numpy(), cmap='jet')
            # plt.title('Deformation Field (X)')
            # plt.show()
            
        
        mean_loss = np.mean(pair_loss)
        mean_ssim = np.mean(pair_ssim)
        mean_dice = np.mean(pair_dice)
        mean_dice_median = np.mean(pair_dice_median)
        loss_per_epoch.append(mean_loss)
        ssim_per_epoch.append(mean_ssim)
        val_dice_scores.append(mean_dice)
        val_dice_medians.append(mean_dice_median)


        tqdm_epochs.set_postfix(loss=mean_loss, ssim=mean_ssim, dice=mean_dice, dice_median=mean_dice_median)




    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot Loss
    axs[0, 0].plot(loss_per_epoch, label='Loss', color='blue')
    axs[0, 0].set_title('Loss over epochs')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Plot SSIM
    axs[0, 1].plot(ssim_per_epoch, label='SSIM', color='green')
    axs[0, 1].set_title('SSIM over epochs')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('SSIM')
    axs[0, 1].legend()

    # Plot Dice Scores
    axs[1, 0].plot(val_dice_scores, label='Dice Mean', color='red')
    axs[1, 0].set_title('Dice Mean over epochs')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Dice Mean')
    axs[1, 0].legend()

    # Plot Dice Medians
    axs[1, 1].plot(val_dice_medians, label='Dice Median', color='purple')
    axs[1, 1].set_title('Dice Median over epochs')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Dice Median')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # #animation of the phi_inv
    # fig, ax = plt.subplots(figsize=(10, 10))
    # for i in range(len(plot_phis)):
    #     ax.clear()
    #     ax.imshow(plot_phis[i][:, :, 0], cmap='jet')
    #     ax.set_title('Deformation Field (X)')
    #     plt.pause(0.1)

    #     plt.show()

        


    return phi_inv, y_src, loss_per_epoch, ssim_per_epoch

def compute_segmentation(I1_seg, phi_inv, I2_seg, dev):
    phi_inv = phi_inv.permute(2, 0, 1).unsqueeze(0)
    spat_trans = SpatialTransformer(size=I1_seg.shape[2:], mode='nearest').to(dev)
    warped_seg = spat_trans(I1_seg, phi_inv)

    warped_seg_np = warped_seg.squeeze().cpu().detach().numpy()
    fixed_seg_np = I2_seg.squeeze().cpu().detach().numpy()

    labels = np.unique(fixed_seg_np)
    labels = labels[labels != 0]

    dice_scores = compute_dice(warped_seg_np, fixed_seg_np, labels)

    # Filtrar valores NaN y ceros
    filtered_scores = [d for d in dice_scores if not np.isnan(d) and d > 0]

    if len(filtered_scores) > 0:
        dice_mean = np.mean(filtered_scores)
        dice_median = np.median(filtered_scores)
    else:
        dice_mean = 0.0
        dice_median = 0.0

    return warped_seg_np, fixed_seg_np, dice_mean, dice_median


def net_test_model(net, test_images, test_segs, flag, device):
    pairs = [(i, j) for i in range(len(test_images)) for j in range(len(test_images)) if i != j]
    print(f'Testing {len(pairs)} pairs of images...')

    ssims = []
    rmses = []
    dice_means = []
    dice_medians = []
    phis = []

    with torch.no_grad():
        net.eval()
        for i, j in pairs:
            I1 = test_images[i].to(device).float()
            I2 = test_images[j].to(device).float()
            I1_seg = test_segs[i]
            I2_seg = test_segs[j]

            y_src, _, _, new_locs = net(I1, I2, registration=True, shooting=flag, return_phi=True)
            phis.append(new_locs[0,...])
            _, _, dice_mean, dice_median = compute_segmentation(I1_seg, new_locs[0, ...], I2_seg, device)

            ssim_score = ssim(
                y_src.squeeze().cpu().detach().numpy(),
                I2.squeeze().cpu().detach().numpy(),
                data_range=I2.max().item() - I2.min().item()
            )
            rmse_score = np.sqrt(np.mean((I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()) ** 2))

            ssims.append(ssim_score)
            rmses.append(rmse_score)
            dice_means.append(dice_mean)
            dice_medians.append(dice_median)

            print(f'Pair {i},{j} - SSIM: {ssim_score:.4f}, RMSE: {rmse_score:.4f}, Dice Mean: {dice_mean:.4f}, Dice Median: {dice_median:.4f}')

        print(f'\nAverage - SSIM: {np.mean(ssims):.4f}, RMSE: {np.mean(rmses):.4f}, Dice Mean: {np.mean(dice_means):.4f}, Dice Median: {np.median(dice_medians):.4f}')

        #Plot some results
        for i, j in pairs:
            I1 = test_images[i].to(device).float()
            I2 = test_images[j].to(device).float()
            I1_seg = test_segs[i]
            I2_seg = test_segs[j]

            phi_inv = phis[i]

            y_src, _, _, new_locs = net(I1, I2, registration=True, shooting=flag, return_phi=True)
            warped_img, _, dice_mean, dice_median = compute_segmentation(I1_seg, new_locs[0, ...], I2_seg, device)

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

# Utility function to save NIfTI files
def save_nifti(image_tensor, file_path):
    """
    Save a PyTorch tensor as a NIfTI file.
    """

    #if it is already in numpy format, skip this step
    if isinstance(image_tensor, np.ndarray):
        image_np = image_tensor
    else:
        image_np = image_tensor.squeeze().cpu().detach().numpy()

    nifti_image = nib.Nifti1Image(image_np, affine=np.eye(4))
    nib.save(nifti_image, file_path)

def load_nifti(file_path):
    """
    Load a NIfTI file as a PyTorch tensor.
    """
    nifti_image = nib.load(file_path)
    image_np = nifti_image.get_fdata()
    image_tensor = torch.tensor(image_np, dtype=torch.float32)
    return image_tensor.unsqueeze(0)  # Add batch dimension

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
        all_dataset = load_all_data(slice_index=128, view=3)

        # augment = T.Compose([
        #             T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),
        # ])

        # # Extend the dataset with augmented data
        # augmented_dataset = []

        # # Apply the augmentations to the images and segmentations
        # for i in range(len(all_dataset)):
        #     print(f'Augmenting image {i}')
        #     original_image, original_segmentation = all_dataset[i]
            
        #     # Apply augmentation once to generate a slightly modified sample
        #     augmented_image = augment(original_image)
        #     augmented_segmentation = augment(original_segmentation)
        #     augmented_dataset.append((augmented_image, augmented_segmentation))

            

        # # Combine the original dataset with the augmented dataset
        # all_dataset.extend(augmented_dataset)

        # print(f'Original dataset size: {len(all_dataset) - len(augmented_dataset)}')
        # print(f'Augmented dataset size: {len(augmented_dataset)}')

        # Dejar dos imagenes de test fijas al igual que slice_idx.

        # shuffled_dataset = random.sample(all_dataset, len(all_dataset))
        shuffled_dataset = all_dataset.copy()
        num_total = len(shuffled_dataset)
        num_train = 14
        num_val = 2  # Remaining for validation
        # num_total = len(shuffled_dataset)
        # num_train = int(num_total * 0.8)  # 80% for training
        # num_val = num_total - num_train  # Remaining for validation

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


        Dtrain_images = [data[0] for data in train_dataset]
        Dtrain_seg = [data[1] for data in train_dataset]

        Dval_images = [data[0] for data in val_dataset]
        Dval_seg = [data[1] for data in val_dataset]

        Dtest_images = [data[0] for data in test_dataset]
        Dtest_seg = [data[1] for data in test_dataset]

        # Convert to tensors
        train_images = get_tensor_dataset(Dtrain_images, device)
        train_seg = get_tensor_dataset(Dtrain_seg, device)
        #--
        val_images = get_tensor_dataset(Dval_images, device)
        val_seg = get_tensor_dataset(Dval_seg, device)
        #--
        test_images = get_tensor_dataset(Dtest_images, device)
        test_seg = get_tensor_dataset(Dtest_seg, device)
        

        #Plot the test images
        for i in range(len(val_images)):
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes[0].imshow(val_images[i].squeeze().cpu().detach().numpy(), cmap='gray')
            axes[0].set_title('Test Image (I1)')
            axes[1].imshow(val_seg[i].squeeze().cpu().detach().numpy(), cmap='gray')
            axes[1].set_title('Test Segmentation (I1_seg)')
            plt.show()


        net, criterion, optimizer = initialize_network_optimizer2D(128, 128, para, device)

        shooting_flag = 'SVF'

        print(f"Before training...")
        net_test_model(net, val_images, val_seg, shooting_flag, device)

        time_init = time.time()
        # ef exhaustive_train_model_with_validation(net, optimizer, num_epochs, train_dataset, val_dataset, val_segmentations, weight_dist, weight_reg, device, val_every=5)
        # phi_inv, _, loss, ssim = exhaustive_train_model_with_validation(net, optimizer, args.pretrain, train_images, val_images, val_seg, 10, 0.001, device, val_every=20) #, flag = 'SVF', val_every=20)
        
        input("Training with all possible pairs of images...")
        # phi_inv, _, loss, ssim = train_model_with_validation(net, optimizer, args.pretrain, train_images, val_images, val_seg, 10, 0.001, device, criterion, shooting_flag, val_every=20)
        phi_inv, _, loss, ssim = exhaustive_train_model_with_validation(net, optimizer, args.pretrain, train_images, train_seg, device, shooting_flag)
        torch.save(net.state_dict(), os.path.join(args.output, 'model_trained.pth'))
        time_end = time.time()
        print(f'Training time: {time_end - time_init} seconds')

        #load the model in eval mode
        net.load_state_dict(torch.load(os.path.join(args.output, 'model_trained.pth')))

        # net_test_model(net, val_images, test_seg, shooting_flag, device)

        # WE DONT HAVE TEST ANY MORE, ONLY VALIDATION
        print(f"After training...")
        net_test_model(net, val_images, val_seg, shooting_flag, device)

        # if shooting_flag == 'SVF':
        #     fine_tune_deformation(net, test_images, test_seg, device, criterion, num_steps=80, lr=0.01)

if __name__ == '__main__':
    main()
