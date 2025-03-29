
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

    return phi_inv, y_src, val_ssim_scores, val_rmse_scores, val_dice_scores

def train_model_with_validation(net, optimizer, num_epochs, train_dataset, val_dataset, val_segmentations, weight_dist, weight_reg, device, val_every=5):
    loss_total = 0
    phi_inv = None
    ssim_per_epoch, loss_per_epoch = [], []
    val_ssim_scores, val_rmse_scores, val_dice_scores = [], [], []

    tqdm_epochs = tqdm.tqdm(range(num_epochs))

    pairs = [(i, j) for i in range(len(train_dataset)) for j in range(len(train_dataset))]
    print(f'Training with {len(pairs)} pairs of images...')
    for epoch in tqdm_epochs:

        idx = random.randint(0, len(train_dataset) - 1)
        I1, I2 = train_dataset[idx], train_dataset[(idx + 1) % len(train_dataset)]
        

        # Per each epoch, train with all possible pairs of images? can be that done? i think it would improve
        tqdm_epochs.set_description(f'Epoch {epoch + 1}/{num_epochs}')
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

    return phi_inv, y_src, val_ssim_scores, val_rmse_scores, val_dice_scores

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

def net_test_model(net, test_images, test_segs, device):
    # with the trained model, test the images (phi inverted)

    # can be also use the net?
    pairs = [(i, j) for i in range(len(test_images)) for j in range(len(test_images))]
    print(f'Testing {len(pairs)} pairs of images...')

    with torch.no_grad():
        for i, j in pairs:
            I1 = test_images[i]
            I2 = test_images[j] 
            I1_seg = test_segs[i]
            I2_seg = test_segs[j]

            I1 = I1.to(device).float()
            I2 = I2.to(device).float()
            net.eval()  # Modo evaluación
            y_src, _, _, new_locs = net(I1, I2, registration=True, shooting='SVF', return_phi=True)
            warped_seg_np, fixed_seg_np, dice_score = compute_segmentation(I1_seg, new_locs[0,...], I2_seg, device)

            #obtain the metrics and save them
            ssim_score = ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(), data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min())
            rmse_score = np.sqrt(np.mean((I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()) ** 2))
            mean_dice_score = np.mean(dice_score)

            print(f'Test - SSIM: {ssim_score:.4f}, RMSE: {rmse_score:.4f}, Dice: {mean_dice_score:.4f}')

            save_metrics('output', I2, y_src, ssim_score, rmse_score, mean_dice_score)

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

        # Load data
        all_dataset = load_all_data()

        random.shuffle(all_dataset)

        num_total = len(all_dataset)
        # num_train = int(0.8 * num_total)
        # num_val = int(0.1 * num_total)
        # num_test = num_total - num_train - num_val

        num_train = 12
        num_val = 2
        num_test = 2


        # Split the dataset into train, validation and test sets
        train_dataset = all_dataset[:num_train]
        val_dataset = all_dataset[num_train:num_train + num_val]
        test_dataset = all_dataset[num_train + num_val:]

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

        net, _, optimizer = initialize_network_optimizer2D(128, 128, para, device)

        time_init = time.time()
        phi_inv, _, _, _, _ = train_model_with_validation(net, optimizer, args.pretrain, train_images, val_images, val_seg, 10, 0.001, device, val_every=5)
        time_end = time.time()

        print(f'Training time: {time_end - time_init} seconds')
        # test_model(phi_inv, test_images, test_seg, device)
        net_test_model(net, test_images, test_seg, device)



if __name__ == '__main__':
    main()

