
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
def train_model(net, optimizer, num_epochs, train_dataset, test_dataset, weight_dist, weight_reg,device):
    loss_total = 0
    phi_inv = None
    
    print('Pre-training for', num_epochs, 'epochs')

    ssim_per_epoch = []
    loss_per_epoch = []

    I1_fixed = test_dataset[0]

    phi_inv, y_src = None, None

    tqdm_epochs = tqdm.tqdm(range(num_epochs))
    for epoch in tqdm_epochs:
        tqdm_epochs.set_description(f'Epoch {epoch + 1}/{num_epochs}')

        net.eval()
        optimizer.zero_grad()

        #The training now is by selecting one random image from the train dataset while fixing the other image (that will be used for the training)
        idx = random.randint(0, len(train_dataset) - 1)
        I1 = I1_fixed
        I2 = train_dataset[idx]

        # plot_pairs(I1, I2, title1='Fixed Image', title2='Random Image')

        # #wait keyboard
        # input("Press Enter to continue...")


        I1 = I1.to(device).float()
        I2 = I2.to(device).float()



        y_src, momentum, _, new_locs = net(I1, I2, registration=True, shooting='SVF', return_phi=True)
        
        dist_loss = NCC(win=[9,9]).loss(y_src, I2)
        reg_loss = Grad(penalty='l2').loss2D(momentum)
        
        loss_total = weight_dist * dist_loss + weight_reg * reg_loss
        loss_total.backward()
        optimizer.step()

        loss_per_epoch.append(loss_total.item())

        loss_total = 0.0
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

def evaluate_model(phi_inv, test_dataset, test_shapes, train_dataset, train_shapes, device, args):


    #Take the test images
    I1 = convert_to_tensor(test_dataset[0], device)
    I2 = convert_to_tensor(test_dataset[1], device)

    I1_seg = convert_to_tensor(test_shapes[0], device)
    I2_seg = convert_to_tensor(test_shapes[1], device)

    # I1 composed with phi_inv should be close to I2 -> warping I1 with phi_inv should give I2
    y_src = F.grid_sample(I1, phi_inv.unsqueeze(0), align_corners=True, mode='bilinear')

    warped_seg_np, fixed_seg_np, dice_score = compute_segmentation(I1_seg, phi_inv, I2_seg, device)

    if args.output:
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
        print(f'Got the tensor files...')

    net, _, optimizer = initialize_network_optimizer2D(128, 128, para, device)
    phi_inv, y_src = train_model(net, optimizer, args.pretrain, train_tensor_dataset, test_tensor_dataset, 10, 0.001, device)
    
    
    evaluate_model(phi_inv, test_dataset, test_shapes,train_dataset, train_shapes, device, args)

    
    
    print('Done!')

if __name__ == '__main__':
    main()

