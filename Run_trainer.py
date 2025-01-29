from os import PathLike
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import os
import glob
import json
import subprocess
import sys
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from easydict import EasyDict as edict
import random
import yaml
from losses import NCC, MSE, Grad
from networks import UnetDense
from SitkDataSet import SitkDataset as SData
from uEpdiff import Epdiff
from networks import *
from classifiers import *
import lagomorph as lm 
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode



def visualize_3d_volume(volume):
    """
    Visualizes a 3D volume using Plotly.
    """
    x, y, z = np.mgrid[:volume.shape[0], :volume.shape[1], :volume.shape[2]]

    fig = go.Figure(data=go.Volume(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=volume.flatten(),
    isomin=np.min(volume) + 0.1,  # Auto-adjusted min
    isomax=np.max(volume) - 0.1,  # Auto-adjusted max
    opacity=0.3,  # Increase opacity for better visibility
    surface_count=10,  # Reduce surfaces for better performance
    ))
    fig.show()



def visualize_images(atlas, target, deformed, zDim):
    """
    Visualize the atlas, target, and deformed images.
    """
    plt.figure(figsize=(15, 5))
    
    # Atlas, target, and deformed images
    plt.subplot(1, 3, 1)
    plt.imshow(atlas[0, 0, :, :, zDim//2].cpu().detach().numpy(), cmap='gray')
    plt.title('Atlas')
    plt.axis('off')
    

    plt.subplot(1, 3, 2)
    plt.imshow(target[0, 0, :, :, zDim//2].cpu().detach().numpy(), cmap='gray')
    plt.title('Target')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(deformed[0, 0, :, :, zDim//2].cpu().detach().numpy(), cmap='gray')
    plt.title('Deformed')
    plt.axis('off')
    
    plt.show()


def save_image_as_nifti(image, filename):
    """
    Guarda una imagen en formato NIfTI.
    """
    image_np = image.cpu().detach().numpy()
    image_sitk = sitk.GetImageFromArray(image_np)
    sitk.WriteImage(image_sitk, filename)

def visualize_deformation_field(deformation_field, zDim):
    """
    Visualize the deformation field (in 2D, i.e., x and y directions).
    """
    plt.figure(figsize=(10, 10))
    plt.quiver(deformation_field[0, :, :, zDim//2, 0].cpu().detach().numpy(),
               deformation_field[0, :, :, zDim//2, 1].cpu().detach().numpy())
    plt.title('Deformation Field')
    plt.axis('off')
    plt.show()




def plot_loss(losses):
    """
    Loss plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()



def get_device():
    """Returns the device available (cuda or cpu)"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def read_yaml(path):
    """Reads a YAML file and returns its contents as a dictionary."""
    try:
        with open(path, 'r') as f:
            file = edict(yaml.load(f, Loader=yaml.FullLoader))
        return file
    except:
        print('NO FILE READ!')
        return None


def load_and_preprocess_data(data_dir, json_file, keyword):
    """
    Loads and preprocesses data from a specified directory and JSON file.
    Returns the dimensions of the loaded data.
    """
    readfilename = f'{data_dir}/{json_file}.json'
    try:
        with open(readfilename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f'Error loading JSON data: {e}')
        return None
    outputs = []
    temp_scan = sitk.GetArrayFromImage(sitk.ReadImage(f'{data_dir}/{data[keyword][0]["image"]}'))
    xDim, yDim, zDim = temp_scan.shape
    return xDim, yDim, zDim


def initialize_network_optimizer(xDim, yDim, zDim, para, dev):
    """
    Initializes the atlas building neural network, classifier, loss functions, optimizer, and scheduler.
    Returns the initialized objects.
    """
    # Initialize the atlas building network (UnetDense)
    net = UnetDense(inshape=(xDim, yDim, zDim),
                      nb_unet_features=[[16, 32,32], [ 32, 32, 32, 16, 16]], #[16, 32,32], [ 32, 32, 32, 16, 16]
                      nb_unet_conv_per_level=1,
                      int_steps=7,
                      int_downsize=2,
                      src_feats=1,
                      trg_feats=1,
                      unet_half_res=True)
    net = net.to(dev)

    # Initialize the image classifier (Flexi3DCNN)
    in_channels = 1
    conv_channels = [8, 16, 16, 32, 32]  # Number of channels for each convolutional layer
    conv_kernel_sizes = [3, 3, 3,3, 3]  # Kernel sizes for each convolutional layer
    activation = 'ReLU'  # Activation function
    num_classes = 2 # Number of classes
    clfer = Flexi3DCNN(in_channels, conv_channels, conv_kernel_sizes, num_classes, activation)
    clfer = clfer.to(dev)

    # Combine parameters for optimization
    params = list(net.parameters()) + list(clfer.parameters())

    # Initialize loss functions
    criterion_clf = nn.CrossEntropyLoss()
    if para.model.loss == 'L2':
        criterion = nn.MSELoss()
    elif para.model.loss == 'L1':
        criterion = nn.L1Loss()

    # Initialize optimizer
    if para.model.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=para.solver.lr)
    elif para.model.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=para.solver.lr, momentum=0.9)

    # Initialize scheduler (CosineAnnealingLR)
    scheduler = CosineAnnealingLR(optimizer, T_max=para.solver.epochs)

    return net, clfer, criterion, criterion_clf, num_classes, optimizer, scheduler


def train_network(trainloader, aveloader, net, clfer, para, criterion, criterion_clf, num_classes, optimizer, scheduler, DistType, RegularityType, weight_dist, weight_reg, weight_latent, reduced_xDim, reduced_yDim, reduced_zDim, xDim, yDim, zDim, dev, flag):
    """
    Trains the atlas building neural network and classifier.
    """

    print(f"Training on {len(trainloader.dataset)} images")
    print(f"Validation on {len(aveloader.dataset)} images")
    print(f"Using device: {dev}")
    print(f"Network architecture: {net}")
    print(f"Classifier architecture: {clfer}")
    print(f"Number of epochs: {para.solver.epochs}")
    print(f"Batch size: {para.solver.batch_size}")
    print(f"Learning rate: {para.solver.lr}")
    print(f"Atlas learning rate: {para.solver.atlas_lr}")
    print(f"Pretrain epochs: {para.model.pretrain_epoch}")
    print(f"Loss function: {para.model.loss}")
    print(f"Optimizer: {para.model.optimizer}")

    losses = []
    running_loss = 0
    total = 0
    ''' Define fluid paramerts if using vector-momenta to shoot forward'''
    fluid_params = [1.0, 0.1, 0.05]
    lddmm_metirc = lm.FluidMetric(fluid_params)
    # Get an initialization of the atlas
    for ave_scan in trainloader:
        atlas, temp = ave_scan
    atlas.requires_grad=True
    opt = optim.Adam([atlas], lr=para.solver.atlas_lr) 

    for epoch in range(para.solver.epochs):
        net.train()
        clfer.train()
        print('epoch:', epoch)
        for j, tar_bch in enumerate(trainloader):
            b, c, w, h, l = tar_bch[0].shape
            optimizer.zero_grad()
            phiinv_bch = torch.zeros(b, w, h, l, 3).to(dev)
            reg_save = torch.zeros(b, w, h, l, 3).to(dev)
            
            # Shuffle the pairs then pretrain the atlas building network
            if epoch <= para.model.pretrain_epoch:
                perm_indices = torch.randperm(b)
                atlas_bch = tar_bch[0][perm_indices]
            else:
                atlas_bch = torch.cat(b*[atlas]).reshape(b, c, w, h, l)

            atlas_bch = atlas_bch.to(dev).float() 
            tar_bch_img = tar_bch[0].to(dev).float() 
            
            # Train atlas building with extracted latent features
            pred = net(atlas_bch, tar_bch_img, registration=True, shooting = flag) 

            # Train image classifier with feature fusion strategy using a specified weighting parameter, this network will not be updated unless the atlas building is pretrained
            cl_pred = clfer (tar_bch_img ,pred[2], weight_latent)

            # Create a tensor from the ground truth label, one-hot for multi-classes
            tar_bch_lbl = F.one_hot(torch.tensor(int(tar_bch[1][0])), num_classes).to(dev).float()
            clf_loss = criterion_clf(cl_pred[0], tar_bch_lbl)
            
            # Characterize the geometric shape information using different methods after obtaining the momentum from the atlas building network
            if (flag == "FLDDMM"): # LDDMM to perform geodesic shooting 
                momentum = pred[0].permute(0, 4, 3, 2, 1)
                identity = get_grid2(xDim, dev).permute([0, 4, 3, 2, 1])  
                epd = Epdiff(dev, (reduced_xDim, reduced_yDim, reduced_zDim), (xDim, yDim, zDim), para.solver.Alpha, para.solver.Gamma, para.solver.Lpow)

                for b_id in range(b):
                    v_fourier = epd.spatial2fourier(momentum[b_id,...].reshape(w, h , l, 3))
                    velocity = epd.fourier2spatial(epd.Kcoeff * v_fourier).reshape(w, h , l, 3)  
                    # sitk.WriteImage(sitk.GetImageFromArray(velocity.detach().cpu().numpy()), "./Velocity0.nii.gz")
                    reg_temp = epd.fourier2spatial(epd.Lcoeff * v_fourier * v_fourier)
                    num_steps = para.solver.Euler_steps
                    v_seq, displacement = epd.forward_shooting_v_and_phiinv(velocity, num_steps)  
                    phiinv = displacement.unsqueeze(0) + identity
                    phiinv_bch[b_id,...] = phiinv 
                    reg_save[b_id,...] = reg_temp

                dfm = Torchinterp(atlas_bch,phiinv_bch) 
                Dist = criterion(dfm, tar_bch_img)
                Reg_loss =  reg_save.sum()
                if epoch <= para.model.pretrain_epoch:
                    loss_total =  Dist + weight_reg * Reg_loss
                else:
                    loss_total =  Dist + weight_reg * Reg_loss + clf_loss

            elif (flag == "SVF"): # Stationary velocity fields to shoot forward 
                print (pred[1].shape)
                Dist = NCC().loss(pred[0], tar_bch_img)   
                Reg = Grad( penalty= RegularityType)
                Reg_loss  = Reg.loss(pred[1])
                if epoch <= para.model.pretrain_epoch:
                    loss_total = weight_dist * Dist + weight_reg * Reg_loss 
                else:
                    loss_total = weight_dist * Dist + weight_reg * Reg_loss + clf_loss

            elif (flag == "VecMome"): # A spatial version of LDDMM on CUDA to perform geodesic shooting 
                h = lm.expmap(lddmm_metirc, pred[1], num_steps= para.solver.Euler_steps)
                Idef = lm.interp(atlas_bch, h)
                v = lddmm_metirc.sharp(pred[1])
                reg_term = (v*pred[1]).mean()
                
                if epoch <= para.model.pretrain_epoch:
                    loss_total= (1/(para.solver.Sigma*para.solver.Sigma))*NCC().loss(Idef, tar_bch_img) + reg_term
                else:
                    loss_total= (1/(para.solver.Sigma*para.solver.Sigma))*NCC().loss(Idef, tar_bch_img) + reg_term + clf_loss

            loss_total.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss_total.item()
            total += running_loss
            running_loss = 0.0

        scheduler.step()  # Update learning rate

        '''Using Adam to update the atlas'''
        if epoch > para.model.pretrain_epoch:
            opt.step()
            opt.zero_grad()

        print('Total training loss:', total)
        losses.append(total)

        # Visualize the atlas, target, and deformed images
        visualize_images(atlas_bch, tar_bch_img, pred[0], zDim)

        # Visualize the deformation field
        visualize_deformation_field(pred[1], zDim)

    # Plot the training loss over epochs
    plot_loss(losses)

    # Save the trained atlas as a NIfTI file
    save_image_as_nifti(atlas_bch, 'atlas.nii.gz')

    #3D visualization of the atlas
    visualize_3d_volume(atlas_bch[0, 0, :, :, :].cpu().detach().numpy())



def main():
    """
    Main function to run the training process.
    """
    dev = get_device()
    para = read_yaml('./parameters.yml')
    data_dir = '.'
    json_file = 'train_json'
    keyword = 'train'
    xDim, yDim, zDim = load_and_preprocess_data(data_dir, json_file, keyword)
    dataset = SData('./train_json.json', "train")
    ave_data = SData('./train_json.json', 'train')
    trainloader = DataLoader(dataset, batch_size=para.solver.batch_size, shuffle=True)
    aveloader = DataLoader(ave_data, batch_size=1, shuffle=False)
    combined_loader = zip(trainloader, aveloader)
    net, clfer, criterion, criterion_clf, num_classes, optimizer, scheduler = initialize_network_optimizer(xDim, yDim, zDim, para, dev)

    train_network(trainloader, aveloader, net, clfer, para, criterion, criterion_clf, num_classes, optimizer, scheduler, NCC, 'l2', 0.5, 0.5, 0.2, 16, 16, 16, xDim, yDim, zDim, dev, "VecMome")
    


if __name__ == "__main__":
    main()
