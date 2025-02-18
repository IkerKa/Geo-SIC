


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
from losses import NCC, MSE, Grad
from networks import UnetDense  
from SitkDataSet import SitkDataset as SData
from uEpdiff import Epdiff
from uEpdiff2D import Epdiff2D
from networks import *
import argparse
import matplotlib.pyplot as plt # type: ignore
from datasets.datasetloader import GoogleDrawDataset2d, DataLoaderHandler
from datasets.datasetloader3d import MHD2DDataset
from datasets.datasetloader3d import DataLoaderHandler as d3d
import SimpleITK as sitk # type: ignore

from skimage.metrics import structural_similarity as ssim # type: ignore

#logger
from logger import Logger as log 


torch.cuda.empty_cache()



#--a bunch of functions that are metrics--
def compute_ssim_metric(atlas_img, target_img):
    # Convert tensors to numpy arrays (assume they are in [1, H, W])
    atlas_np = atlas_img.squeeze().detach().cpu().numpy()
    target_np = target_img.squeeze().detach().cpu().numpy()
    ssim_val = ssim(atlas_np, target_np, data_range=target_np.max() - target_np.min())
    return ssim_val
def compute_loss_components(dfm, target_img, reg_save, weight_reg):
    # Compute individual loss components
    distance_loss = F.mse_loss(dfm, target_img)
    regularity_loss = reg_save.sum()
    total_loss = distance_loss + weight_reg * regularity_loss
    return distance_loss.item(), regularity_loss.item(), total_loss

def compute_atlas_gradient_metrics(atlas):
    # Compute norm, mean and max of atlas gradients
    grad_norm = torch.norm(atlas.grad)
    grad_mean = torch.mean(torch.abs(atlas.grad))
    grad_max = torch.max(torch.abs(atlas.grad))
    return grad_norm.item(), grad_mean.item(), grad_max.item()


#function to obtain the average atlas from a set of images in order to compare it with the final atlas
def get_average_atlas(aveloader, _debug=False):
    # Get the average atlas from a set of images
    total = 0
    count = 0
    for batch in aveloader:
        img = batch[0].squeeze().cpu().numpy()
        total += img
        count += 1
    average_atlas = total / count

    if _debug:
        print('Average Atlas Shape:', average_atlas.shape)
        plt.imshow(average_atlas, cmap='gray')
        plt.title('Average Atlas')
        plt.axis('off')
        plt.show()



def save_atlas_2D(atlas, filename):
    """
    Save the atlas as a PNG image.
    
    Args:
        atlas (torch.Tensor): 2D tensor with the atlas. [1, H, W] || [H, W]
        filename (str): Path to save the image.
    """
    # Convert the tensor to a numpy array.
    atlas_np = atlas.squeeze().detach().cpu().numpy()
    
    # Opcional: Denormalize the image.
    atlas_np = (atlas_np * 255).astype(np.uint8)
    
    # PIL Image
    img = Image.fromarray(atlas_np)
    img.save(filename)


def visualize_atlas_2D(atlas):
    """
    Visualizes a 2D atlas image.

    Args:
        atlas (torch.Tensor): The atlas tensor, expected to have shape [1, H, W] or [H, W].
    """
    # Remove extra dimensions and move the tensor to CPU
    atlas_np = atlas.squeeze().detach().cpu().numpy()
    
    # Display the image using matplotlib with a grayscale colormap
    plt.imshow(atlas_np, cmap='gray')
    plt.title("Atlas")
    plt.axis('off')  # Hide axis for a cleaner view
    plt.show()


def visualize_loss(losses):
    """
    Visualizes the loss values.

    Args:
        losses (list): List of loss values.
    """
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.show()

def visualize_time(time_values):
    """
    Visualizes the time values.

    Args:
        time_values (list): List of time values.
    """
    plt.plot(time_values)
    plt.title("Time")
    plt.xlabel("Epoch")
    plt.ylabel("Time")
    plt.show()

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

################Parameter Loading#######################
def read_yaml(path):
    try:
        with open(path, 'r') as f:
            file = edict(yaml.load(f, Loader=yaml.FullLoader))
        return file
    except:
        print('NO FILE READ!')
        return None


##################Data Loading##########################
def load_and_preprocess_data(data_dir, json_file, keyword):
    readfilename = f'{data_dir}/{json_file}.json'
    try:
        with open(readfilename, 'r') as f:
            data = json.load(f)
            # print(data)
    except Exception as e:
        print(f'Error loading JSON data: {e}')
        return None
    outputs = []
    temp_scan = sitk.GetArrayFromImage(sitk.ReadImage(f'{data_dir}/{data[keyword][0]["image"]}'))
    print("temp_scan shape:", temp_scan.shape)
    temp_scan = temp_scan.astype(np.float32)
    xDim, yDim,  zDim = temp_scan.shape
    return xDim, yDim, zDim

# Same function as before, but with the reduced dimensions (2D images)
def load_and_preprocess_data2D(data_dir, json_file, keyword):
    readfilename = f'{data_dir}/{json_file}.json'
    try:
        with open(readfilename, 'r') as f:
            data = json.load(f)
            # print(data)
    except Exception as e:
        print(f'Error loading JSON data: {e}')
        return None
    outputs = []
    temp_scan = sitk.GetArrayFromImage(sitk.ReadImage(f'{data_dir}/{data[keyword][0]["image"]}'))
    temp_scan = temp_scan.astype(np.float32)
    print("temp_scan shape:", temp_scan.shape)

    xDim, yDim, _ = temp_scan.shape
    return xDim, yDim


def save_atlas(atlas, filename):
    
    atlas_np = atlas.squeeze().detach().cpu().numpy() 
    print("Atlas shape:", atlas_np.shape) 
    atlas_image = sitk.GetImageFromArray(atlas_np)
    # visualize_atlas(atlas)
    sitk.WriteImage(atlas_image, filename)



def initialize_network_optimizer(xDim, yDim, zDim, para, dev):
    net = UnetDense(inshape=(xDim, yDim, zDim),
                      nb_unet_features=[[16, 32,32], [ 32, 32, 32, 16, 16]], #[16, 32,32], [ 32, 32, 32, 16, 16]
                      nb_unet_conv_per_level=1,
                      int_steps=7,
                      int_downsize=2,
                      src_feats=1,
                      trg_feats=1,
                      unet_half_res=True)
    net = net.to(dev)

    if para.model.loss == 'L2':
        criterion = nn.MSELoss()
    elif para.model.loss == 'L1':
        criterion = nn.L1Loss()
    if para.model.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=para.solver.lr)
    elif para.model.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=para.solver.lr, momentum=0.9)

    return net, criterion, optimizer

def initialize_network_optimizer2D(xDim, yDim, para, dev):
    net = UnetDense(inshape=(xDim, yDim),
                      nb_unet_features=[[16, 32,32], [ 32, 32, 32, 16, 16]],
                      nb_unet_conv_per_level=1,
                      int_steps=7,
                      int_downsize=2,
                      src_feats=1,
                      trg_feats=1,
                      unet_half_res=True)
    net = net.to(dev)

    if para.model.loss == 'L2':
        criterion = nn.MSELoss()
    elif para.model.loss == 'L1':
        criterion = nn.L1Loss()
    if para.model.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=para.solver.lr)
    elif para.model.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=para.solver.lr, momentum=0.9)

    return net, criterion, optimizer

def train_network2D(trainloader, aveloader, net, para, criterion, optimizer, DistType, RegularityType, weight_dist, weight_reg,  reduced_xDim, reduced_yDim, xDim, yDim, dev, logger):
    """
    Train the network.

    Parameters:
    trainloader (DataLoader): DataLoader for training data.
    aveloader (DataLoader): DataLoader for average scans or random scans to intial atlas.
    net: The neural network model.
    para: Parameters object.
    criterion: The loss criterion.
    optimizer: The optimizer.
    DistType: Type of distance.
    RegularityType: Type of regularity.
    weight_dist: Weight for distance loss.
    weight_reg: Weight for regularity loss.
    reduced_xDim, reduced_yDim, reduced_zDim: Dimensions of reduced space.
    xDim, yDim, zDim: Dimensions of the input image.
    dev: Device to run on.
    """
    

    # Same function as before, but with the reduced dimensions (2D images)

    running_loss = 0
    total = 0
    loss_per_epoch = []
    times = []

    # Get an initialization of the atlas
    # for ave_scan in aveloader:
    #     # print(ave_scan)
    #     logger.info(message=f"Average scan shape: {ave_scan.shape}")
    #     # atlas, temp = ave_scan
    #     atlas = ave_scan
    #     break;

    random_idx = random.randint(0, len(aveloader)-1)
    for idx, ave_scan in enumerate(aveloader):
        if idx == random_idx:
            atlas = ave_scan
            logger.info(message=f"Average scan shape: {ave_scan.shape} from index {idx}")
            visualize_atlas_2D(atlas)
            break


    # Instead of getting the initial atlas from the training data, the initialization will be the average of the training data
    atlas.requires_grad=True
    opt = optim.Adam([atlas], lr=para.solver.atlas_lr)  #<---they used a different optimizer for the atlas


    logger.divider("Training setup")
    logger.custom(f"Training setup:", "green")
    logger.custom(f"Atlas shape: {atlas.shape}", "green")
    logger.custom(f"Number of epochs: {para.solver.epochs}", "green")
    logger.custom(f"Batch size: {para.solver.batch_size}", "green")
    logger.custom(f"Learning rate: {para.solver.lr}", "green")
    logger.custom(f"Images for training: {len(trainloader)}", "green")
    logger.divider()

    logger.banner("Training started")
    
    for epoch in range(para.solver.epochs):
        init = time.time()
        #save the current atlas in atlas_snapshots (and the raw file)
        save_atlas_2D(atlas, f'atlas_snapshots/atlas_epoch_{epoch}.png')
        save_atlas(atlas, f'atlas_snapshots/atlas_epoch_{epoch}.nii.gz')

        net.train()
        logger.divider(f'Epoch: {epoch} & Current loss: {total}')
        for j, tar_bch in enumerate(trainloader):
            logger.divider(f'Batch: {j}')
            #-take the dimensions of the batch for 2D images.
            logger.info(message=f"Batch shape: {tar_bch.shape}")

            b, c, w, h = tar_bch.shape
            #-restart the optimizer gradient
            optimizer.zero_grad()
            #-initialize the phiinv and reg_save tensors
            phiinv_bch = torch.zeros(b, w, h, 2).to(dev)
            reg_save = torch.zeros(b, w, h, 2).to(dev)

            # Now we wont pretrain the atlas building network 
            atlas_bch = torch.cat(b*[atlas]).reshape(b, c, w, h)
            atlas_bch = atlas_bch.to(dev).float()
            tar_bch_img = tar_bch.to(dev).float()

            #pass the atlas and the target image to the network
            y_src, momentum, latent_feat  = net(atlas_bch, tar_bch_img, registration=True) # ? y_src and latent_feat is not used
            momentum = momentum.permute(0, 3, 2, 1) # ? ARE THE SIZES CORRECT?
            
            #MATHS things
            img_size = w    # ASSUMING SQUARE IMAGES
            identity = get_grid2D(img_size, dev).permute([0, 3, 2, 1])
            # epd = Epdiff(dev, (reduced_xDim, reduced_yDim), (xDim, yDim), para.solver.Alpha, para.solver.Gamma, para.solver.Lpow)
            epd = Epdiff2D(dev, (reduced_xDim, reduced_yDim), (xDim, yDim), para.solver.Alpha, para.solver.Gamma, para.solver.Lpow)
            # logger.divider("Math part")

            for b_id in range(b):   #adapted to 2D images
                v_fourier = epd.spatial2fourier(momentum[b_id,...].reshape(w, h, 2))
                velocity = epd.fourier2spatial(epd.Kcoeff * v_fourier).reshape(w, h, 2)  
                reg_temp = epd.fourier2spatial(epd.Lcoeff * v_fourier * v_fourier)
                num_steps = para.solver.Euler_steps
                v_seq, displacement = epd.forward_shooting_v_and_phiinv(velocity, num_steps)    # ! Bottleneck for complexity
                phiinv = displacement.unsqueeze(0) + identity
                phiinv_bch[b_id,...] = phiinv 
                reg_save[b_id,...] = reg_temp

            dfm = Torchinterp2D(atlas_bch,phiinv_bch)
            Dist = criterion(dfm, tar_bch_img)
            Reg_loss =  reg_save.sum()
            loss_total =  Dist + weight_reg * Reg_loss
            loss_total.backward(retain_graph=True)
            
            #Update the network parameters
            optimizer.step()

            running_loss += loss_total.item()
            loss_per_epoch.append(loss_total.item())
            total += running_loss
            running_loss = 0.0

            #metrics
            grad_norm, grad_mean, grad_max = compute_atlas_gradient_metrics(atlas)
            logger.info(message=f"--Atlas gradient, max: {grad_max}, min: {grad_mean}")
            logger.info(message=f"--Atlas gradient norm: {grad_norm}")



        #if epoch >= para.model.pretrain_epoch:
        opt.step()
        opt.zero_grad() 
        
        end = time.time()
        times.append(end - init)
    
    logger.success(message="Training finished")
    save_atlas_2D(atlas, f'atlas_snapshots/final_atlas.png')
    save_atlas(atlas, f'atlas_snapshots/final_atlas.nii.gz')

    #plot the loss per epoch
    visualize_loss(loss_per_epoch)
    visualize_time(times)

    return atlas
    
def train_network(trainloader, aveloader, net, para, criterion, optimizer, DistType, RegularityType, weight_dist, weight_reg,  reduced_xDim, reduced_yDim, reduced_zDim, xDim, yDim, zDim, dev):
    """
    Train the network.

    Parameters:
    trainloader (DataLoader): DataLoader for training data.
    aveloader (DataLoader): DataLoader for average scans or random scans to intial atlas.
    net: The neural network model.
    para: Parameters object.
    criterion: The loss criterion.
    optimizer: The optimizer.
    DistType: Type of distance.
    RegularityType: Type of regularity.
    weight_dist: Weight for distance loss.
    weight_reg: Weight for regularity loss.
    reduced_xDim, reduced_yDim, reduced_zDim: Dimensions of reduced space.
    xDim, yDim, zDim: Dimensions of the input image.
    dev: Device to run on.
    """
    running_loss = 0
    total = 0

    # Get an initialization of the atlas
    for ave_scan in trainloader:
        atlas, temp = ave_scan
        break;  #<---they only want the first one but didn't use the break statement

    # Instead of getting the initial atlas from the training data, the initialization will be the average of the training data
    # atlas = get_average_atlas(aveloader)
    atlas.requires_grad=True
    opt = optim.Adam([atlas], lr=para.solver.atlas_lr) 

    print("Training setup:")
    print("Atlas shape:", atlas.shape)
    print("Number of epochs:", para.solver.epochs)
    print("Batch size:", para.solver.batch_size)
    print("Learning rate:", para.solver.lr)
    print("Images for training:", len(trainloader))

    print("\n\n")


    for epoch in range(para.solver.epochs):

        # Agregar en cada epoch:
        # 1. Visualizar el atlas
        # slice_idx = atlas.shape[2] // 2
        # plt.figure()
        # plt.imshow(atlas[0,0,slice_idx].detach().cpu().numpy(), cmap='gray')
        # plt.title(f"Epoch {epoch} - Max: {atlas.max():.3f}, Min: {atlas.min():.3f}")
        # plt.show()

        #we will save the whole atlas per epoch to visualize it later in a .nii file 
            
        net.train()
        print('epoch:', epoch)
        save_atlas(atlas, f'atlas_snapshots/atlas_epoch_{epoch}.nii.gz')
        for j, tar_bch in enumerate(trainloader):
            print('-batch:', j)
            b, c, w, h, l = tar_bch[0].shape
            optimizer.zero_grad()
            phiinv_bch = torch.zeros(b, w, h, l, 3).to(dev)
            reg_save = torch.zeros(b, w, h, l, 3).to(dev)
            
            # Pretrain the atlas building network
            # if epoch <= para.model.pretrain_epoch:
            #     perm_indices = torch.randperm(b)
            #     atlas_bch = tar_bch[0][perm_indices]
            # else:
            atlas_bch = torch.cat(b*[atlas]).reshape(b, c, w, h, l)

            atlas_bch = atlas_bch.to(dev).float() 
            tar_bch_img = tar_bch[0].to(dev).float() 
            _ , momentum, latent_feat  = net(atlas_bch, tar_bch_img, registration=True) 
            momentum = momentum.permute(0, 4, 3, 2, 1)
            identity = get_grid2(xDim, dev).permute([0, 4, 3, 2, 1])  
            epd = Epdiff(dev, (reduced_xDim, reduced_yDim, reduced_zDim), (xDim, yDim, zDim), para.solver.Alpha, para.solver.Gamma, para.solver.Lpow)

            for b_id in range(b):
                v_fourier = epd.spatial2fourier(momentum[b_id,...].reshape(w, h , l, 3))
                velocity = epd.fourier2spatial(epd.Kcoeff * v_fourier).reshape(w, h , l, 3)  
                reg_temp = epd.fourier2spatial(epd.Lcoeff * v_fourier * v_fourier)
                num_steps = para.solver.Euler_steps
                v_seq, displacement = epd.forward_shooting_v_and_phiinv(velocity, num_steps)  
                phiinv = displacement.unsqueeze(0) + identity
                phiinv_bch[b_id,...] = phiinv 
                reg_save[b_id,...] = reg_temp

            dfm = Torchinterp(atlas_bch,phiinv_bch) 
            Dist = criterion(dfm, tar_bch_img)
            Reg_loss =  reg_save.sum()
            loss_total =  Dist + weight_reg * Reg_loss
            loss_total.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss_total.item()
            total += running_loss
            running_loss = 0.0

            print(f"--Atlas gradient, max: {atlas.grad.max()}, min: {atlas.grad.min()}")
            print(f"--Atlas gradient norm: {torch.norm(atlas.grad)}")

        # if epoch >= para.model.pretrain_epoch:
        opt.step()
        opt.zero_grad()

        print('Total training loss:', total)

    print('Finished Training')
    save_atlas(atlas, 'final_atlas.nii.gz')
    visualize_atlas(atlas)
    return atlas


def main():


    lg = log(log_file='aplication.log')

    parser = argparse.ArgumentParser(description='Run Atlas Trainer')
    args = parser.parse_args()

    dev = get_device()
    para = read_yaml('./parameters.yml')
    two_dims = 1

    if two_dims == 1:
        lg.custom("Running Atlas Trainer with 2D images", "green")
        datadir = 'datasets/jsons/circle.ndjson'
        #load the ndjson file and get the dimensions of the image
        lg.info(message=f"Loading dataset from: {datadir}")
        dataset = GoogleDrawDataset2d(datadir, samples=200)
        trainloader = DataLoader(dataset, batch_size=para.solver.batch_size, shuffle=True)   # ? Batch size?
        aveloader = DataLoader(dataset, batch_size=1, shuffle=False)

        #log the sizes of the dataset 2D
        lg.custom(f"Training dataset size: {len(trainloader)}", "green")


        datahandler = DataLoaderHandler(ndjson_file=datadir, samples=5, resize=256, batch_size=16)
        datahandler.show_example()
        datahandler.save_dataloader('dataloaderCIRCLES.pt')

        #obtain the dimensions of the image, generic way, take an image and obtain its dimensions
        for batch in trainloader:
            image = batch[0]
            lg.info(message=f"Image shape: {image.shape}")
            break

        xDim, yDim = image.shape[1], image.shape[2]
        lg.info(message=f"2D image dimensions: {xDim}, {yDim}")

    
        combined_loader = zip(trainloader, aveloader )  # ? Why do we need this combined_loader?
        net, criterion, optimizer = initialize_network_optimizer2D(xDim, yDim, para, dev)

        # pause();

    else:
        lg.custom("Running Atlas Trainer with Brain slices", "green")
        datadir = 'datasets/dcm/'
        #load the ndjson file and get the dimensions of the image
        lg.info(message=f"Loading dataset from: {datadir}")
        dataset = MHD2DDataset(datadir)
        trainloader = DataLoader(dataset, batch_size=para.solver.batch_size, shuffle=True)   # ? Batch size?
        aveloader = DataLoader(dataset, batch_size=1, shuffle=False)

        #log the sizes of the dataset 2D
        lg.custom(f"Training dataset size: {len(trainloader)}", "green")


        datahandler = d3d(mhd_folder=datadir, batch_size=8)
        datahandler.show_example()
        # datahandler.save_dataloader('dataloader.pt')

        #obtain the dimensions of the image, generic way, take an image and obtain its dimensions
        for batch in trainloader:
            image = batch[0]
            lg.info(message=f"Image shape: {image.shape}")
            break

        xDim, yDim = image.shape[1], image.shape[2]
        lg.info(message=f"2D image dimensions: {xDim}, {yDim}")

    
        combined_loader = zip(trainloader, aveloader )  # ? Why do we need this combined_loader?
        net, criterion, optimizer = initialize_network_optimizer2D(xDim, yDim, para, dev)
    
    #plot the average atlas
    # get_average_atlas(aveloader, _debug=True)
    


    atlas = train_network2D(trainloader, aveloader, net, para, criterion, optimizer, NCC, 'l2', 10, 0.001, 16,16, xDim, yDim, dev, lg)

        
    
    
    
    # overlay_atlas_and_image(atlas, image)
  
if __name__ == "__main__":
    main()








       
    
 
        


