from os import PathLike
from pathlib import Path
from signal import pause
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
import nibabel as nib
import random 
import yaml
from losses import NCC, MSE, Grad
from networks import UnetDense  
from SitkDataSet import SitkDataset as SData
from uEpdiff import Epdiff
from networks import *
import argparse
import matplotlib.pyplot as plt # type: ignore

import SimpleITK as sitk # type: ignore

def read_atlas():
    try:
        atlas_path = './atlas.nii.gz'
        final_atlas_path = './final_atlas.nii.gz'
        atlas = sitk.ReadImage(atlas_path)
        final_atlas = sitk.ReadImage(final_atlas_path)
        #visualize the atlas
        atlas_np = sitk.GetArrayFromImage(atlas)
        final_atlas_np = sitk.GetArrayFromImage(final_atlas)
        visualize_atlas(torch.tensor(atlas_np))
        visualize_atlas(torch.tensor(final_atlas_np))
    except Exception as e:
        print(f'Error reading atlas: {e}')
        return None
    

def overlay_atlas_and_image(atlas, image):
    # Convierte los tensores a arrays de NumPy
    atlas_np = atlas.squeeze().detach().cpu().numpy()
    image_np = image.squeeze().cpu().numpy()

    # Selecciona una sección transversal (por ejemplo, la mitad en el eje z)
    slice_idx = atlas_np.shape[0] // 2
    atlas_slice = atlas_np[slice_idx, :, :]
    image_slice = image_np[slice_idx, :, :]

    # Superpone las imágenes
    plt.imshow(image_slice, cmap='gray', alpha=0.5)  # Imagen de entrenamiento (semitransparente)
    plt.imshow(atlas_slice, cmap='hot', alpha=0.5)   # Atlas (semitransparente)
    plt.title('Superposición: Atlas e Imagen de Entrenamiento')
    plt.axis('off')
    plt.show()



def save_training_image(trainloader, filename):
    # Obtén un batch de imágenes de entrenamiento
    for batch in trainloader:
        images, _ = batch  # Las imágenes están en el primer elemento del batch
        break  # Solo toma el primer batch

    # Selecciona la primera imagen del batch
    image = images[0].squeeze().cpu().numpy()  # Elimina dimensiones adicionales y convierte a CPU

    # Guarda la imagen en un archivo
    image_sitk = sitk.GetImageFromArray(image)
    sitk.WriteImage(image_sitk, filename)


#function to obtain the average atlas from a set of images in order to compare it with the final atlas
def get_average_atlas(aveloader):
    # Get the average atlas from a set of images
    total = 0
    for batch in aveloader:
        images, _ = batch  # Las imágenes están en el primer elemento del batch
        total += images
    average_atlas = total / len(aveloader)
    visualize_atlas(average_atlas)
    return average_atlas

def visualize_training_image(trainloader):
    # Obtén un batch de imágenes de entrenamiento
    for batch in trainloader:
        images, _ = batch  # Las imágenes están en el primer elemento del batch
        print("Batch shape:", images.shape)
        break  # Solo toma el primer batch

    # Selecciona la primera imagen del batch
    image = images[0].squeeze().cpu().numpy()  # Elimina dimensiones adicionales y convierte a CPU

    #añade una dimension para que sea 3D
    # image = image[np.newaxis, :, :]


    print('(PLOT) Dimensiones de la imagen de entrenamiento:', image.shape)

    #mostrar mas informacion de la imagen leida
    print('Tipo de datos:', image.dtype)
    print('Valor mínimo:', image.min())
    print('Valor máximo:', image.max())
    print('Valor promedio:', image.mean())
    print('Desviación estándar:', image.std())



    # Visualiza una sección transversal de la imagen (por ejemplo, la mitad en el eje z)
    plt.imshow(image[image.shape[0] // 2, :, :], cmap='gray')
    plt.title('Imagen de Entrenamiento')
    plt.axis('off')
    plt.show()


def visualize_all_training_images(images):
  
    # Visualize all the training images
    for i in range(len(images)):
        image = images[i].squeeze().cpu().numpy()
        plt.imshow(image[image.shape[0] // 2, :, :], cmap='gray')
        plt.title(f'Imagen de Entrenamiento {i+1}')
        plt.axis('off')
        plt.show()

def save_atlas(atlas, filename):
    # Convierte el tensor de PyTorch a un array de NumPy
    atlas_np = atlas.squeeze().detach().cpu().numpy()  # Elimina dimensiones adicionales y convierte a CPU
    # Crea una imagen SimpleITK a partir del array de NumPy
    atlas_image = sitk.GetImageFromArray(atlas_np)
    # Guarda la imagen en un archivo
    sitk.WriteImage(atlas_image, filename)

def visualize_atlas_training(atlas_tensor, epoch, save_path='atlas_snapshots'):
    os.makedirs(save_path, exist_ok=True)
    # Detach, move to CPU, and convert to numpy
    atlas_np = atlas_tensor.detach().cpu().numpy()
    # Assuming shape: [batch, channel, x, y, z]
    # Example: Save middle slice of the first channel and batch
    slice_idx = atlas_np.shape[2] // 2
    plt.imshow(atlas_np[0, 0, slice_idx, :, :], cmap='gray')
    plt.title(f'Atlas Epoch {epoch}')
    plt.savefig(f'{save_path}/epoch_{epoch}.png')
    plt.close()
    # Optionally save as NIfTI
    nib.save(nib.Nifti1Image(atlas_np[0, 0], np.eye(4)), f'{save_path}/epoch_{epoch}.nii.gz')


def visualize_atlas(atlas):
    # Convierte el tensor de PyTorch a un array de NumPy
    atlas_np = atlas.squeeze().detach().cpu().numpy()

    # Visualiza una sección transversal del atlas
    slice = atlas_np.shape[0] // 2
    print("taking the slice:", slice)
    plt.imshow(atlas_np[slice], cmap='gray')
    plt.title('Atlas')
    plt.axis('off')
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

##################2D data loading#######################
def load_and_preprocess_data2D(data_dir, json_file, keyword):
    readfilename = f'{data_dir}/{json_file}.json'
    print(readfilename)
    try:
        with open(readfilename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f'Error loading JSON data: {e}')
        return None
    
    outputs = []
    temp_scan = sitk.GetArrayFromImage(sitk.ReadImage(f'{data_dir}/datasets/triangle_sketches/{data[keyword][0]["image"]}'))
    print(temp_scan.shape)
    xDim, yDim, _ = temp_scan.shape
    return xDim, yDim



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


def train_network_mine(trainloader, aveloader, net, para, criterion, optimizer, DistType, RegularityType, weight_dist, weight_reg,  reduced_xDim, reduced_yDim, reduced_zDim, xDim, yDim, zDim, dev):
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
        atlas = atlas.float().to(dev)
        print("Atlas shape:", atlas.shape)
        visualize_atlas(atlas)
        break  #<---they only want the first one but didn't use the break statement 

    
    # atlas = torch.nn.Parameter(atlas, requires_grad=True) 
    opt = optim.Adam([atlas], lr=para.solver.atlas_lr)
    atlas_history = []
    for epoch in range(para.solver.epochs):
        net.train()
        print('epoch:', epoch)

        visualize_atlas_training(atlas, epoch)
        atlas_history.append(atlas.detach().clone().cpu().numpy())

        for j, tar_bch in enumerate(trainloader):

            print('batch:', j)
            b, c, w, h, l = tar_bch[0].shape
            optimizer.zero_grad()
            opt.zero_grad()
            phiinv_bch = torch.zeros(b, w, h, l, 3).to(dev)
            reg_save = torch.zeros(b, w, h, l, 3).to(dev)
            
            # Pretrain the atlas building network
            #I delete the pretrain condition, not sure if it is necessary (at least for now that we dont have too many images and the training is short)
            atlas_bch = torch.cat(b*[atlas]).reshape(b, c, w, h, l)

            atlas_bch = atlas_bch.to(dev).float() 
            # atlas_bch.requires_grad = True
            tar_bch_img = tar_bch[0].to(dev).float() 

            # print("atlas_bch shape:", atlas_bch.shape)
            # print("tar_bch_img shape:", tar_bch_img.shape)

            
            
            _ , momentum, latent_feat = net(atlas_bch, tar_bch_img, registration=True)  #When TRUE it returns 3 parameters
            
            # print(res)
            # latent_feat, momentum, _ = res
            # print("Network output (momentum) shape:", momentum.shape)
            # print("Network output (latent_feat) shape:", latent_feat.shape)
            momentum = momentum.permute(0, 4, 3, 2, 1)
            # print(momentum.shape)
            identity = get_grid2(xDim, dev).permute([0, 4, 3, 2, 1])  
            epd = Epdiff(dev, (reduced_xDim, reduced_yDim, reduced_zDim), (xDim, yDim, zDim), para.solver.Alpha, para.solver.Gamma, para.solver.Lpow)

            for b_id in range(b):
                v_fourier = epd.spatial2fourier(momentum[b_id,...].reshape(w, h, l, 3))
                velocity = epd.fourier2spatial(epd.Kcoeff * v_fourier).reshape(w, h , l, 3)  
                reg_temp = epd.fourier2spatial(epd.Lcoeff * v_fourier * v_fourier)
                num_steps = para.solver.Euler_steps
                v_seq, displacement = epd.forward_shooting_v_and_phiinv(velocity, num_steps)  
                phiinv = displacement.unsqueeze(0) + identity
                phiinv_bch[b_id,...] = phiinv 
                reg_save[b_id,...] = reg_temp

            dfm = Torchinterp(atlas_bch,phiinv_bch) 

            #update the atlas based on the distance and regularity loss
            Dist = criterion(dfm, tar_bch_img)
            Reg_loss =  reg_save.sum()
            loss_total =  Dist + weight_reg * Reg_loss
            loss_total.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss_total.item()
            total += running_loss
            running_loss = 0.0

            print("atlas_bch.requires_grad:", atlas_bch.requires_grad)  # Should be True
            print("dfm.requires_grad:", dfm.requires_grad)  # Should be True

        # if epoch >= para.model.pretrain_epoch:

            print("--before optimization mean:", atlas.mean())
            print("Gradiente de atlas:", atlas.grad)
            # print("Atlas Gradient Mean:", atlas.grad.mean())
            # print("Atlas Gradient Max:", atlas.grad.max())
            # print("Atlas Gradient Min:", atlas.grad.min())
            opt.step()
            print("--after optimization mean:", atlas.mean())

            
        print('Total training loss:', total)
    
    print('Finished Training')
    #save final atlas
    # save_atlas(atlas, 'final_atlas.nii.gz')
    visualize_atlas(atlas)
    return atlas


def save_atlas(atlas, filename):
    
    atlas_np = atlas.squeeze().detach().cpu().numpy() 
    print("Atlas shape:", atlas_np.shape) 
    atlas_image = sitk.GetImageFromArray(atlas_np)
    # visualize_atlas(atlas)
    sitk.WriteImage(atlas_image, filename)


    
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
    # for ave_scan in trainloader:
    #     atlas, temp = ave_scan

    # Instead of getting the initial atlas from the training data, the initialization will be the average of the training data
    atlas = get_average_atlas(aveloader)
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
        slice_idx = atlas.shape[2] // 2
        plt.figure()
        plt.imshow(atlas[0,0,slice_idx].detach().cpu().numpy(), cmap='gray')
        plt.title(f"Epoch {epoch} - Max: {atlas.max():.3f}, Min: {atlas.min():.3f}")
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

        if epoch >= para.model.pretrain_epoch:
            opt.step()
            opt.zero_grad()

        print('Total training loss:', total)

    print('Finished Training')
    save_atlas(atlas, 'final_atlas.nii.gz')
    visualize_atlas(atlas)
    return atlas


def main():
    parser = argparse.ArgumentParser(description='Run Atlas Trainer')
    parser.add_argument('--json_file', type=str, required=True, help='Name of the JSON file')
    args = parser.parse_args()

    json_file = args.json_file
    dev = get_device()
    para = read_yaml('./parameters.yml')
    data_dir = '.'
    # json_file = 'train_json'
    keyword = 'train'
    # print(f'Running Atlas Trainer with JSON file: {json_file}')
    xDim, yDim, zDim= load_and_preprocess_data(data_dir, json_file, keyword)



    print (xDim, yDim, zDim)
    dataset = SData(json_file + '.json', 'train')
    ave_data = SData(json_file + '.json', 'train')
    trainloader = DataLoader(dataset, batch_size= para.solver.batch_size, shuffle=True)
    aveloader = DataLoader(ave_data, batch_size= 1 , shuffle = False)
    combined_loader = zip(trainloader, aveloader )
    net, criterion, optimizer = initialize_network_optimizer(xDim, yDim, zDim, para, dev)
    print (xDim, yDim, zDim)
    
    print("Training data loader length:", len(trainloader))
    print("Atlas data loader length:", len(aveloader))

    print(para.solver.epochs)

    #get the average atlas from the set of images
    get_average_atlas(aveloader)

    
    input("Press Enter to continue...")
    
    atlas = train_network(trainloader, aveloader, net, para, criterion, optimizer, NCC, 'l2', 10, 0.001, 16,16,16, xDim, yDim, zDim, dev)
    
    visualize_atlas(atlas)
    
    # overlay_atlas_and_image(atlas, image)
  
if __name__ == "__main__":
    main()








       
    
 
        


