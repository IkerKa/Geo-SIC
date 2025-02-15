


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
import nibabel as nib #type: ignore
import random 
import yaml
from losses import NCC, MSE, Grad
from networks import UnetDense  
from SitkDataSet import SitkDataset as SData
from uEpdiff import Epdiff
from networks import *
import argparse
import matplotlib.pyplot as plt # type: ignore
from datasets.datasetloader import GoogleDrawDataset2d, DataLoaderHandler
import SimpleITK as sitk # type: ignore


torch.cuda.empty_cache()

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
    

def overlay_atlas_and_image(atlas, image, is_2d=False):
    # Convierte los tensores a arrays de NumPy
    atlas_np = atlas.squeeze().detach().cpu().numpy()
    image_np = image.squeeze().cpu().numpy()

    if is_2d:
        # Para imágenes 2D, simplemente superpone las imágenes
        plt.imshow(image_np, cmap='gray', alpha=0.5)  # Imagen de entrenamiento (semitransparente)
        plt.imshow(atlas_np, cmap='hot', alpha=0.5)   # Atlas (semitransparente)
    else:
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

def train_network2D(trainloader, aveloader, net, para, criterion, optimizer, DistType, RegularityType, weight_dist, weight_reg,  reduced_xDim, reduced_yDim, xDim, yDim, dev):
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

    # Get an initialization of the atlas
    for ave_scan in trainloader:
        atlas, temp = ave_scan
        break;

    # Instead of getting the initial atlas from the training data, the initialization will be the average of the training data
    atlas.requires_grad=True
    opt = optim.Adam([atlas], lr=para.solver.atlas_lr)  #<---they used a different optimizer for the atlas


    print("Training setup:")
    print("-----------------------------------")
    print("Atlas shape:", atlas.shape)
    print("Number of epochs:", para.solver.epochs)
    print("Batch size:", para.solver.batch_size)
    print("-----------------------------------")
    for epoch in range(para.solver.epochs):
        net.train()
        print("Computing epoch:", epoch, ". Current loss:", total)
        for j, tar_bch in enumerate(trainloader):
            print("Computing batch:", j)
            #-take the dimensions of the batch for 2D images.
            b, c, w, h = tar_bch[0].shape
            #-restart the optimizer gradient
            optimizer.zero_grad()
            #-initialize the phiinv and reg_save tensors
            phiinv_bch = torch.zeros(b, w, h, 2).to(dev)
            reg_save = torch.zeros(b, w, h, 2).to(dev)

            # Now we wont pretrain the atlas building network 
            atlas_bch = torch.cat(b*[atlas]).reshape(b, c, w, h)
            atlas_bch = atlas_bch.to(dev).float()
            tar_bch_img = tar_bch[0].to(dev).float()

            #pass the atlas and the target image to the network
            _ , momentum, latent_feat  = net(atlas_bch, tar_bch_img, registration=True)
            momentum = momentum.permute(0, 3, 2, 1) # ? ARE THE SIZES CORRECT?
            
            #MATHS things
            identity = get_grid2D(w, h, dev).permute([0, 3, 2, 1])
            epd = Epdiff(dev, (reduced_xDim, reduced_yDim), (xDim, yDim), para.solver.Alpha, para.solver.Gamma, para.solver.Lpow)

            for b_id in range(b):   #adapted to 2D images
                v_fourier = epd.spatial2fourier(momentum[b_id,...].reshape(w, h, 2))
                velocity = epd.fourier2spatial(epd.Kcoeff * v_fourier).reshape(w, h, 2)  
                reg_temp = epd.fourier2spatial(epd.Lcoeff * v_fourier * v_fourier)
                num_steps = para.solver.Euler_steps
                v_seq, displacement = epd.forward_shooting_v_and_phiinv(velocity, num_steps)  
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
            total += running_loss
            running_loss = 0.0

            #debug information after epoch
            
            print(f"--Atlas gradient, max: {atlas.grad.max()}, min: {atlas.grad.min()}")
            print(f"--RUNNING LOSS: {running_loss}")



        opt.step()
        opt.zero_grad() 
    
    print('Finished Training')
    save_atlas(atlas, 'final_atlas.nii.gz')

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
    parser = argparse.ArgumentParser(description='Run Atlas Trainer')
    parser.add_argument('--json_file', type=str, required=True, help='Name of the JSON file')
    parser.add_argument('--two_dims', type=bool, default=True, help='Whether to use 2D images')
    args = parser.parse_args()

    json_file = args.json_file
    dev = get_device()
    para = read_yaml('./parameters.yml')
    data_dir = '.'
    # json_file = 'train_json'
    keyword = 'train'
    # print(f'Running Atlas Trainer with JSON file: {json_file}')

    if args.two_dims:
        datadir = 'datasets/jsons/circle.ndjson'
        #load the ndjson file and get the dimensions of the image
        dataset = GoogleDrawDataset2d(datadir, samples=5)
        trainloader = DataLoader(dataset, batch_size=16, shuffle=True)
        aveloader = DataLoader(dataset, batch_size=1, shuffle=False)

        datahandler = DataLoaderHandler(ndjson_file=datadir, samples=5, resize=128, batch_size=16)
        datahandler.show_example()

        #obtain the dimensions of the image
        xDim, yDim = 128, 128
    
        combined_loader = zip(trainloader, aveloader )  # ? Why do we need this combined_loader?
        net, criterion, optimizer = initialize_network_optimizer2D(xDim, yDim, para, dev)
        print("2D image dimensions:", xDim, yDim)

    else:
        xDim, yDim, zDim = load_and_preprocess_data(data_dir, json_file, keyword)
        dataset = SData(json_file + '.json', 'train')
        ave_data = SData(json_file + '.json', 'train')
        trainloader = DataLoader(dataset, batch_size= para.solver.batch_size, shuffle=True)
        aveloader = DataLoader(ave_data, batch_size= 1 , shuffle = False)
        combined_loader = zip(trainloader, aveloader )
        net, criterion, optimizer = initialize_network_optimizer(xDim, yDim, zDim, para, dev)
        print("3D image dimensions:", xDim, yDim, zDim)
    
    for batch in trainloader:
        print("Tamaño del batch:", batch[0].shape)
        break

    #plot the average atlas
    get_average_atlas(aveloader, _debug=True)
    

    if args.two_dims:
        atlas = train_network2D(trainloader, aveloader, net, para, criterion, optimizer, NCC, 'l2', 10, 0.001, 16,16, xDim, yDim, dev)
    else:
        atlas = train_network(trainloader, aveloader, net, para, criterion, optimizer, NCC, 'l2', 10, 0.001, 16,16,16, xDim, yDim, zDim, dev)
    
    
    
    # overlay_atlas_and_image(atlas, image)
  
if __name__ == "__main__":
    main()








       
    
 
        


