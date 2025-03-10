
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
#animation
import matplotlib.animation as animation
import matplotlib.pyplot as plt # type: ignore
# from datasets.datasetloader import GoogleDrawDataset2d, DataLoaderHandler
# from datasets.datasetloader3d import MHD2DDataset
# from datasets.datasetloader3d import DataLoaderHandler as d3d
# from datasets.createDataset import DataLoaderHandler as d2d
# from datasets.createDataset import ImageTransformDataset
# from datasets.shapedsloader import ShapesDataLoaderHandler as sdh
from datasets.datasethandler import DataHandler as dh
import SimpleITK as sitk # type: ignore

from Run_Atlas_trainer import initialize_network_optimizer2D, read_yaml

from skimage.metrics import structural_similarity as ssim # type: ignore

#debug argument
import pdb

#ignore warnigns flag!
import warnings
warnings.filterwarnings("ignore")

# pretrain_epochs_list = [5, 10, 20, 50, 100, 200, 300, 450, 600, 800, 1000]
# pretrain_epochs_list = [5, 10, 15]
pretrain_epochs_list = [50, 100, 200]
list_len = len(pretrain_epochs_list)

#TODO: Test also different learning rates / parameters 

phi_results = []
ssim_results = []
loss_results = []
y_srcs = []

results = []

# -Parameters
parser = argparse.ArgumentParser(description='Atlas Registration')
parser.add_argument('--debug', action='store_true', help='Debug mode')
# parser.add_argument('--net', type=str, default='pretrained_networks/net_epochs_100.pth', help='Path to the network')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load parameters from the YAML file
para = read_yaml("parameters.yml")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

xDim, yDim = 128, 128
reduced_xDim, reduced_yDim = 16,16
shooting_flag = 'SVF'


nifti_datadir = 'nirep/nifti/'

datahandler = dh(
                dataset_type='nifti',
                directory=nifti_datadir,
                size=128
                )

input_image = datahandler.get_image(0)
target_image = datahandler.get_image(1)

#-Convert the images to tensors
I1 = torch.tensor(input_image, dtype=torch.float32).to(device)
I2 = torch.tensor(target_image, dtype=torch.float32).to(device)

I1 = I1.unsqueeze(0)
I2 = I2.unsqueeze(0)




print(f'Starting evaluation of pre-trained networks for {pretrain_epochs_list} epochs')
for pretr in pretrain_epochs_list:
    
    # Reinitialize network and optimizer
        net, criterion, optimizer = initialize_network_optimizer2D(xDim, yDim, para, dev)
        optimizer = optim.Adam(net.parameters(), lr=para.solver.lr)
        
        # Variables to track the best performance
        total_loss = 0
        ssim_score = 0

        phi_inv = None

        ssim_per_epoch = []
        loss_per_epoch = []

        for epoch in range(pretr):
            net.eval()
            optimizer.zero_grad()
            
            # Forward pass
            y_src, momentum, latent_feat, new_locs = net(I1, I2, registration=True, shooting=shooting_flag, return_phi=True)
            
            # Calculate loss
            Dist = NCC().loss(y_src, I2)
            Reg = Grad(penalty='l2')  
            Reg_loss = Reg.loss2D(momentum)
            loss_total = Dist + Reg_loss
            total_loss += loss_total.item()

            loss_per_epoch.append(loss_total.item())
            I2_np = I2.squeeze().cpu().detach().numpy()
            y_src_np = y_src.squeeze().cpu().detach().numpy()
            ssim_score = ssim(I2_np, y_src_np, data_range=y_src_np.max() - y_src_np.min())
            ssim_per_epoch.append(ssim_score)

            
            # Backpropagation
            loss_total.backward()
            optimizer.step()

            if epoch == pretr-1:
                phi_inv = new_locs[0,...]

            

        # Calculate SSIM after training for this configuration
        y_src_np = y_src.squeeze().cpu().detach().numpy()
        I2_np = I2.squeeze().cpu().detach().numpy()
        ssim_score = ssim(I2_np, y_src_np, data_range=y_src_np.max() - y_src_np.min())
        
        # Store results
        result_entry = {
            'epochs': pretr,
            'ssim_score': ssim_score,
            'total_loss': total_loss
            # 'ssim_per_epoch': ssim_per_epoch,
            # 'loss_per_epoch': loss_per_epoch

        }


        y_srcs.append(y_src)
        phi_results.append(phi_inv)
        ssim_results.append(ssim_per_epoch)
        loss_results.append(loss_per_epoch)

        results.append(result_entry)
        
        print(f"Finished Grid Search with {pretr} epochs")
        print(f"Total Loss: {total_loss}, SSIM: {ssim_score}")

# Save results to a JSON file
with open("grid_search_results.json", "w") as outfile:
    json.dump(results, outfile, indent=4)

# Optionally, you can also print or plot the results
print("\nGrid Search Results:")
for result in results:
    print(result)



# fig, ax = plt.subplots(figsize=(10, 10))

# def update_frame(frame):
#     ax.clear()  
#     ax.imshow(y_srcs[frame].squeeze().cpu().detach().numpy(), cmap='gray')
#     ax.set_title(f"Epoch {frame + 1} - Registered Image")
#     ax.axis('off')  


# ani_y_srcs = animation.FuncAnimation(fig, update_frame, frames=len(y_srcs), interval=500)

# plt.show()


# fig, ax = plt.subplots()

# interval = 2

# def update_grid(frame):
#     ax.clear() 
    
#     phi = phi_results[frame]
    
#     for row in range(0, phi.shape[0], interval):
#         ax.plot(phi[row, :, 1].cpu().detach().numpy(), phi[row, :, 0].cpu().detach().numpy(), 'm', alpha=0.5)  # Filas
    
#     for col in range(0, phi.shape[1], interval):
#         ax.plot(phi[:, col, 1].cpu().detach().numpy(), phi[:, col, 0].cpu().detach().numpy(), 'm', alpha=0.5)  # Columnas
    
#     ax.set_title(f"Epoch {pretrain_epochs_list[frame]} - Deformation Grid")
#     ax.axis('off')  


# ani_grid = animation.FuncAnimation(fig, update_grid, frames=len(phi_results), interval=500)


# plt.show()

#TODO: Comparar cada imagen, mostrarla y sacar errors por ejemplo primera vs ultima.

#SSIM 
fig, ax1 = plt.subplots(figsize=(10, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(pretrain_epochs_list)))
color_ssim = 'tab:blue'
for i, pretr in enumerate(pretrain_epochs_list):
    ax1.plot(range(pretr), ssim_results[i], label=f'{pretr} epochs (SSIM)', color=colors[i])
ax1.tick_params(axis='y', labelcolor=color_ssim)
ax1.legend(loc='upper left')

# Comparison between I2 and the last registered image
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].imshow(I2.squeeze().cpu().detach().numpy(), cmap='gray')
axes[0].set_title('Target Image (I2)')
axes[0].axis('off')

axes[1].imshow(y_srcs[-1].squeeze().cpu().detach().numpy(), cmap='gray')
axes[1].set_title('Registered Image (y_src)')
axes[1].axis('off')

plt.show()


#Error between last registered image and target image
error_map = np.abs(I2.squeeze().cpu().detach().numpy() - y_srcs[-1].squeeze().cpu().detach().numpy())

plt.imshow(error_map, cmap='hot')
plt.colorbar()
plt.title("Error Map")
plt.axis('off')

plt.show()

# plt.hist(error_map.flatten(), bins=50, color='blue', alpha=0.7)
# plt.title('Histograma de Errores de Registro')
# plt.xlabel('Valor del Error')
# plt.ylabel('Frecuencia')
# plt.show()