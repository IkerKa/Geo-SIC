
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


print("\n\n\n\n\n\n\n\n\n\n")
#-Parameters
parser = argparse.ArgumentParser(description='Atlas Registration')
parser.add_argument('--debug', action='store_true', help='Debug mode')
# parser.add_argument('--net', type=str, default='pretrained_networks/net_epochs_100.pth', help='Path to the network')
args = parser.parse_args()

# net_name = args.net
# net_name = net_name.split('_')
# epochs = net_name[-1].split('.')[0]

# print("Evaluating network pre-trained for", epochs, "epochs")



#-Parameters (directly from parameters.yaml file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
para = read_yaml('parameters.yml')
# print(para)
xDim, yDim = 128, 128
reduced_xDim, reduced_yDim = 16,16
dev = device

#-Take the frozen model
net, criterion, optimizer = initialize_network_optimizer2D(xDim,yDim, para, dev)
# net.load_state_dict(torch.load(args.net, map_location=device))
# net.to(device)

nifti_datadir = 'nirep/nifti/'

datahandler = dh(
                dataset_type='nifti',
                directory=nifti_datadir,
                size=128
                )

input_image = datahandler.get_image(0)
target_image = datahandler.get_image(1)

#-Convert the images to tensors
input_image = torch.tensor(input_image, dtype=torch.float32).to(device)
target_image = torch.tensor(target_image, dtype=torch.float32).to(device)



I1 = input_image.unsqueeze(0)
I2 = target_image.unsqueeze(0)

#-Evaluate the network (forward pass and save the new_locs betwen iterations)
# number_epochs = para.solver.pre_train
number_epochs = 25

#-Initialize the optimizer
optimizer = optim.Adam(net.parameters(), lr=para.solver.lr)
total_init = time.time()
shooting_flag = 'SVF'
print('Pre-training for', number_epochs, 'epochs')
loss_total = 0
phi_inv = None

init_momentum = torch.zeros_like(I1)
final_momentum = torch.zeros_like(I1)

for epoch in range(number_epochs):
    net.eval()
    optimizer.zero_grad()
    

    #Stationary velocity field means v(x,t) = v(x)
    #what this means that the differential equation is the same at all times t
    # Math: \partial{\phi(x,t)}/\partial{t} = (v(\phi(x,t),t)
    y_src , momentum, latent_feat, new_locs = net(I1, I2, registration=True, shooting=shooting_flag, return_phi = True)

    #momentum represente la transformacion final para llevar I1 a I2

    if epoch == 0:
        init_momentum = momentum
    if epoch == number_epochs-1:
        final_momentum = momentum


    Dist = NCC().loss(y_src, I2)
    Reg = Grad(penalty='l2')  
    Reg_loss = Reg.loss2D(momentum)

    loss_total = Dist + Reg_loss
    loss_total.backward()

    optimizer.step()
    # print('Epoch:', epoch, 'Loss:', loss_total.item())

    with torch.no_grad():
        print(f'Shape of new_locs: {new_locs.shape}')
        phi_inv = new_locs[0, ...]
        print(f'Shape of phi_inv: {phi_inv.shape}')

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(init_momentum[0, 0].cpu().detach().numpy(), cmap='jet')
ax[0].set_title('Initial Momentum')

ax[1].imshow(final_momentum[0, 0].cpu().detach().numpy(), cmap='jet')
ax[1].set_title('Final Momentum')

plt.show()



fig, ax = plt.subplots()
print(f'Shape of phi_inv: {phi_inv.shape}')
interval = 2

# Uncomment the following line if you want to show I1
# ax.imshow(I1[0].cpu().detach().numpy().squeeze(), cmap='gray')

# Iterate over the rows and columns with the specified interval
for row in range(0, phi_inv.shape[0], interval):
    ax.plot(phi_inv[row, :, 1].cpu().detach().numpy(), phi_inv[row, :, 0].cpu().detach().numpy(), 'm')
for col in range(0, phi_inv.shape[1], interval):
    ax.plot(phi_inv[:, col, 1].cpu().detach().numpy(), phi_inv[:, col, 0].cpu().detach().numpy(), 'm')

plt.title("Diffeomorphic deformation grid")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(I1.squeeze().cpu().detach().numpy(), cmap='gray')
axes[0].set_title('Source Image (I1)')

axes[1].imshow(I2.squeeze().cpu().detach().numpy(), cmap='gray')
axes[1].set_title('Target Image (I2)')

axes[2].imshow(y_src.squeeze().cpu().detach().numpy(), cmap='gray')
axes[2].set_title('Registered Image (y_src)')

plt.show()

error_map = np.abs(I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy())

plt.imshow(error_map, cmap='hot')
plt.colorbar()
plt.title("Error Map")
plt.show()

# Alpha blending to overlay I1 and y_src on I2
alpha = 0.5

fig, ax = plt.figure(), plt.axes()
# Display blended image of I2 and y_src
blended_image = alpha * I2.squeeze().cpu().detach().numpy() + (1 - alpha) * y_src.squeeze().cpu().detach().numpy()
plt.imshow(blended_image, cmap='gray')
plt.title("Target Image (I2) and Registered Image (y_src) Overlayed")

plt.show()

#metrics
ssim_score = ssim(I2.squeeze().cpu().detach().numpy(), y_src.squeeze().cpu().detach().numpy(), data_range=y_src.squeeze().cpu().detach().numpy().max() - y_src.squeeze().cpu().detach().numpy().min())
print(f'SSIM Score: {ssim_score}')



#-Preguntas:
# 1. Tengo alguna manera de observar los v(x,t) en cada iteracion? para ver si al usar SVF no cambian.
# 2. El momentum es lo que transforma la imagen I1 a I2
# 3. Otra manera de evaluar el regsitro es usar NODEO?