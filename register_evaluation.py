
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

#-Parameters
parser = argparse.ArgumentParser(description='Atlas Registration')
parser.add_argument('--debug', action='store_true', help='Debug mode')
parser.add_argument('--net', type=str, default='pretrained_networks/net_epochs_100.pth', help='Path to the network')
args = parser.parse_args()

net_name = args.net
net_name = net_name.split('_')
epochs = net_name[-1].split('.')[0]

print("Evaluating network pre-trained for", epochs, "epochs")



#-Parameters (directly from parameters.yaml file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
para = read_yaml('parameters.yml')
# print(para)
xDim, yDim = 256, 256
reduced_xDim, reduced_yDim = 16,16
dev = device

#-Take the frozen model
net, criterion, optimizer = initialize_network_optimizer2D(xDim,yDim, para, dev)
net.load_state_dict(torch.load(args.net, map_location=device))
net.to(device)
net.eval()          #Evaluation mode

#-Load two images from the brain dataset
nifti_datadir = 'nirep/nifti/'

datahandler = dh(
                dataset_type='nifti',
                directory=nifti_datadir,
                size=256
                )

input_image = datahandler.get_image(0)
target_image = datahandler.get_image(1)

#-Convert the images to tensors
input_image = torch.tensor(input_image, dtype=torch.float32).to(device)
target_image = torch.tensor(target_image, dtype=torch.float32).to(device)

#-visualize
if args.debug:
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image.cpu().detach().numpy().squeeze(), cmap='gray')
    plt.title('Input Image')
    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy().squeeze(), cmap='gray')
    plt.title('Target Image')
    plt.show()

#-Register the images, sent them trough the network to get the phi^(-1) 

#We will use I1 as "atlas" and we will register it to compare with I2
I1 = input_image.unsqueeze(0)
I2 = target_image.unsqueeze(0)

if args.debug:
    print(I1.shape)
    print(I2.shape)


with torch.no_grad():
    #Take the deformation field (momentum) from the network
    _ , momentum, _ = net(I1, I2, registration=True)
    momentum = momentum.permute(0,3,1,2)

    print("Momentum mean:", momentum.mean().item())
    print("Momentum max:", momentum.max().item())
    print("Momentum min:", momentum.min().item())



#phi^(-1) = phi + momentum
img_size = xDim
identity = get_grid2D(img_size, dev).permute([0, 3, 2, 1])
epd = Epdiff2D(dev, (reduced_xDim, reduced_yDim), (xDim, yDim), para.solver.Alpha, para.solver.Gamma, para.solver.Lpow)

#assuming batch size 1 (testing only with a pair of images)
with torch.no_grad():
    v_fourier = epd.spatial2fourier(momentum[0].reshape(img_size, img_size, 2))
    velocity = epd.fourier2spatial(epd.Kcoeff * v_fourier).reshape(img_size, img_size, 2)
    # !! BOTTLENECK
    _, displacement = epd.forward_shooting_v_and_phiinv(velocity, para.solver.Euler_steps)
    # displace_ment + identity = phi_inv
    phi_inv = displacement.unsqueeze(0) + identity  # [1, H, W, 2]


#register the image
with torch.no_grad():
    registered_image = Torchinterp2D(I1, phi_inv)
    dist = criterion(registered_image, I2)
    print("Distance between registered image and target image:", dist.item())


# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(phi_inv[0, :, :, 0].cpu().detach().numpy(), cmap='coolwarm')
# plt.title('phi_inv X displacement')

# plt.subplot(1, 2, 2)
# plt.imshow(phi_inv[0, :, :, 1].cpu().detach().numpy(), cmap='coolwarm')
# plt.title('phi_inv Y displacement')

# plt.show()

registered_image_np = registered_image.cpu().detach().numpy().squeeze()
diff_image = np.abs(registered_image_np - I1.cpu().detach().numpy().squeeze())
diff_image_2 = np.abs(registered_image_np - I2.cpu().detach().numpy().squeeze())



# plt.figure(figsize=(10, 5))
# plt.imshow(diff_image, cmap='jet')
# plt.colorbar()
# plt.title("Diferencia entre imagen registrada e imagen original")
# plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(diff_image_2, cmap='jet')
plt.colorbar()
plt.title("Diferencia entre imagen registrada e imagen target")
img_name = 'diff_image_' + epochs + '.png'
plt.imsave(img_name, diff_image_2, cmap='jet')

plt.show()




#dump all the info in a json
info = {
    "net": args.net,
    "epochs": epochs,
    "momentum_mean": momentum.mean().item(),
    "momentum_max": momentum.max().item(),
    "momentum_min": momentum.min().item(),
    "distance": dist.item()
}

output_name = 'evaluation_' + epochs + '.json'
with open(output_name, 'w') as outfile:
    json.dump(info, outfile)

print("Evaluation results saved in", output_name)




# #-visualize results
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 3, 1)
# plt.imshow(I1.cpu().detach().numpy().squeeze(), cmap='gray')
# plt.title('Input Image')
# plt.subplot(1, 3, 2)
# plt.imshow(registered_image.cpu().detach().numpy().squeeze(), cmap='gray')
# plt.title('Registered Image')
# plt.subplot(1, 3, 3)
# plt.imshow(I2.cpu().detach().numpy().squeeze(), cmap='gray')
# plt.title('Target Image')
# plt.show()


