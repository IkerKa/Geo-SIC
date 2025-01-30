import io
from os import PathLike
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import os
import glob
import json
import subprocess
import sys
from PIL import Image, ImageDraw
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

import json
from classifiers import *
import lagomorph as lm 
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode

#https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/binary;tab=objects?inv=1&invt=AboKXg&prefix=&forceOnObjectsSortingFiltering=false

#The Sketch-RNN QuickDraw dataset does not store images directly.
#Instead, it stores stroke-based vector data, meaning that each sample consists of a sequence of pen strokes rather than a 28×28 pixel grid.



# Function to plot a stroke-based drawing --> Each stroke is a (dx, dy, pen_lift) triplet 
# where dx and dy are the change in x and y coordinates and pen_lift is a binary value indicating whether the pen is lifted or not.
# Function to plot a stroke-based drawing
def plot_sketch(stroke_data, idx = None, path=None):
    """Plots a single sketch from the dataset"""
    fig, ax = plt.subplots()
    x, y = 0, 0  # Starting position
    
    # Loop through each stroke (each stroke is a sequence of triplets)
    for stroke in stroke_data:
        stroke = stroke.reshape(-1, 3)
        # Unpack each stroke's triplets: (dx, dy, pen_lift)
        for triplet in stroke:
            dx, dy, pen_lift = triplet
            new_x, new_y = x + dx, y + dy
            if pen_lift == 0:  # If the pen is down, we draw a line
                ax.plot([x, new_x], [y, new_y], 'k', linewidth=2)
            x, y = new_x, new_y  # Update the current position
            
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # Invert Y-axis for correct orientation

    # Convert the plot to a numpy array
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = np.mean(image, axis=2).astype(np.uint8)  # Take the average of the RGB channels
    plt.close(fig)
    
    return image

def convert2DtoMHD(image, image_name):
    """_summary_

    Args:
        image (_type_): _description_
    """

    #from a PNG image of 2 dimensions, convert it into an MHD version where the third dimension is 1 (because it is a 2D image)
    # Convert the 2D image to a numpy array
    image_array = np.array(image)

    # Add a third dimension to the array
    # image_array = image_array[:, :, np.newaxis]
    image_array = np.expand_dims(image_array, axis=2)

    #repeat the image 100 times in the third dimension
    image_array = np.repeat(image_array, 10, axis=2)


    # Convert the numpy array to a SimpleITK image
    itk_image = sitk.GetImageFromArray(image_array)

    itk_image.SetSpacing([1.0, 1.0, 1.0])

    print(f"Image shape: {itk_image.GetSize()}")

    # Save the image as an MHD file
    path = './datasets/triangle_sketches'
    sitk.WriteImage(itk_image, os.path.join(path, f'{image_name}.mhd'))

    #print the dimensions of the image




def main():

    # Load the dataset
    data_path = 'datasets/triangles.full.npz'
    data = np.load(data_path, encoding='latin1', allow_pickle=True)

    # Extract training data (strokes)
    train_data = data['train']
    print(f"Number of sketches: {len(train_data)}")

    # With that dataset, create a small one to use for training in atlas2D
    # We will use the triangles dataset
    # We will use the first 10 triangles

    # Create a new dataset with the first 10 triangles
    new_data = train_data[:10]



    #see the images and store them to make a dataset
    #triangles dir
    os.makedirs("datasets/triangle_sketches", exist_ok=True)
    for i in range(10):
        image = plot_sketch(new_data[i], i, path="datasets/triangle_sketches")
        convert2DtoMHD(image, "triangle_"+str(i))



if __name__ == '__main__':
    main()