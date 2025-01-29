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
    # plt.show()

    #save the image to a file
    if idx is not None:
        # plt.savefig(f"{path}/sketch_{idx}.png")
        # Convert the plot to a PIL image and save it
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img.save(f"{path}/sketch_{idx}.png")
        buf.close()




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
        plot_sketch(new_data[i], i, path="datasets/triangle_sketches")



if __name__ == '__main__':
    main()