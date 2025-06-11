import config  # Este es el archivo que contiene todas las importaciones
from evaluate_validation_test import exhaustive_train_model_with_validation, compute_segmentation, SpatialTransformer, compute_dice
from Run_Atlas_trainer import initialize_network_optimizer, read_yaml

import torch
import torch.nn.functional as F
import numpy as np

from networks import UnetDense
from matplotlib import pyplot as plt

# New imports
from scipy.io import loadmat
from logger import Logger
from scipy.io import savemat
from skimage.transform import resize



### Functions

#Evaluate the phi saved
import torch
import numpy as np
from scipy.io import loadmat
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data_imc import NiftiMatDataset

def get_device():
    """Returns the device available (cuda or cpu)"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

dev = get_device()

# Load dataset
dataset_path = 'Baseline/NIREP_Matlab/'
dataset = NiftiMatDataset(dataset_path)

# Take reference 1 - 2 pair
nirep01, _ = dataset.__getitem__(0)
nirep02, _ = dataset.__getitem__(1)

nirep01 = nirep01.to(dev).float()
nirep02 = nirep02.to(dev).float()

nirep01 = nirep01.unsqueeze(0)  # Add batch dimension
nirep02 = nirep02.unsqueeze(0)  # Add batch dimension

print("NIREP01 shape:", nirep01.shape)
print("NIREP02 shape:", nirep02.shape)

mse0 = ((nirep01 - nirep02) ** 2).mean()

print("MSE between NIREP01 and NIREP02:", mse0.item())

# Plot the two images (DEBUG)

# Plot the axial slice at index 90 for both images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(nirep01[0, 0, :, 90, :].cpu().numpy(), cmap='gray')
plt.title('NIREP01 - Axial 90')
plt.subplot(1, 2, 2)
plt.imshow(nirep02[0, 0, :, 90, :].cpu().numpy(), cmap='gray')
plt.title('NIREP02 - Axial 90')
plt.show()


def read_yaml(path):
    """Reads a YAML file and returns its contents as a dictionary."""
    try:
        with open(path, 'r') as f:
            file = config.edict(config.yaml.load(f, Loader=config.yaml.FullLoader))
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
            data = config.json.load(f)
    except Exception as e:
        print(f'Error loading JSON data: {e}')
        return None
    outputs = []
    temp_scan = config.sitk.GetArrayFromImage(config.sitk.ReadImage(f'{data_dir}/{data[keyword][0]["image"]}'))
    xDim, yDim, zDim = temp_scan.shape
    return xDim, yDim, zDim


def initialize_network_optimizer(xDim, yDim, zDim, para, dev):
    """
    Initializes the atlas building neural network, classifier, loss functions, optimizer, and scheduler.
    Returns the initialized objects.
    """
    # Initialize the atlas building network (UnetDense)
    net = UnetDense(inshape=(xDim, yDim, zDim),
                      nb_unet_features=[[16, 32, 32], [ 32, 32, 32, 16, 16]], #[16, 32,32], [ 32, 32, 32, 16, 16]
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
    conv_kernel_sizes = [3, 3, 3, 3, 3]  # Kernel sizes for each convolutional layer
    activation = 'ReLU'  # Activation function
    num_classes = 2 # Number of classes
    clfer = config.Flexi3DCNN(in_channels, conv_channels, conv_kernel_sizes, num_classes, activation)
    clfer = clfer.to(dev)

    # Combine parameters for optimization
    params = list(net.parameters()) + list(clfer.parameters())

    # Initialize loss functions
    criterion_clf = config.nn.CrossEntropyLoss()
    if para.model.loss == 'L2':
        criterion = config.nn.MSELoss()
    elif para.model.loss == 'L1':
        criterion = config.nn.L1Loss()

    # Initialize optimizer
    if para.model.optimizer == 'Adam':
        optimizer = config.optim.Adam(params, lr=para.solver.lr)
    elif para.model.optimizer == 'SGD':
        optimizer = config.optim.SGD(params, lr=para.solver.lr, momentum=0.9)

    # Initialize scheduler (CosineAnnealingLR)
    scheduler = config.CosineAnnealingLR(optimizer, T_max=para.solver.epochs)

    return net, clfer, criterion, criterion_clf, num_classes, optimizer, scheduler

def train_network(trainloader, aveloader, net, clfer, para, criterion, criterion_clf, num_classes, optimizer, scheduler, DistType, RegularityType, weight_dist, weight_reg, weight_latent, reduced_xDim, reduced_yDim, reduced_zDim, xDim, yDim, zDim, dev, flag):
    """
    Trains the atlas building neural network and classifier.
    """
    running_loss = 0
    total = 0

    total_rmse = []

    for epoch in range(para.solver.epochs):

        net.train()
        clfer.train()
        print('epoch:', epoch)

        batch = next(iter(trainloader))
        atlas_bch, temp = batch

        for j, tar_bch in enumerate(trainloader):

            if torch.equal(atlas_bch, tar_bch[0]):
                continue

            print(f'Batch {j+1} / {len(trainloader)} ', end='\r')

            optimizer.zero_grad()


            atlas_bch = atlas_bch.to(dev).float() 
            tar_bch_img = tar_bch[0].to(dev).float()

            # Train atlas building with extracted latent features
            pred = net(atlas_bch, tar_bch_img, registration=True, shooting = flag) 

            # Train image classifier with feature fusion strategy using a specified weighting parameter,
            # this network will not be updated unless the atlas building is pretrained
            cl_pred = clfer (tar_bch_img ,pred[2], weight_latent)

            # Create a tensor from the ground truth label, one-hot for multi-classes
            tar_bch_lbl = F.one_hot(torch.tensor(int(tar_bch[1][0])), num_classes).to(dev).float()
            clf_loss = criterion_clf(cl_pred[0], tar_bch_lbl)
            
            if (flag == "SVF"): # Stationary velocity fields to shoot forward 
                # print (pred[1].shape)
                Dist = config.NCC().loss(pred[0], tar_bch_img)   
                Reg = config.Grad( penalty= RegularityType)
                Reg_loss  = Reg.loss(pred[1])
                if epoch <= para.model.pretrain_epoch:
                    loss_total = weight_dist * Dist + weight_reg * Reg_loss 
                else:
                    loss_total = weight_dist * Dist + weight_reg * Reg_loss + clf_loss


            loss_total.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss_total.item()
            total += running_loss
            running_loss = 0.0

        scheduler.step()  # Update learning rate

        print('Total training loss:', total)

        # rMSE for the 1-2 reference pair

        pred = net(nirep01, nirep02, registration=True, shooting=flag)
        mse = ((pred[0] - nirep02) ** 2).mean()

        print( "rMSE in baseline pair 1 - 2", (mse / mse0).item() )

        total_rmse.append((mse / mse0).item())

    # create sampling grid
    vectors = [torch.arange(0, s) for s in nirep01.shape[2:]]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor)

    savemat( 'results.mat', {
            'nirep01': nirep01.detach().cpu().numpy(),
            'nirep02': nirep02.detach().cpu().numpy(),
            'warped': pred[0].detach().cpu().numpy(),
            'rmse': total_rmse,
            'id': grid,
            'disp': pred[1].detach().cpu().numpy()
            })


def main():
    """
    Main function to run the training process.
    """
    dev = get_device()
    para = read_yaml('./parameters.yml')

    trainloader = DataLoader(dataset, batch_size=para.solver.batch_size, shuffle=True)
    aveloader = DataLoader(dataset, batch_size=1, shuffle=False)
    combined_loader = zip(trainloader, aveloader)

    batch = next(iter(trainloader))
    inputs, labels = batch

    _, _, xDim, yDim, zDim = inputs.shape

    net, clfer, criterion, criterion_clf, num_classes, optimizer, scheduler = initialize_network_optimizer(xDim, yDim, zDim, para, dev)

    train_network(trainloader, aveloader, net, clfer, para, criterion, criterion_clf, num_classes, optimizer, scheduler, config.NCC, 'l2', 0.5, 0.5, 0.2, 16, 16, 16, xDim, yDim, zDim, dev, "SVF")

if __name__ == "__main__":
    main()

