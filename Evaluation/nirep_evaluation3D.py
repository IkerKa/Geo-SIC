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



### Functions

#Evaluate the phi saved
import torch
import numpy as np
from scipy.io import loadmat
import torch.nn.functional as F

def random_training(net, optimizer, num_epochs, train_images, device, num_batches=5, batch_size=2):
    loss_total = 0
    loss_per_epoch = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        epoch_loss = 0

        for batch in range(num_batches):
            # Use logger instead of print to avoid overlapping outputs
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch+1}/{num_batches}", end='\r')
            batch_loss = 0
            for _ in range(batch_size):
                source_idx = np.random.randint(len(train_images))
                moving_idx = np.random.randint(len(train_images))
                source_image = train_images[source_idx].to(device)
                moving_image = train_images[moving_idx].to(device)

                y_src, momentum, _, _ = net(moving_image, source_image, registration=True, shooting="SVF", return_phi=True)
                Dist = config.NCC().loss(y_src, source_image)
                Reg = config.Grad(penalty='l2')
                Reg_loss = Reg.loss(momentum)
                loss = Dist + Reg_loss
                batch_loss += loss

            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += batch_loss.item()

        mean_loss = epoch_loss / num_batches
        loss_per_epoch.append(mean_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {mean_loss:.4f}")

    logger.info(f"Training completed. Total loss: {sum(loss_per_epoch):.4f}")
    logger.info(f"Mean loss per epoch: {np.mean(loss_per_epoch):.4f}")

    #Plot a graph of the loss per epoch
    _ , ax = plt.subplots()
    ax.plot(range(1, num_epochs + 1), loss_per_epoch, marker='o')
    ax.set_title('Training Loss per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)
    plt.show()

        
def selected_training(net, optimizer, num_epochs, train_images, device, num_batches=5, batch_size=2):
    loss_total = 0
    loss_per_epoch = []

    num_images = len(train_images)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        epoch_loss = 0

        for batch in range(num_batches):
            fixed_idx = batch % num_images      # We can select it randomly 
            fixed_image = train_images[fixed_idx].to(device)
            batch_loss = 0

            for _ in range(batch_size):
                # # Select a moving image different from the fixed image
                # moving_indices = [i for i in range(num_images) if i != fixed_idx]
                # Select a moving image (can be the same as the fixed image)
                moving_indices = [i for i in range(num_images)]
                moving_idx = np.random.choice(moving_indices)
                moving_image = train_images[moving_idx].to(device)

                y_src, momentum, _, _ = net(moving_image, fixed_image, registration=True, shooting="SVF", return_phi=True)
                Dist = config.NCC().loss(y_src, fixed_image)
                Reg = config.Grad(penalty='l2')
                Reg_loss = Reg.loss(momentum)
                loss = Dist + Reg_loss
                batch_loss += loss

            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += batch_loss.item()

        mean_loss = epoch_loss / num_batches
        loss_per_epoch.append(mean_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {mean_loss:.4f}")

    logger.info(f"Training completed. Total loss: {sum(loss_per_epoch):.4f}")
    logger.info(f"Mean loss per epoch: {np.mean(loss_per_epoch):.4f}")

    # Plot a graph of the loss per epoch
    _, ax = plt.subplots()
    ax.plot(range(1, num_epochs + 1), loss_per_epoch, marker='o')
    ax.set_title('Training Loss per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)
    plt.show()


def pad_to_multiple(tensor, multiple=16):
    # tensor shape: (1, 1, D, H, W)
    _, _, d, h, w = tensor.shape
    pad_d = (multiple - d % multiple) % multiple
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    # F.pad uses (W_left, W_right, H_left, H_right, D_left, D_right)
    padding = (0, pad_w, 0, pad_h, 0, pad_d)
    # print(f"Padding: {padding}")
    # print(f"Tensor shape before padding: {tensor.shape}")
    tensor_padded = F.pad(tensor, padding, mode='constant', value=0)
    # print(f"Tensor shape after padding: {tensor_padded.shape}")
    return tensor_padded

def convert_to_tensor(image):
    return torch.tensor(image, dtype=torch.float32)

def rescale_image(image):
    """Rescale image to [0, 1]"""
    return (image - image.min()) / (image.max() - image.min())

def load_all_data(dataset_path, dataset_name):
    """
    Load all the 3D images and segmentations from the specified dataset path.
    Assumes images are named NIREP_XX-Sub.mat and segmentations NIREP_XX-Seg.mat, with XX from 01 to 16.
    """
    data = {'images': [], 'segmentations': []}

    for i in range(1, 17):  # 01 to 16 inclusive
        img_path = f"{dataset_path}NIREP_{i:02d}-Sub.mat"
        seg_path = f"{dataset_path}NIREP_{i:02d}-Seg.mat"

        img_data = loadmat(img_path)['im']
        seg_data = loadmat(seg_path)['seg']

        data['images'].append(img_data)
        data['segmentations'].append(seg_data)

    return data


logger = Logger('NIREP_3D.log')

dataset_path = 'Baseline/NIREP_Matlab/'

logger.divider('Loading all the 3D images and segmentations')

data = load_all_data(dataset_path, 'NIREP_3D')

# Re-scale images to [0, 1]
data['images'] = [rescale_image(img) for img in data['images']]

# Convert images and segmentations to tensors
data_tensors = {
    'images': [convert_to_tensor(img) for img in data['images']],
    'segmentations': [convert_to_tensor(seg) for seg in data['segmentations']]
}


logger.info(f'Number of images loaded: {len(data["images"])}')


test_data = {'images': data_tensors['images'][:2], 'segmentations': data_tensors['segmentations'][:2]}
train_data = {'images': data_tensors['images'][2:12], 'segmentations': data_tensors['segmentations'][2:12]}
val_data = {'images': data_tensors['images'][12:14], 'segmentations': data_tensors['segmentations'][12:14]}

logger.info(f'Training set size: {len(train_data["images"])}')
logger.info(f'Validation set size: {len(val_data["images"])}')
logger.info(f'Test set size: {len(test_data["images"])}')

### Debug: Check the shapes of the tensors and plot the first image

# i = 0  # Index of the first training image
# img = train_data['images'][i].numpy()  # Convert tensor to numpy array
# logger.info(f'Train image {i} shape: {img.shape}')
# plt.imshow(img[:, :, 90], cmap='gray')
# plt.title('First Training Image Slice')
# plt.show()


### Image registration

# To register the images we will train with a set of images using N batches where for each batch we will have two options:
# 1. Use random pairs of images from the training set.
# 2. For each batch, take a fixed source and random moving ones

# Add batch dimensions !!!! CUIDADO CON ESTA PARTE !!!! (imagino que necesario para que la red funcione correctamente)
logger.info('Shape of images before padding: ' + str(train_data['images'][0].shape))
train_data['images'] = [img.unsqueeze(0).unsqueeze(0) for img in train_data['images']]
train_data['images'] = [pad_to_multiple(img) for img in train_data['images']]
logger.info('Shape of images after padding: ' + str(train_data['images'][0].shape))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
para = read_yaml('parameters.yml')

xDim, yDim, zDim = train_data['images'][0].shape[2:5]
net, _ , optimizer = initialize_network_optimizer(xDim, yDim, zDim, para, device)

net.train()
logger.info('Starting training with random pairs of images')
random_training(net, optimizer, num_epochs=100, train_images=train_data['images'], device=device)






