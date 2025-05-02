
import config  # Este es el archivo que contiene todas las importaciones
from evaluate_validation_test import load_all_data, convert_to_tensor, exhaustive_train_model_with_validation, compute_segmentation, SpatialTransformer, compute_dice
from Run_Atlas_trainer import initialize_network_optimizer2D, read_yaml

import torch
import torch.nn.functional as F
import numpy as np

from networks import UnetDense

# New imports
from scipy.io import loadmat
from logger import Logger


### Functions
def custom_training(net, optimizer, num_epochs, fixed, moving, device):
    loss_total = 0
    phi_inv = None
    ssim_per_epoch, loss_per_epoch = [], []
    
    tqdm_epochs = config.tqdm.tqdm(total=num_epochs, desc="Training Progress", leave=False)
    for epoch in range(num_epochs):
        tqdm_epochs.update(1)
        tqdm_epochs.set_description(f'Epoch {epoch + 1}/{num_epochs}')

        pair_loss = []
        pair_ssim = []


        I1 = moving
        I2 = fixed

        I1, I2 = I1.to(device).float(), I2.to(device).float()

        # print(f"Source shape: {I1.shape}, Target shape: {I2.shape}")
        y_src, momentum,  _, new_locs = net(I1, I2, registration=True, shooting="SVF", return_phi=True)


        Dist = config.NCC().loss(I2, y_src)
        Reg = config.Grad(penalty='l2')
        Reg_loss = Reg.loss2D(momentum)

        loss_total = (1 * Dist + 0.01 * Reg_loss)
        loss_total.backward()

        optimizer.step()                        # Update model parameters
        optimizer.zero_grad()                   # Reset gradients
        pair_loss.append(loss_total.item())
        pair_ssim.append(config.ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(),
                                data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min()))

        with torch.no_grad():
            phi_inv = new_locs[0,...]
       
        mean_loss = config.np.mean(pair_loss)
        mean_ssim = config.np.mean(pair_ssim)
        loss_per_epoch.append(mean_loss)
        ssim_per_epoch.append(mean_ssim)


        tqdm_epochs.set_postfix(loss=mean_loss, ssim=mean_ssim)

def pad_to_multiple(tensor, multiple=16):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    # Pad at the bottom and right only (to avoid shifting content)
    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    return F.pad(tensor, padding, mode='constant', value=0)

def net_test_model(net, fixed, moving, fixed_seg, moving_seg, device):

    ssims = []
    rmses = []
    phis = []

    with torch.no_grad():
        
        I1 = moving
        I2 = fixed
        I1_seg = moving_seg
        I2_seg = fixed_seg

        I1, I2 = I1.to(device).float(), I2.to(device).float()
        I1_seg, I2_seg = I1_seg.to(device).float(), I2_seg.to(device).float()

        y_src, _, _, new_locs = net(I1, I2, registration=True, shooting='SVF', return_phi=True)
        warped_seg, _, _, dice_median = CustomSegmentation(I1_seg, new_locs[0, ...], I2_seg, device)

        phis.append(new_locs[0,...])

        ssim_score = config.ssim(
            y_src.squeeze().cpu().detach().numpy(),
            I2.squeeze().cpu().detach().numpy(),
            data_range=I2.max().item() - I2.min().item()
        )
        rmse_score = config.np.sqrt(config.np.mean((I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()) ** 2))

        ssims.append(ssim_score)
        rmses.append(rmse_score)
        print(f'SSIM: {ssim_score:.4f}, RMSE: {rmse_score:.4f}, Dice: {dice_median:.4f}')

    print(f'\nAverage - SSIM: {config.np.mean(ssims):.4f}, RMSE: {config.np.mean(rmses):.4f}')

 

    fig, axes = config.plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(I1.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[0].set_title('Source Image (I1)')
    axes[1].imshow(I2.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[1].set_title('Target Image (I2)')
    axes[2].imshow(y_src.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[2].set_title('Registered Image (y_src)')
    error = I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()
    axes[3].imshow(error, cmap='gray')
    axes[3].set_title('Error (I2 - y_src)')

    # Plotting the deformation grid for phi_inv
    ax = axes[4]
    interval = 2
    phi_inv = phis[0].cpu().detach().numpy()

    for row in range(0, phi_inv.shape[0], interval):
        ax.plot(phi_inv[row, :, 0],
            -phi_inv[row, :, 1],  # Flip in X axis
            'm')

    for col in range(0, phi_inv.shape[1], interval):
        ax.plot(phi_inv[:, col, 0],
            -phi_inv[:, col, 1],  # Flip in X axis
            'm')

    ax.set_title("Diffeomorphic deformation grid")
    config.plt.tight_layout()
    config.plt.show()

    # Same with the segmentation
    fig, axes = config.plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(I1_seg.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[0].set_title('Source Segmentation (I1_seg)')
    axes[1].imshow(I2_seg.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[1].set_title('Target Segmentation (I2_seg)')
    axes[2].imshow(warped_seg, cmap='gray')
    axes[2].set_title('Warped Segmentation (I1_seg)')
    error_seg = I2_seg.squeeze().cpu().detach().numpy() - warped_seg
    axes[3].imshow(error_seg, cmap='gray')
    axes[3].set_title('Error Segmentation (I2_seg - warped_seg)')

    

    config.plt.tight_layout()
    config.plt.show()


def CustomSegmentation(I1_seg, phi_inv, I2_seg, dev):

    if phi_inv.dim() == 3:
        phi_inv = phi_inv.unsqueeze(0) 

    phi_resized = F.interpolate(
        phi_inv.permute(0, 3, 1, 2),
        size=I1_seg.shape[-2:], 
        mode='bilinear', 
        align_corners=True
    )
    phi_resized = phi_resized.permute(0, 1, 2, 3) # B, C, H, W because grid of transformer is B, H, W, C

    print(f"phi_resized shape: {phi_resized.shape}")
    
    spat_trans = SpatialTransformer(size=I1_seg.shape[2:], mode='nearest').to(dev)

    warped_seg = spat_trans(I1_seg, phi_resized)

    warped_seg_np = warped_seg.squeeze().cpu().detach().numpy()
    fixed_seg_np = I2_seg.squeeze().cpu().detach().numpy()

    labels = np.unique(fixed_seg_np)
    labels = labels[labels != 0]

    dice_scores = compute_dice(warped_seg_np, fixed_seg_np, labels)

    # Filtrar valores NaN y ceros
    filtered_scores = [d for d in dice_scores if not np.isnan(d) and d > 0]

    if len(filtered_scores) > 0:
        dice_mean = np.mean(filtered_scores)
        dice_median = np.median(filtered_scores)
    else:
        dice_mean = 0.0
        dice_median = 0.0

    return warped_seg_np, fixed_seg_np, dice_mean, dice_median


###Logger
logger = Logger('NIREP_2D.log')

dataset_path = 'Baseline/NIREP_Matlab/'


### Load the dataset

logger.divider('Loading the dataset')

moving_image_index = 2
fixed_image_index = 1

if moving_image_index < 10:
    moving_image_index = f'0{moving_image_index}'
else:
    moving_image_index = f'{moving_image_index}'

if fixed_image_index < 10:
    fixed_image_index = f'0{fixed_image_index}'
else:
    fixed_image_index = f'{fixed_image_index}'

# Load fixed image
fixed = loadmat(f'{dataset_path}NIREP_{fixed_image_index}-Sub.mat')
fixed = fixed['im']

# Load its segmentation
segmentation_fixed = loadmat(f'{dataset_path}NIREP_{fixed_image_index}-Seg.mat')
segmentation_fixed = segmentation_fixed['seg']

# Load moving image
moving = loadmat(f'{dataset_path}NIREP_{moving_image_index}-Sub.mat')
moving = moving['im']

segmentation_moving = loadmat(f'{dataset_path}NIREP_{moving_image_index}-Seg.mat')
segmentation_moving = segmentation_moving['seg']


### Re-scale between 0 and 1 
logger.divider('Rescaling the images')
fix_max = fixed.max()
fix_min = fixed.min()

moving_max = moving.max()
moving_min = moving.min()

fixed = (fixed - fix_min) / (fix_max - fix_min)
moving = (moving - moving_min) / (moving_max - moving_min)


### Equivalent to 'single_channel' in the original code
source = torch.tensor(fixed, dtype=torch.float32)
target = torch.tensor(moving, dtype=torch.float32)


### Extract the slice from the 3D volume
slice_idx = 88

## PRO TIP! For sagital views: [slice_idx, :, :], for coronal views: [:, slice_idx, :], for axial views: [:, :, slice_idx]
logger.divider('Slicing the 3D volume')

# Transpose and flip to fix the orientation
source = source[:, :, slice_idx].T
target = target[:, :, slice_idx].T

source = torch.flipud(source)
target = torch.flipud(target)

# Extract the 2D axial slice and adjust orientation for the segmentation
aslice = 0.7 
fixed_seg = torch.flipud(torch.tensor(segmentation_fixed[:, :, round(slice_idx / aslice)].T, dtype=torch.float32))
fixed_mov = torch.flipud(torch.tensor(segmentation_moving[:, :, round(slice_idx / aslice)].T, dtype=torch.float32))


logger.info(f"Source image shape after slicing: {source.shape}")
logger.info(f"Target image shape after slicing: {target.shape}")
logger.info(f"Fixed segmentation shape after slicing: {fixed_seg.shape}")
logger.info(f"Moving segmentation shape after slicing: {fixed_mov.shape}")

figure, ax = config.plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(source, cmap='gray')
ax[0].set_title('Source Image')
ax[0].axis('off')
ax[1].imshow(target, cmap='gray')
ax[1].set_title('Target Image')
ax[1].axis('off')
ax[2].imshow(source - target, cmap='gray')
ax[2].set_title('Difference Image')
ax[2].axis('off')
config.plt.show()

figure, ax = config.plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(fixed_seg, cmap='gray')
ax[0].set_title('Fixed Segmentation')
ax[0].axis('off')
ax[1].imshow(fixed_mov, cmap='gray')
ax[1].set_title('Moving Segmentation')
ax[1].axis('off')
ax[2].imshow(fixed_seg - fixed_mov, cmap='gray')
ax[2].set_title('Difference Segmentation')
ax[2].axis('off')
config.plt.show()


### Now is the registration part, for this we will use the UnetDense model
logger.divider('Registering the images')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
para = read_yaml('parameters.yml')

# Add batch and padding
source = source.unsqueeze(0).unsqueeze(0)
target = target.unsqueeze(0).unsqueeze(0)
# Same with the segmentation
fixed_seg = fixed_seg.unsqueeze(0).unsqueeze(0)
fixed_mov = fixed_mov.unsqueeze(0).unsqueeze(0)


source = pad_to_multiple(source, 16)
target = pad_to_multiple(target, 16)
# not necessary for the segmentation
# fixed_seg = pad_to_multiple(fixed_seg, 16)
# fixed_mov = pad_to_multiple(fixed_mov, 16)

xDim, yDim = source.shape[2], source.shape[3]

logger.info(f"Source shape after padding: {source.shape}")
logger.info(f"Target shape after padding: {target.shape}")
logger.info(f"Fixed segmentation shape after padding: {fixed_seg.shape}")
logger.info(f"Moving segmentation shape after padding: {fixed_mov.shape}")


net, _, optimizer = initialize_network_optimizer2D(xDim, yDim, para, device)
net.to(device)

logger.info(f"Before training...")
net.eval()
net_test_model(net, fixed=target, moving=source, fixed_seg=fixed_seg, moving_seg=fixed_mov, device=device)

net.train()
custom_training(net, optimizer, num_epochs=200, fixed=target, moving=source, device=device)

logger.info(f"After training...")
net.eval()
net_test_model(net, fixed=target, moving=source, fixed_seg=fixed_seg, moving_seg=fixed_mov, device=device)










