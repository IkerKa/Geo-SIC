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

class RandomPairDataset(Dataset):
    def __init__(self, images, num_pairs=10000):
        self.images = images
        self.npairs = num_pairs

    def __len__(self):
        return self.npairs

    def __getitem__(self, idx):
        idx1 = np.random.randint(len(self.images))
        idx2 = np.random.randint(len(self.images))
        while idx2 == idx1:
            idx2 = np.random.randint(len(self.images))
        img1 = self.images[idx1].squeeze(0).squeeze(0)  # [D, H, W]
        img2 = self.images[idx2].squeeze(0).squeeze(0)
        return img1, img2

#class for selected training
class SelectedPairDataset(Dataset):
    def __init__(self, images, fixed_idx, num_pairs=10000):
        self.images = images
        self.fixed_idx = fixed_idx
        self.npairs = num_pairs

    def __len__(self):
        return self.npairs

    def __getitem__(self, idx):
        moving_idx = np.random.randint(len(self.images))
        while moving_idx == self.fixed_idx:
            moving_idx = np.random.randint(len(self.images))
        img_fixed = self.images[self.fixed_idx].squeeze(0).squeeze(0)  # [D, H, W]
        img_moving = self.images[moving_idx].squeeze(0).squeeze(0)
        return img_fixed, img_moving

def test_datasets():
    # Test the RandomPairDataset and SelectedPairDataset classes by printing the indexes of the images
    images = [torch.randn(1, 1, 64, 64, 64) for _ in range(10)]  # Simulated images
    random_dataset = RandomPairDataset(images, num_pairs=5)
    selected_dataset = SelectedPairDataset(images, fixed_idx=0, num_pairs=5)
    print("Random Pair Dataset:")
    for i in range(len(random_dataset)):
        # Get the random indices used for this pair
        idx1 = np.random.randint(len(images))
        idx2 = np.random.randint(len(images))
        while idx2 == idx1:
            idx2 = np.random.randint(len(images))
        print(f"Pair {i}: Index1={idx1}, Index2={idx2}")
    print("\nSelected Pair Dataset:")
    for i in range(len(selected_dataset)):
        moving_idx = np.random.randint(len(images))
        while moving_idx == selected_dataset.fixed_idx:
            moving_idx = np.random.randint(len(images))
        print(f"Pair {i}: Fixed Index={selected_dataset.fixed_idx}, Moving Index={moving_idx}")
def debug_plotting():
    # Plot from the images dataset an image and its segmentation
    data = load_all_data('Baseline/NIREP_Matlab/', 'NIREP_3D')
    data['images'] = [rescale_image(img) for img in data['images']]

    # Convert images and segmentations to tensors
    data_tensors = {
        'images': [convert_to_tensor(img) for img in data['images']],
        'segmentations': [convert_to_tensor(seg) for seg in data['segmentations']]
    }

    # Select the first image and its segmentation
    img = data_tensors['images'][0].squeeze().numpy()  # Convert tensor to numpy array
    seg = data_tensors['segmentations'][0].squeeze().numpy()

    logger.info(f'Image shape: {img.shape}, Segmentation shape: {seg.shape}')
    slice_idx = 90
    logger.divider('Slicing the 3D volume')

    # Transpose and flip to fix the orientation for the image
    img_axial = img[:, :, slice_idx].T
    seg_axial = seg[:, :, slice_idx].T

    img_axial = np.flipud(img_axial)

    # Extract the 2D axial slice and adjust orientation for the segmentation
    aslice = 0.7
    fixed_seg = torch.flipud(torch.tensor(seg[:, :, round(slice_idx / aslice)].T, dtype=torch.float32))
    # If you have a moving segmentation, you can do similar:
    # fixed_mov = torch.flipud(torch.tensor(segmentation_moving[:, :, round(slice_idx / aslice)].T, dtype=torch.float32))

    

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_axial, cmap='gray')
    plt.title('Image Slice at Z=90 (Axial, flipped)')
    plt.subplot(1, 2, 2)
    plt.imshow(fixed_seg, cmap='tab20')
    plt.title('Segmentation Slice at Z=90 (Axial, flipped)')
    plt.show()


### DATASET TRAINING FUNCTIONS ###
def random_training_dataloader(net, optimizer, num_epochs, train_images, device, batch_size=2, num_workers=0, npairs=10000):
    dataset = RandomPairDataset(train_images, num_pairs=npairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    loss_per_epoch = []
    net.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (source_batch, moving_batch) in enumerate(loader):
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(loader)}", end='\r')
            # source_batch, moving_batch: [batch, D, H, W]
            # Añade dims: [batch, 1, D, H, W]
            source_batch = source_batch.unsqueeze(1).to(device)
            moving_batch = moving_batch.unsqueeze(1).to(device)

            optimizer.zero_grad()
            y_src, momentum, _, _ = net(moving_batch, source_batch, registration=True, shooting="SVF", return_phi=True)
            Dist = config.NCC().loss(y_src, source_batch)
            Reg = config.Grad(penalty='l2')
            loss = Dist + Reg.loss(momentum)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # # To limit 
            # if batch_idx >= 5:
            #     break

        mean_loss = epoch_loss / (batch_idx + 1)
        loss_per_epoch.append(mean_loss)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {mean_loss:.4f}")

    logger.info(f"Training completed. Mean loss per epoch: {np.mean(loss_per_epoch):.4f}")

    #plotear la loss por epoch
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_per_epoch, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def selected_training_dataloader(net, optimizer, num_epochs, train_images, device, batch_size=2, num_workers=0, npairs=10000):
    """
    Different index per epoch, but same fixed index for all pairs in the epoch.
    """
    loss_per_epoch = []
    net.train()

    for epoch in range(num_epochs):

        fixed_idx = np.random.randint(len(train_images))
        dataset = SelectedPairDataset(train_images, fixed_idx=fixed_idx, num_pairs=npairs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        epoch_loss = 0
        for batch_idx, (fixed_batch, moving_batch) in enumerate(loader):
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(loader)}, Fixed idx: {fixed_idx}", end='\r')
            fixed_batch = fixed_batch.unsqueeze(1).to(device)    # [batch, 1, D, H, W]
            moving_batch = moving_batch.unsqueeze(1).to(device)

            optimizer.zero_grad()
            y_src, momentum, _, _ = net(moving_batch, fixed_batch, registration=True, shooting="SVF", return_phi=True)
            Dist = config.NCC().loss(y_src, fixed_batch)
            Reg = config.Grad(penalty='l2')
            loss = Dist + Reg.loss(momentum)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        mean_loss = epoch_loss / (batch_idx + 1)
        loss_per_epoch.append(mean_loss)
        logger.info(f"Epoch {epoch+1}/{num_epochs} (fixed_idx={fixed_idx}) - Loss: {mean_loss:.4f}")

    logger.info(f"Training completed. Mean loss per epoch: {np.mean(loss_per_epoch):.4f}")

    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_per_epoch, marker='o')
    plt.title('Training Loss per Epoch (Selected Pair)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

### HANDMADE TRAINING FUNCTIONS ###
def random_training(net, optimizer, num_epochs, train_images, device, num_batches=5, batch_size=2):
    loss_per_epoch = []
    net.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in range(num_batches):
            batch_loss = 0
            optimizer.zero_grad()
            for i in range(batch_size):
                print(f"Batch {batch+1}, iteration {i+1}/{batch_size}, Epoch {epoch+1}/{num_epochs}", end='\r')
                source_idx = np.random.randint(len(train_images))
                moving_idx = np.random.randint(len(train_images))
                while moving_idx == source_idx:
                    moving_idx = np.random.randint(len(train_images))

                source_image = train_images[source_idx].to(device)
                moving_image = train_images[moving_idx].to(device)

                y_src, momentum, _, _ = net(moving_image, source_image, registration=True, shooting="SVF", return_phi=True)
                

                Dist = config.NCC().loss(y_src, source_image)
                Reg = config.Grad(penalty='l2')
                loss = Dist + 0.01 * Reg.loss(momentum)
                (loss / batch_size).backward() 
                batch_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += batch_loss / batch_size  

        mean_loss = epoch_loss / num_batches
        loss_per_epoch.append(mean_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {mean_loss:.4f}")

    logger.info(f"Training completed. Mean loss per epoch: {np.mean(loss_per_epoch):.4f}")

    # Plot
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_per_epoch, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def selected_training(net, optimizer, num_epochs, train_images, device, num_batches=5, batch_size=2):
    loss_per_epoch = []
    num_images = len(train_images)

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch in range(num_batches):
            fixed_idx = batch % num_images
            fixed_image = train_images[fixed_idx].to(device)
            batch_loss = 0
            optimizer.zero_grad()

            for _ in range(batch_size):
                moving_indices = [i for i in range(num_images)]
                moving_idx = np.random.choice(moving_indices)
                moving_image = train_images[moving_idx].to(device)

                y_src, momentum, _, _ = net(moving_image, fixed_image, registration=True, shooting="SVF", return_phi=True)
                Dist = config.NCC().loss(y_src, fixed_image)
                Reg = config.Grad(penalty='l2')
                Reg_loss = Reg.loss(momentum)
                loss = Dist + Reg_loss

                (loss / batch_size).backward()  # backward por muestra
                batch_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += batch_loss / batch_size  # promedio de la loss del batch

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


### INFERENCE FUNCTIONS ###
def net_test_model_3d(net, test_dataset, save_phi, device):
    """
    Test the registration network on a 3D test dataset.
    Args:
        net: The trained registration network.
        test_dataset: dict with 'images' and 'segmentations' lists (each element: (1,1,D,H,W) tensor).
        save_phi: bool, whether to save the deformation field.
        device: torch.device.
    """
    ssims = []
    rmses = []
    phis = []

    num_cases = len(test_dataset['images'])
    for i in range(0, num_cases, 2):
        # Use pairs: (i, i+1)
        if i+1 >= num_cases:
            break
        fixed = test_dataset['images'][i].to(device).float()
        moving = test_dataset['images'][i+1].to(device).float()
        fixed_seg = test_dataset['segmentations'][i].to(device).float()
        moving_seg = test_dataset['segmentations'][i+1].to(device).float()
        with torch.no_grad():
            y_src, _, _, new_locs = net(moving, fixed, registration=True, shooting='SVF', return_phi=True)

            if save_phi:
                phi_inv = new_locs[0,...].cpu().detach().numpy()
                savemat(f'phi_inv_3d_{i}.mat', {'phi_inv': phi_inv})

            print("phi_inv stats: min", new_locs[0].min().item(), "max", new_locs[0].max().item(), "mean", new_locs[0].mean().item())
            print("phi_inv[0,0,0]:", new_locs[0][0,0,0].cpu().numpy())

            warped_seg, _, dice_mean, dice_median = customSegmentation(moving_seg, new_locs[0], fixed_seg, device, True)
            phis.append(new_locs[0].cpu())

            # Optionally, compute SSIM and RMSE 
            # ssims.append(ssim_score)
            # rmses.append(rmse_score)
            #print RMSE and SSIM
            
            ssim_score = config.ssim(
            y_src.squeeze().cpu().detach().numpy(),
            fixed.squeeze().cpu().detach().numpy(),
            data_range=fixed.max().item() - fixed.min().item()
            )
            rmse_score = config.np.sqrt(config.np.mean((fixed.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()) ** 2))
            logger.info(f"Case {i//2 + 1}/{num_cases//2} - RMSE: {rmse_score:.4f}, SSIM: {ssim_score:.4f}, Dice Mean: {dice_mean:.4f}, Dice Median: {dice_median:.4f}")
            # Plot images and deformation grid for slice slice_idx
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            moving_np = moving.cpu().squeeze().numpy()   # (D,H,W)
            fixed_np = fixed.cpu().squeeze().numpy()
            y_src_np = y_src.cpu().squeeze().numpy()

            slice_idx = 90

            # Prepare axial slices with orientation adjustment (transpose + flipud)
            moving_axial = np.flipud(moving_np[:, :, slice_idx].T)
            fixed_axial = np.flipud(fixed_np[:, :, slice_idx].T)
            y_src_axial = np.flipud(y_src_np[:, :, slice_idx].T)
            error_axial = fixed_axial - y_src_axial

            axes[0].imshow(moving_axial, cmap='gray', origin='upper')
            axes[0].set_title('Moving (Source) Image')
            axes[1].imshow(fixed_axial, cmap='gray', origin='upper')
            axes[1].set_title('Fixed (Target) Image')
            axes[2].imshow(y_src_axial, cmap='gray', origin='upper')
            axes[2].set_title('Registered Image')
            axes[3].imshow(error_axial, cmap='gray', origin='upper')
            axes[3].set_title('Error (Fixed - Registered)')

            # Grid axial
            phi_inv = new_locs[0].cpu().detach().numpy()  # [Z, Y, X, 3]

            # AXIAL = plano XY = corte a lo largo de Z
            phi_slice = phi_inv[:, :, slice_idx, :]       # shape: [Z, Y, 3]
            H, W = phi_slice.shape[:2]                    # H (vertical), W (horizontal)

            # Compute grid coordinates (no rotation, just transpose to match image orientation)
            x_grid = (phi_slice[..., 2] + 1) * (W - 1) / 2  # X = horizontal
            y_grid = (phi_slice[..., 1] + 1) * (H - 1) / 2  # Y = vertical

            # Transpose to match image orientation (like images)
            x_grid_t = x_grid.T
            y_grid_t = y_grid.T

            ax = axes[4]
            interval = 8
            for row in range(0, x_grid_t.shape[0], interval):
                ax.plot(x_grid_t[row, :], y_grid_t[row, :], 'm')
            for col in range(0, x_grid_t.shape[1], interval):
                ax.plot(x_grid_t[:, col], y_grid_t[:, col], 'm')

            ax.set_title(f'Deformation Grid (Axial slice {slice_idx})')
            ax.set_aspect('equal')
            ax.set_xlim(0, W - 1)
            ax.set_ylim(0, H - 1)
            ax.grid(True)





            plt.tight_layout()
            plt.show()

            # plot segmentations
            # Prepare segmentations for plotting (transpose + flipud for correct orientation)
            moving_seg_np = moving_seg.cpu().squeeze().numpy()
            fixed_seg_np = fixed_seg.cpu().squeeze().numpy()
            warped_seg_np = warped_seg

            # For correct orientation, transpose and flipud (like images)
            aslice = 0.7
            moving_seg_axial = torch.flipud(torch.tensor(moving_seg_np[:, :, round(slice_idx / aslice)].T, dtype=torch.float32)).numpy()
            fixed_seg_axial = torch.flipud(torch.tensor(fixed_seg_np[:, :, round(slice_idx / aslice)].T, dtype=torch.float32)).numpy()
            # warped_seg_axial = torch.flipud(torch.tensor(warped_seg_np[:, :, round(slice_idx / aslice)].T, dtype=torch.float32)).numpy()
            warped_seg_axial = torch.flipud(torch.tensor(warped_seg_np[:, :, round(slice_idx / aslice)].T, dtype=torch.float32)).numpy()
            # target_shape = fixed_seg_axial.shape
            # if warped_seg_axial.shape != target_shape:
            #     warped_seg_axial = resize(warped_seg_axial, target_shape, order=0, preserve_range=True, anti_aliasing=False)

            plt.figure(figsize=(20, 5))
            plt.subplot(1, 4, 1)
            plt.imshow(moving_seg_axial, cmap='gray')
            plt.title('Moving Segmentation (Axial, flipped)')
            plt.subplot(1, 4, 2)
            plt.imshow(fixed_seg_axial, cmap='gray')
            plt.title('Fixed Segmentation (Axial, flipped)')
            plt.subplot(1, 4, 3)
            plt.imshow(warped_seg_axial, cmap='gray')
            plt.title('Warped Segmentation (Axial, flipped)')
            
            seg_error = fixed_seg_axial - warped_seg_axial
            max_abs_error = np.max(np.abs(seg_error))
            if max_abs_error != 0:
                seg_error_norm = seg_error / max_abs_error
            else:
                seg_error_norm = seg_error
            plt.subplot(1, 4, 4)
            im = plt.imshow(seg_error_norm, cmap='bwr', vmin=-1, vmax=1)
            plt.title('Segmentation Error (Fixed - Warped)')
            plt.colorbar(im)
            plt.show()


   
def customSegmentation_segmentations(I1_seg, phi_inv, I2_seg, dev):
    # Asegura que las segmentaciones tengan shape [1, 1, D, H, W]
    if I1_seg.dim() == 3:
        I1_seg = I1_seg.unsqueeze(0).unsqueeze(0)
    elif I1_seg.dim() == 4:
        I1_seg = I1_seg.unsqueeze(0)
    if I2_seg.dim() == 3:
        I2_seg = I2_seg.unsqueeze(0).unsqueeze(0)
    elif I2_seg.dim() == 4:
        I2_seg = I2_seg.unsqueeze(0)

    # Ensure shape of phi [1, D, H, W, 3]
    if phi_inv.dim() == 5:
        if phi_inv.shape[1] == 3:
            # shape: [1, 3, D, H, W] → [1, D, H, W, 3]
            phi_inv = phi_inv.permute(0, 2, 3, 4, 1)
        elif phi_inv.shape[-1] != 3:
            raise ValueError(f"phi_inv has invalid shape {phi_inv.shape}")
    elif phi_inv.dim() == 4 and phi_inv.shape[0] == 3:
        # shape: [3, D, H, W] → [1, D, H, W, 3]
        phi_inv = phi_inv.permute(1, 2, 3, 0).unsqueeze(0)
    elif phi_inv.dim() == 4 and phi_inv.shape[-1] == 3:
        phi_inv = phi_inv.unsqueeze(0)
    else:
        raise ValueError(f"phi_inv has unexpected shape {phi_inv.shape}")

    # Reshape segmentations to match phi_inv
    target_shape = phi_inv.shape[1:4]  # (D, H, W)
    if I1_seg.shape[-3:] != target_shape:
        I1_seg = F.interpolate(I1_seg.float(), size=target_shape, mode='nearest')
    if I2_seg.shape[-3:] != target_shape:
        I2_seg = F.interpolate(I2_seg.float(), size=target_shape, mode='nearest')

    print("I1_seg shape:", I1_seg.shape)
    print("phi_inv shape:", phi_inv.shape)

    # Warping
    
    phi_inv_for_st = phi_inv.permute(0, 4, 1, 2, 3).contiguous()
    spat_trans = SpatialTransformer(size=target_shape, mode='nearest').to(dev)
    warped_seg = spat_trans(I1_seg, phi_inv_for_st)

    warped_seg_np = warped_seg.squeeze().cpu().detach().numpy().astype(np.uint8)
    fixed_seg_np = I2_seg.squeeze().cpu().detach().numpy().astype(np.uint8)

    # Etiquetas: puedes usar np.arange(1, 33) para NIREP o las presentes en la referencia
    # labels = np.unique(fixed_seg_np)
    # labels = labels[labels != 0]
    labels = np.arange(1, 33)  # Assuming labels from 1 to 32
    dice_scores = compute_dice(warped_seg_np, fixed_seg_np, labels)
    filtered_scores = [d for d in dice_scores if not np.isnan(d) and d > 0]
    dice_mean = np.mean(filtered_scores) if filtered_scores else 0.0
    dice_median = np.median(filtered_scores) if filtered_scores else 0.0

    return warped_seg_np, fixed_seg_np, dice_mean, dice_median

def customSegmentation(I1_seg, phi_inv, I2_seg, dev, test_artificial_warp=False):
    """
    Warps the moving segmentation using the deformation field and computes Dice scores.
    Args:
        I1_seg: Moving segmentation (tensor, shape [D,H,W] or [1,1,D,H,W])
        phi_inv: Deformation field (tensor, shape [1,D,H,W,3] or similar)
        I2_seg: Fixed segmentation (tensor, shape [D,H,W] or [1,1,D,H,W])
        dev: torch.device
        test_artificial_warp: bool, if True, applies a manual shift for debugging
    Returns:
        warped_seg_np: Warped moving segmentation (numpy array)
        fixed_seg_np: Fixed segmentation (numpy array)
        dice_mean: Mean Dice score (float)
        dice_median: Median Dice score (float)
    """
    # Ensure segmentations have shape [1, 1, D, H, W]
    for name, seg in zip(['I1_seg', 'I2_seg'], [I1_seg, I2_seg]):
        if seg.dim() == 3:
            seg = seg.unsqueeze(0).unsqueeze(0)
        elif seg.dim() == 4:
            seg = seg.unsqueeze(0)
        elif seg.dim() == 5:
            pass
        else:
            raise ValueError(f"{name} has unexpected shape {seg.shape}")
        if name == 'I1_seg':
            I1_seg = seg
        else:
            I2_seg = seg

    # Ensure phi_inv shape [1, D, H, W, 3]
    if phi_inv.dim() == 5:
        if phi_inv.shape[1] == 3:
            phi_inv = phi_inv.permute(0, 2, 3, 4, 1)
        elif phi_inv.shape[-1] != 3:
            raise ValueError(f"phi_inv has invalid shape {phi_inv.shape}")
    elif phi_inv.dim() == 4 and phi_inv.shape[0] == 3:
        phi_inv = phi_inv.permute(1, 2, 3, 0).unsqueeze(0)
    elif phi_inv.dim() == 4 and phi_inv.shape[-1] == 3:
        phi_inv = phi_inv.unsqueeze(0)
    else:
        raise ValueError(f"phi_inv has unexpected shape {phi_inv.shape}")

    # Match spatial shape
    target_shape = I1_seg.shape[-3:]
    if phi_inv.shape[1:4] != target_shape:
        phi_inv_tr = phi_inv.permute(0, 4, 1, 2, 3)
        phi_inv_tr = F.interpolate(phi_inv_tr, size=target_shape, mode='trilinear', align_corners=True)
        phi_inv = phi_inv_tr.permute(0, 2, 3, 4, 1)
    if I2_seg.shape[-3:] != target_shape:
        I2_seg = F.interpolate(I2_seg.float(), size=target_shape, mode='nearest')

    # Debug prints
    print(f"I1_seg shape: {I1_seg.shape}, I2_seg shape: {I2_seg.shape}, phi_inv shape: {phi_inv.shape}")
    print(f"phi_inv min/max: {phi_inv.min().item():.4f}/{phi_inv.max().item():.4f}")

    # --- ARTIFICIAL WARP TEST ---
    # if test_artificial_warp:
    #     D, H, W = target_shape
    #     identity_grid = torch.stack(torch.meshgrid(
    #         torch.linspace(-1, 1, D),
    #         torch.linspace(-1, 1, H),
    #         torch.linspace(-1, 1, W),
    #         indexing='ij'
    #     ), dim=-1).unsqueeze(0).to(dev)  # [1, D, H, W, 3]
    #     shift = 2.0 / (W - 1)  # 1 voxel in X
    #     phi_inv = identity_grid.clone()
    #     phi_inv[..., 2] = torch.clamp(phi_inv[..., 2] + shift, -1, 1)
    #     print("Using artificial shift field for debugging.")

    # Warping
    phi_inv_for_st = phi_inv.permute(0, 4, 1, 2, 3)
    spat_trans = SpatialTransformer(size=target_shape, mode='nearest').to(dev)
    warped_seg = spat_trans(I1_seg, phi_inv_for_st)

    warped_seg_np = warped_seg.squeeze().cpu().detach().numpy().astype(np.uint8)
    fixed_seg_np = I2_seg.squeeze().cpu().detach().numpy().astype(np.uint8)

    # Print unique labels for debugging
    print("Unique labels in fixed segmentation:", np.unique(fixed_seg_np))
    print("Unique labels in warped segmentation:", np.unique(warped_seg_np))
    print("Are warped and moving segmentations equal?", np.array_equal(warped_seg_np, I1_seg.squeeze().cpu().numpy()))

    # Use only labels present in either segmentation, excluding background (0)
    labels = np.unique(np.concatenate([fixed_seg_np, warped_seg_np]))
    labels = labels[labels != 0]

    dice_scores = compute_dice(warped_seg_np, fixed_seg_np, labels)
    dice_mean = np.mean(dice_scores)
    dice_median = np.median(dice_scores)

    return warped_seg_np, fixed_seg_np, dice_mean, dice_median

### AUXILIARY FUNCTIONS ###
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



### MAIN FUNCTION ###

logger = Logger('NIREP_3D.log')

def main():

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

    # Shape of images and segmentations
    logger.info(f'Shape of training images: {train_data["images"][0].shape}')
    logger.info(f'Shape of training segmentations: {train_data["segmentations"][0].shape}')

    ### Debug: Check the shapes of the tensors and plot the first image

    # i = 0  # Index of the first training image
    # img = train_data['images'][i].numpy()  # Convert tensor to numpy array
    # logger.info(f'Train image {i} shape: {img.shape}')
    # plt.imshow(img[:, :, slice_idx], cmap='gray')
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
    test_data['images'] = [img.unsqueeze(0).unsqueeze(0) for img in test_data['images']]
    test_data['images'] = [pad_to_multiple(img) for img in test_data['images']]
    logger.info('Shape of images after padding: ' + str(train_data['images'][0].shape))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    para = read_yaml('parameters.yml')

    xDim, yDim, zDim = train_data['images'][0].shape[2:5]
    net, _ , optimizer = initialize_network_optimizer(xDim, yDim, zDim, para, device)

    net.train()
    logger.info(f'Number of available CPU cores: {torch.get_num_threads()}')
    

    ### Training with dataloaders ###

    # Monica 1a opcion: Random pairs of images
    # random_training_dataloader(
    #     net, optimizer,
    #     num_epochs=15,
    #     train_images=train_data['images'],
    #     device=device,
    #     batch_size=2,
    #     num_workers=8,
    #     npairs=46
    # )

    # Monica 2a opcion: Fixed source and random moving images
    # selected_training_dataloader(
    #     net, optimizer,
    #     num_epochs=15,
    #     train_images=train_data['images'],
    #     device=device,
    #     batch_size=2,
    #     num_workers=8,
    #     npairs=46
    # )

    ### Quick debug training
    # selected_training_dataloader(
    #     net, optimizer,
    #     num_epochs=1,
    #     train_images=train_data['images'],
    #     device=device,
    #     batch_size=2,
    #     num_workers=4,
    #     npairs=10
    # )


    ### Training with handmade functions ###
    random_training(net, optimizer, num_epochs=1, train_images=train_data['images'], device=device, num_batches=10, batch_size=1)
    # logger.info('Starting training with fixed source and random moving images')
    # selected_training(net, optimizer, num_epochs=10, train_images=train_data['images'], device=device, num_batches=3, batch_size=1)

    #Evaluation
    logger.info('Testing the trained model on the test dataset')
    net.eval()
    net_test_model_3d(net, test_dataset=test_data, save_phi=False, device=device)

if __name__ == "__main__":
    _test = False
    if _test:
        test_datasets()
        debug_plotting()
    else:
        main()



