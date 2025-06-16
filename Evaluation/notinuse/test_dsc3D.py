import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def warp_image_spatial_transformer(moving, displacement, mode='nearest'):
    moving_torch = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0).float()
    disp_torch = torch.from_numpy(displacement).float()
    nx, ny, nz = moving.shape
    grid_x = torch.linspace(-1, 1, nx)
    grid_y = torch.linspace(-1, 1, ny)
    grid_z = torch.linspace(-1, 1, nz)
    grid = torch.stack(torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij'), dim=-1)
    disp_norm = torch.zeros_like(disp_torch)
    disp_norm[..., 0] = 2 * disp_torch[..., 0] / (nx - 1)
    disp_norm[..., 1] = 2 * disp_torch[..., 1] / (ny - 1)
    disp_norm[..., 2] = 2 * disp_torch[..., 2] / (nz - 1)
    grid_disp = grid + disp_norm
    grid_disp = grid_disp.unsqueeze(0)
    grid_disp = grid_disp[..., [2,1,0]]
    warped = F.grid_sample(
        moving_torch,
        grid_disp,
        mode=mode,
        padding_mode='border',
        align_corners=False
    )
    return warped.squeeze().cpu().numpy()

def dice_coefficient(seg1, seg2):
    intersection = np.sum(seg1 * seg2)
    size1 = np.sum(seg1)
    size2 = np.sum(seg2)
    if size1 + size2 == 0:
        return 1.0
    return 2.0 * intersection / (size1 + size2)

# Crear un volumen 3D binario: cubo en el centro
shape = (32, 32, 32)
img = np.zeros(shape, dtype=np.float32)
img[12:20, 12:20, 12:20] = 1.0

# Campo de desplazamiento: mover 5 voxels en eje X (hacia la derecha visualmente)
disp = np.zeros(shape + (3,), dtype=np.float32)
disp[..., 0] = -5  # Eje X negativo para mover a la derecha visualmente

# Warping de la segmentación
warped_img = warp_image_spatial_transformer(img, disp, mode='nearest')
warped_seg = (warped_img > 0.5).astype(np.float32)

# Calcular DSC
dsc = dice_coefficient(img, warped_seg)
print(f"DSC: {dsc:.3f}")

# Visualización de un corte axial
slice_idx = shape[2] // 2
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.imshow(img[:, :, slice_idx], cmap='gray')
plt.title('Segmentación original (corte)')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(warped_seg[:, :, slice_idx], cmap='gray')
plt.title('Segmentación warpeada (corte)')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(img[:, :, slice_idx], cmap='gray', alpha=0.5)
plt.imshow(warped_seg[:, :, slice_idx], cmap='Reds', alpha=0.5)
plt.title('Solapamiento (corte)')
plt.axis('off')
plt.show()

#Visualización 3D
from mpl_toolkits.mplot3d import Axes3D

# Máscaras para visualización 3D
original_mask = img > 0.5
warped_mask = warped_seg > 0.5
intersection_mask = original_mask & warped_mask

fig = plt.figure(figsize=(18, 6))

# Segmentación original
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.voxels(original_mask, facecolors='gray', edgecolor='k', alpha=0.7)
ax1.set_title('Segmentación original')
ax1.set_axis_off()

# Segmentación warpeada
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.voxels(warped_mask, facecolors='red', edgecolor='k', alpha=0.7)
ax2.set_title('Segmentación warpeada')
ax2.set_axis_off()

# Intersección (solapamiento)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.voxels(original_mask, facecolors='gray', edgecolor='k', alpha=0.3)
ax3.voxels(warped_mask, facecolors='red', edgecolor='k', alpha=0.3)
ax3.voxels(intersection_mask, facecolors='green', edgecolor='k', alpha=0.9)
ax3.set_title('Solapamiento (verde = intersección)')
ax3.set_axis_off()

plt.tight_layout()
plt.show()