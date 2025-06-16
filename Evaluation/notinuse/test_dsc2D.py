import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def warp_image_spatial_transformer_2d(moving, displacement, mode='bilinear'):
    moving_torch = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0).float()
    disp_torch = torch.from_numpy(displacement).float()
    nx, ny = moving.shape
    grid_x = torch.linspace(-1, 1, nx)
    grid_y = torch.linspace(-1, 1, ny)
    grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='ij'), dim=-1)
    disp_norm = torch.zeros_like(disp_torch)
    disp_norm[..., 0] = 2 * disp_torch[..., 0] / (nx - 1)
    disp_norm[..., 1] = 2 * disp_torch[..., 1] / (ny - 1)
    grid_disp = grid + disp_norm
    grid_disp = grid_disp.unsqueeze(0)
    grid_disp = grid_disp[..., [1, 0]]
    warped = F.grid_sample(
        moving_torch,
        grid_disp,
        mode=mode,
        padding_mode='border',
        align_corners=False
    )
    return warped.squeeze().cpu().numpy()

# Imagen binaria: cuadrado blanco en el centro
img = np.zeros((32, 32), dtype=np.float32)
img[12:20, 12:20] = 1.0

# Campo de desplazamiento: mueve 5 píxeles a la derecha visualmente
disp = np.zeros((32, 32, 2), dtype=np.float32)
disp[..., 1] = -5  # Negativo para mover a la derecha visualmente

# Warping de la segmentación
warped_img = warp_image_spatial_transformer_2d(img, disp, mode='nearest')
warped_seg = (warped_img > 0.5).astype(np.float32)

# Intersección
intersection = img * warped_seg

# Visualización
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Segmentación original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(warped_seg, cmap='gray')
plt.title('Segmentación warpeada')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img, cmap='gray', alpha=0.5, label='Original')
plt.imshow(warped_seg, cmap='Reds', alpha=0.5, label='Warped')
plt.imshow(intersection, cmap='Greens', alpha=0.7, label='Intersección')
plt.title('Solapamiento (verde = intersección)')
plt.axis('off')

plt.show()