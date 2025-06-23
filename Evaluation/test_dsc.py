import numpy as np
from scipy.io import loadmat, savemat
import torch
import torch.nn.functional as F


def dice_coefficient(seg1, seg2):
    """Calcula el Dice Similarity Coefficient entre dos segmentaciones binarias."""
    seg1 = seg1.astype(np.bool_)
    seg2 = seg2.astype(np.bool_)
    intersection = np.logical_and(seg1, seg2).sum()
    return 2. * intersection / (seg1.sum() + seg2.sum() + 1e-8)

# Cargar los datos desde un .mat
data = loadmat('fixed_pair_results/results3.mat')

# Extraer y eliminar dimensiones extra con squeeze
nirep01 = np.squeeze(data['nirep01'])
nirep02_img = np.squeeze(data['nirep02'])
disp = data['disp']
id_grid = data['id']

# Extraer coordenadas X, Y, Z desde 'id'
X = np.squeeze(id_grid[0, 1, :, :, :])
Y = np.squeeze(id_grid[0, 0, :, :, :])
Z = np.squeeze(id_grid[0, 2, :, :, :])

# Desplazamientos
u = np.squeeze(disp[0, 1, :, :, :])
v = np.squeeze(disp[0, 0, :, :, :])
w = np.squeeze(disp[0, 2, :, :, :])

# Coordenadas deformadas
XI = X + u
YI = Y + v
ZI = Z + w

shape = disp.shape[2:]

# Crear grid deformado para grid_sample
XI = torch.from_numpy(XI)
YI = torch.from_numpy(YI)
ZI = torch.from_numpy(ZI)
nnew_locs = torch.stack([YI, XI, ZI], dim=-1)  # (D, H, W, 3)
nnew_locs = nnew_locs.permute(3,0,1,2).unsqueeze(0)  # (1, 3, D, H, W)

# Normalizar a [-1, 1] para grid_sample
for i in range(len(shape)):
    nnew_locs[:, i, ...] = 2 * (nnew_locs[:, i, ...] / (shape[i] - 1) - 0.5)

# Cambiar a formato (N, D, H, W, 3) y revertir canales
nnew_locs = nnew_locs.permute(0, 2, 3, 4, 1)
nnew_locs = nnew_locs[..., [2, 1, 0]]

# Aplicar warping a la imagen nirep01 usando grid_sample
nirep01_torch = torch.from_numpy(nirep01).float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
wwarped = F.grid_sample(nirep01_torch, nnew_locs, align_corners=True, mode='bilinear')
wwarped_np = wwarped.squeeze().cpu().numpy()

# Guardar la imagen deformada
savemat('wwarped.mat', {'kk': wwarped_np})

# Crear el campo de deformación (warp)
warp = np.zeros((*XI.shape, 3), dtype=np.float32)
warp[..., 0] = u
warp[..., 1] = v
warp[..., 2] = w

# Plot wwarped and warped from data
import matplotlib.pyplot as plt
def plot_slices(image, title, cmap='gray'):
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(image[:, :, image.shape[2] // 2 + i - 1], cmap=cmap)
        plt.title(f'{title} Slice {i + 1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# plot_slices(wwarped_np, 'Warped Image', cmap='gray')
# plot_slices(nirep01, 'Original Image', cmap='gray')
# plot_slices(np.squeeze(data['warped']), 'Warp Field', cmap='gray')

# Error between warped and original
error = wwarped_np - nirep02_img
plot_slices(error, 'Error between Warped and Original', cmap='gray')
# Error between wwarped and warped from data
error_warped = wwarped_np - np.squeeze(data['warped'])
plot_slices(error_warped, 'Error between Warped and Data Warped', cmap='gray')
print("Error between warped and original:", np.mean(np.abs(error)))
print("Error between warped and data warped:", np.mean(np.abs(error_warped)))


# DSC calculation
# Take the segmentations 
datadir = 'Baseline/NIREP_Matlab/'
nirep02 = loadmat(f'{datadir}/NIREP_02-Seg.mat')['seg']
nirep01_seg = loadmat(f'{datadir}/NIREP_01-Seg.mat')['seg']

# Warp the segmentation in the same way
nirep01_seg_torch = torch.from_numpy(nirep01_seg).float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
warped_seg = F.grid_sample(nirep01_seg_torch, nnew_locs, align_corners=True, mode='nearest')
warped_seg_np = warped_seg.squeeze().cpu().numpy()

# Binarize warped segmentation (in case of interpolation artifacts)
warped_seg_np = (warped_seg_np > 0.5).astype(np.uint8)


from scipy.ndimage import zoom
# Resample reference segmentation to match warped_seg shape if necessary
if nirep02.shape != warped_seg_np.shape:
    factors = [float(ws) / float(rs) for ws, rs in zip(warped_seg_np.shape, nirep02.shape)]
    nirep02 = zoom(nirep02, factors, order=0)


# Compute Dice Similarity Coefficient
dsc = dice_coefficient(warped_seg_np, nirep02)
print(f"Dice Similarity Coefficient (DSC): {dsc:.4f}")

# Plot error between warped segmentation and reference
error_seg = warped_seg_np - nirep02
plot_slices(error_seg, 'Error between Warped Segmentation and Reference', cmap='gray')
print("Mean Absolute Error between warped segmentation and reference:", np.mean(np.abs(error_seg)))

# -----------------------------
# Visualizaciones adicionales
# -----------------------------

# 1. Overlay de Segmentaciones (Antes vs. Después)
def plot_overlay(seg1, seg2, title):
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        slice_idx = seg1.shape[2] // 2 + i - 1
        plt.imshow(seg1[:, :, slice_idx], cmap='Reds', alpha=0.5)
        plt.imshow(seg2[:, :, slice_idx], cmap='Blues', alpha=0.5)
        plt.title(f'{title} Slice {i + 1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_overlay(warped_seg_np, nirep02, 'Overlay Warped vs Reference Segmentation')

# 2. Magnitud del campo de deformación
deformation_magnitude = np.sqrt(u**2 + v**2 + w**2)
plot_slices(deformation_magnitude, 'Deformation Field Magnitude', cmap='viridis')



#plotear nirep01, nirep02, warped, warp
def plot_images(images, titles, cmap='gray'):
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img[:, :, img.shape[2] // 2], cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
plot_images(
    [nirep01, nirep02_img, wwarped_np],
    ['NIREP01', 'NIREP02', 'Warped Image'], 
    cmap='gray'
)

# def dice_per_class(seg1, seg2, n_classes):
#     for c in range(n_classes):
#         dsc = dice_coefficient((seg1 == c), (seg2 == c))
#         print(f'DSC clase {c}: {dsc:.4f}')
# dice_per_class(warped_seg_np, nirep02, n_classes=3)

def plot_contours(seg1, seg2, title):
    from skimage import measure
    plt.figure(figsize=(6,6))
    slice_idx = seg1.shape[2] // 2
    plt.imshow(seg2[:,:,slice_idx], cmap='gray', alpha=0.5)
    for c in measure.find_contours(seg1[:,:,slice_idx], 0.5):
        plt.plot(c[:, 1], c[:, 0], 'r', linewidth=2)
    for c in measure.find_contours(seg2[:,:,slice_idx], 0.5):
        plt.plot(c[:, 1], c[:, 0], 'b', linewidth=2)
    plt.title(title)
    plt.axis('off')
    plt.show()
plot_contours(warped_seg_np, nirep02, 'Contornos Warped (rojo) vs Reference (azul)')

abs_diff = np.abs(warped_seg_np - nirep02)
plot_slices(abs_diff, 'Mapa de diferencias absolutas', cmap='hot')