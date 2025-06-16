import numpy as np
from scipy.io import loadmat
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import nibabel as nib

def warp_image_spatial_transformer(moving, displacement, mode='bilinear'):
    """
    moving: numpy array (nx, ny, nz)
    displacement: numpy array (nx, ny, nz, 3)
    """
    import torch
    import torch.nn.functional as F

    moving_torch = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0).float()
    disp_torch = torch.from_numpy(displacement).float()
    shape = moving.shape

    # Construir grilla de coordenadas deformadas
    grid = np.stack(np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij'
    ), axis=-1).astype(np.float32)  # (nx, ny, nz, 3)
    new_locs = grid + displacement  # (nx, ny, nz, 3)

    # Normalizar a [-1, 1]
    for i in range(3):
        new_locs[..., i] = 2 * (new_locs[..., i] / (shape[i] - 1) - 0.5)

    # Convertir a torch y permutar ejes para grid_sample
    new_locs = torch.from_numpy(new_locs).float()
    new_locs = new_locs.unsqueeze(0)  # (1, nx, ny, nz, 3)
    new_locs = new_locs[..., [2, 1, 0]]  # Revertir canales

    warped = F.grid_sample(
        moving_torch,
        new_locs,
        mode=mode,
        padding_mode='border',
        align_corners=True  # ¡IMPORTANTE!
    )
    return warped.squeeze().cpu().numpy()


def dice_coefficient(seg1, seg2):
    intersection = np.sum((seg1 > 0) & (seg2 > 0))
    size1 = np.sum(seg1 > 0)
    size2 = np.sum(seg2 > 0)
    if size1 + size2 == 0:
        return 1.0
    return 2.0 * intersection / (size1 + size2)

# --- Carga de datos ---
nirep_dir = 'Baseline/NIREP_Matlab/'
seg1 = loadmat(f'{nirep_dir}/NIREP_01-Seg.mat')['seg']
seg2 = loadmat(f'{nirep_dir}/NIREP_02-Seg.mat')['seg']
warp = loadmat('results.mat')['disp']

# dir = 'Evaluation'
# seg_path = f'{dir}/na01_seg.nii.gz'
# ref_path = f'{dir}/na02_seg.nii.gz'

# seg1 = nib.load(seg_path).get_fdata()
# seg2 = nib.load(ref_path).get_fdata()



# Preprocesamiento de segmentaciones
if warp.shape[0] == 1:
    warp = np.squeeze(warp, axis=0)  #(3, nx, ny, nz)
if warp.shape[0] == 3:
    warp = np.moveaxis(warp, 0, -1)  #(nx, ny, nz, 3)

# Redimensionamos porque no tienen la misma resolución
target_shape = seg1.shape
factors = [target_shape[i] / warp.shape[i] for i in range(3)]
warp_resized = np.zeros(target_shape + (3,), dtype=np.float32)
for i in range(3):
    warp_resized[..., i] = zoom(warp[..., i], factors, order=1)
    warp_resized[..., i] *= factors[i]

# DSC para todas las etiquetas
num_labels = int(max(seg1.max(), seg2.max()))
dscs = []
for label in range(1, num_labels + 1):
    mask1 = (seg1 == label).astype(np.float32)
    mask2 = (seg2 == label).astype(np.float32)
    if np.sum(mask1) == 0 and np.sum(mask2) == 0:
        dscs.append(np.nan)
        continue
    warped_mask1 = warp_image_spatial_transformer(mask1, warp_resized, mode='nearest')
    warped_mask1_bin = (warped_mask1 > 0.5).astype(np.float32)
    dsc = dice_coefficient(mask2, warped_mask1_bin)
    dscs.append(dsc)
    print(f"DSC (etiqueta {label}): {dsc:.3f}")

print("\nDSC por labels")
for label, dsc in enumerate(dscs, 1):
    print(f"Etiqueta {label}: DSC = {dsc:.3f}")

print(f"\nDSC medio (no nans): {np.nanmean(dscs):.3f}")


print("\nVisualización de segmentaciones y warping")


#Ploteamos una etiqueta específica
label = 2
mask1 = (seg1 == label).astype(np.float32)
mask2 = (seg2 == label).astype(np.float32)
warped_mask1 = warp_image_spatial_transformer(mask1, warp_resized, mode='nearest')
warped_mask1_bin = (warped_mask1 > 0.5).astype(np.float32)

slice_idx = mask1.shape[2] // 2
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.imshow(mask1[:, :, slice_idx], cmap='Blues', alpha=0.7)
plt.title('Segmentación original')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(mask2[:, :, slice_idx], cmap='Greens', alpha=0.7)
plt.imshow(warped_mask1_bin[:, :, slice_idx], cmap='Reds', alpha=0.5)
plt.title('Referencia (verde) + Warpeada (rojo)')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(mask2[:, :, slice_idx], cmap='Greens', alpha=0.7)
plt.imshow(warped_mask1_bin[:, :, slice_idx], cmap='Reds', alpha=0.5)
plt.imshow((mask2[:, :, slice_idx].astype(bool) & warped_mask1_bin[:, :, slice_idx].astype(bool)), cmap='Oranges', alpha=0.8)
plt.title('Solapamiento (naranja = intersección)')
plt.axis('off')
plt.tight_layout()
plt.show()

factor = 0.2  # Reduce al 20% del tamaño original
mask2_small = zoom(mask2, factor, order=0)
warped_small = zoom(warped_mask1_bin, factor, order=0)
intersection_small = (mask2_small > 0.5) & (warped_small > 0.5)

mask2_coords = np.argwhere(mask2_small > 0.5)
warped_coords = np.argwhere(warped_small > 0.5)
inter_coords = np.argwhere(intersection_small)

# --- Visualización 3D devoxelizada mejorada ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

if mask2_coords.size > 0:
    ax.scatter(mask2_coords[:, 0], mask2_coords[:, 1], mask2_coords[:, 2], 
               c='lime', alpha=0.15, s=3, label='Referencia')
if warped_coords.size > 0:
    ax.scatter(warped_coords[:, 0], warped_coords[:, 1], warped_coords[:, 2], 
               c='red', alpha=0.15, s=3, label='Warpeada')
if inter_coords.size > 0:
    ax.scatter(inter_coords[:, 0], inter_coords[:, 1], inter_coords[:, 2], 
               c='orange', alpha=0.7, s=6, label='Intersección')

ax.set_title('Solapamiento 3D devoxelizado\nVerde: Referencia, Rojo: Warpeada, Naranja: Intersección')
ax.legend(loc='upper right', fontsize=12)
ax.view_init(elev=30, azim=120)  

# Ajuste de ejes para mejor visualización
max_range = np.array([mask2_small.shape[0], mask2_small.shape[1], mask2_small.shape[2]]).max()
ax.set_xlim(0, max_range)
ax.set_ylim(0, max_range)
ax.set_zlim(0, max_range)
ax.set_axis_off()
plt.tight_layout()
plt.show()

slice_idx = warp_resized.shape[2] // 2
u = warp_resized[:, :, slice_idx, 0]
v = warp_resized[:, :, slice_idx, 1]
X, Y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))

plt.figure(figsize=(8, 8))
plt.imshow((seg2[:, :, slice_idx] > 0).astype(float), cmap='gray', alpha=0.3)
step = 8  
plt.quiver(X[::step, ::step], Y[::step, ::step], v[::step, ::step], u[::step, ::step], color='red', angles='xy', scale_units='xy', scale=1, width=0.003)
plt.title('Campo de desplazamiento (corte axial, toda la segmentación)')
plt.axis('off')
plt.show()

# --- Visualización y comparación de imágenes originales y warpeadas ---
moving = loadmat(f'{nirep_dir}/NIREP_01-Sub.mat')['im']
fixed = loadmat(f'{nirep_dir}/NIREP_02-Sub.mat')['im']

if warp_resized.shape[:3] != moving.shape:
    factors = [moving.shape[i] / warp_resized.shape[i] for i in range(3)]
    warp_resized_img = np.zeros(moving.shape + (3,), dtype=np.float32)
    for i in range(3):
        warp_resized_img[..., i] = zoom(warp_resized[..., i], factors, order=1)
else:
    warp_resized_img = warp_resized

warped_img = warp_image_spatial_transformer(moving, warp_resized_img, mode='nearest')

def normalize(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

moving_norm = normalize(moving)
fixed_norm = normalize(fixed)
warped_img_norm = normalize(warped_img)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(moving_norm[:, :, slice_idx], cmap='gray')
plt.title('Moving')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(fixed_norm[:, :, slice_idx], cmap='gray')
plt.title('Fixed')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(warped_img_norm[:, :, slice_idx], cmap='gray')
plt.title('Warped')
plt.axis('off')

plt.suptitle('Comparación de imágenes (corte axial)')
plt.show()

eam = np.mean(np.abs(moving_norm - warped_img_norm))
ecm = np.mean((moving_norm - warped_img_norm) ** 2)
print("Error absoluto medio (EAM):", eam)
print("Error cuadrático medio (ECM):", ecm)

plt.figure(figsize=(10, 5))
plt.imshow(moving_norm[:, :, slice_idx] - warped_img_norm[:, :, slice_idx], cmap='gray', vmin=-1, vmax=1)
plt.colorbar()
plt.title('Diferencia entre Moving y Warped')
plt.axis('off')
plt.show()


# Plots adicionales, uncomment si 
# Visualización de segmentaciones y warping en 3D
# mask1 = (seg1 > 0).astype(np.float32)
# mask2 = (seg2 > 0).astype(np.float32)
# warped_mask1 = warp_image_spatial_transformer(mask1, warp_resized, mode='nearest')
# warped_mask1_bin = (warped_mask1 > 0.5).astype(np.float32)

# def sample_coords(coords, max_points=5000):
#     if coords.shape[0] > max_points:
#         idx = np.random.choice(coords.shape[0], max_points, replace=False)
#         return coords[idx]
#     return coords

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# mask1_coords = np.argwhere(mask1 > 0.5)
# mask2_coords = np.argwhere(mask2 > 0.5)
# warped_coords = np.argwhere(warped_mask1_bin > 0.5)

# mask1_coords = sample_coords(mask1_coords)
# mask2_coords = sample_coords(mask2_coords)
# warped_coords = sample_coords(warped_coords)

# if mask1_coords.size > 0:
#     ax.scatter(mask1_coords[:, 0], mask1_coords[:, 1], mask1_coords[:, 2], c='blue', alpha=0.15, s=3, label='Segmentación original')
# if mask2_coords.size > 0:
#     ax.scatter(mask2_coords[:, 0], mask2_coords[:, 1], mask2_coords[:, 2], c='green', alpha=0.15, s=3, label='Segmentación referencia')
# if warped_coords.size > 0:
#     ax.scatter(warped_coords[:, 0], warped_coords[:, 1], warped_coords[:, 2], c='red', alpha=0.15, s=3, label='Segmentación warpeada')
# ax.set_title('Visualización 3D de Segmentaciones y Warping')
# ax.legend(loc='upper right', fontsize=12)
# ax.view_init(elev=30, azim=120)  
# ax.set_axis_off()
# plt.tight_layout()
# plt.show()


