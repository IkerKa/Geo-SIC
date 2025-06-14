import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
import torch
import torch.nn.functional as F
# Mostrar imágenes relevantes
import matplotlib.pyplot as plt



def warp_image_spatial_transformer(moving, displacement, mode='bilinear'):
    """
    moving: numpy array (nx, ny, nz)
    displacement: numpy array (nx, ny, nz, 3)
    """
    moving_torch = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0).float()  
    disp_torch = torch.from_numpy(displacement).float()  

    nx, ny, nz = moving.shape

    # Crear grid normalizado [-1,1]
    grid_x = torch.linspace(-1, 1, nx)
    grid_y = torch.linspace(-1, 1, ny)
    grid_z = torch.linspace(-1, 1, nz)
    grid = torch.stack(torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij'), dim=-1)  # 

    # Normalizar desplazamiento a [-1,1]
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

def NIREP16_GeoSIC_DSC(patient, warp, warped):
    nirep_dir = 'Baseline/NIREP_Matlab/'
    moving = loadmat(f'{nirep_dir}/NIREP_01-Sub.mat')['im']
    fixed = loadmat(f'{nirep_dir}/NIREP_{patient:02d}-Sub.mat')['im']

    moving = (moving - moving.min()) / (moving.max() - moving.min())
    fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())

    nx, ny, nz = fixed.shape
    nnx, nny, nnz, _ = warp.shape

    seg = loadmat(f'{nirep_dir}/NIREP_01-Seg.mat')['seg']
    reference = loadmat(f'{nirep_dir}/NIREP_{patient:02d}-Seg.mat')['seg']

    nwarp = np.zeros((nx, ny, nz, 3))
    nwarp[:nnx, :nny, :nnz, 0] = warp[:, :, :, 0]
    nwarp[:nnx, :nny, :nnz, 1] = warp[:, :, :, 1]
    nwarp[:nnx, :nny, :nnz, 2] = warp[:, :, :, 2]
    warp = nwarp

    # # Antes de llamar a warp_image_spatial_transformer:
    # warp[:, :, :, 0] *= -1  # Invertir dirección X
    # warp[:, :, :, 1] *= -1  # Invertir dirección Y
    # warp[:, :, :, 2] *= -1  # Invertir dirección Z

    warped_img = warp_image_spatial_transformer(moving, warp)


    dim_s = seg.shape
    dim_p = warp.shape[:3]
    factor = np.array(dim_s) / np.array(dim_p)

    xs = np.arange(1, dim_s[0]+1)
    ys = np.arange(1, dim_s[1]+1)
    zs = np.arange(1, dim_s[2]+1)

    xp = np.arange(1, dim_p[0]+1)
    yp = np.arange(1, dim_p[1]+1)
    zp = np.arange(1, dim_p[2]+1)

    Xs, Ys, Zs = np.meshgrid(xs, ys, zs, indexing='ij')
    Xs_warp = Xs / factor[0]
    Ys_warp = Ys / factor[1]
    Zs_warp = Zs / factor[2]

    uu = np.zeros((dim_s[0], dim_s[1], dim_s[2], 3))
    for i in range(3):
        interp_comp = RegularGridInterpolator(
            (xp, yp, zp),
            warp[..., i],
            method='linear',
            bounds_error=False,
            fill_value=0
        )
        points_comp = np.stack([Xs_warp.ravel(), Ys_warp.ravel(), Zs_warp.ravel()], axis=-1)
        uu[..., i] = interp_comp(points_comp).reshape(dim_s)

    iphi = np.zeros_like(uu)
    iphi[..., 0] = Xs + uu[..., 0]
    iphi[..., 1] = Ys + uu[..., 1]
    iphi[..., 2] = Zs + uu[..., 2]

    interp_seg = RegularGridInterpolator(
        (xs, ys, zs),
        seg,
        method='nearest',
        bounds_error=False,
        fill_value=0
    )
    points_seg = np.stack([
        iphi[..., 0].ravel(),
        iphi[..., 1].ravel(),
        iphi[..., 2].ravel()
    ], axis=-1)
    segmentation = interp_seg(points_seg).reshape(seg.shape)

    dsc = np.zeros(32)
    Jaccard = np.zeros(32)
    to = np.zeros(32)

    for label in range(1, 33):
        ref_mask = (reference == label)
        seg_mask = (segmentation == label)

        interVol = np.sum(np.logical_and(ref_mask, seg_mask))
        refVol = np.sum(ref_mask)
        segVol = np.sum(seg_mask)

        dsc_val = 2.0 * interVol / (refVol + segVol) if (refVol + segVol) > 0 else 0
        to_val = interVol / refVol if refVol > 0 else 0

        dsc[label-1] = dsc_val
        to[label-1] = to_val

    Jaccard = dsc / (2 - dsc)

    return dsc, Jaccard, to, warped_img



data = loadmat('results.mat')
nirep01 = np.squeeze(data['nirep01'])
nirep02 = np.squeeze(data['nirep02'])
warped = np.squeeze(data['warped'])
id_data = data['id']
disp_data = data['disp']

X = np.squeeze(id_data[0, 1, :, :, :])  # MATLAB: id(1,2,:,:,:)
Y = np.squeeze(id_data[0, 0, :, :, :])  # MATLAB: id(1,1,:,:,:)
Z = np.squeeze(id_data[0, 2, :, :, :])  # MATLAB: id(1,3,:,:,:)

u = np.squeeze(disp_data[0, 1, :, :, :])  # MATLAB: disp(1,2,:,:,:)
v = np.squeeze(disp_data[0, 0, :, :, :])  # MATLAB: disp(1,1,:,:,:)
w = np.squeeze(disp_data[0, 2, :, :, :])  # MATLAB: disp(1,3,:,:,:)


XI = X + u
YI = Y + v
ZI = Z + w

x_coords = np.unique(Y) 
y_coords = np.unique(X)
z_coords = np.unique(Z)

interp_func = RegularGridInterpolator(
    (x_coords, y_coords, z_coords),
    nirep01.T, 
    method='linear',
    bounds_error=False,
    fill_value=0
)


points = np.stack([YI.ravel(), XI.ravel(), ZI.ravel()], axis=-1)


warp = np.stack([u, v, w], axis=-1)
wwarped_flat = warp_image_spatial_transformer(nirep01, warp)

wwarped = wwarped_flat.reshape(XI.shape)


warp = np.stack([u, v, w], axis=-1)
patient = 2

dsc, Jaccard, to, warped_img = NIREP16_GeoSIC_DSC(patient, warp, wwarped)

print("DSC:", dsc)
print("Jaccard:", Jaccard)
print("To:", to)

print("DSC mean:", np.mean(dsc))


nirep_dir = 'Baseline/NIREP_Matlab/'
moving = loadmat(f'{nirep_dir}/NIREP_01-Sub.mat')['im']
fixed = loadmat(f'{nirep_dir}/NIREP_{patient:02d}-Sub.mat')['im']


moving = (moving - moving.min()) / (moving.max() - moving.min())
fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())
warped_img_norm = (warped_img - warped_img.min()) / (warped_img.max() - warped_img.min())

slice_idx = warped_img.shape[2] // 2

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(moving[:, :, slice_idx], cmap='gray')
plt.title('Moving')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(fixed[:, :, slice_idx], cmap='gray')
plt.title('Fixed')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(warped_img_norm[:, :, slice_idx], cmap='gray')
plt.title('Warped')
plt.axis('off')

plt.suptitle('Comparación de imágenes (corte axial)')
plt.show()

print("Error absoluto medio (EAM):", np.mean(np.abs(moving - warped_img_norm)))
print("Error cuadrático medio (ECM):", np.mean((moving - warped_img_norm) ** 2))

plt.figure(figsize=(10, 5))
plt.imshow(moving[:, :, slice_idx] - warped_img_norm[:, :, slice_idx], cmap='gray', vmin=-1, vmax=1)
plt.colorbar()
plt.title('Diferencia entre Moving y Warped')
plt.axis('off')
plt.show()


