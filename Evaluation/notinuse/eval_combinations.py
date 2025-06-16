import numpy as np
from scipy.io import loadmat
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F

def warp_image_spatial_transformer(moving, displacement, mode='nearest', align_corners=False):
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
        padding_mode='zeros',
        align_corners=align_corners
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

if warp.shape[0] == 1:
    warp = np.squeeze(warp, axis=0)
if warp.shape[0] == 3:
    warp = np.moveaxis(warp, 0, -1)

target_shape = seg1.shape
factors = [target_shape[i] / warp.shape[i] for i in range(3)]
warp_resized = np.zeros(target_shape + (3,), dtype=np.float32)
for i in range(3):
    warp_resized[..., i] = zoom(warp[..., i], factors, order=1)
    warp_resized[..., i] *= factors[i]

# Prueba combinaciones
permutations = [
    None,
    (1, 0, 2, 3),
    (2, 1, 0, 3),
    (0, 2, 1, 3)
]
signs = [1, -1]
align_corners_opts = [False, True]

label = 2
mask1 = (seg1 == label).astype(np.float32)
mask2 = (seg2 == label).astype(np.float32)

print("Probando combinaciones de ejes, signo y align_corners:")
for perm in permutations:
    for sign in signs:
        for align_corners in align_corners_opts:
            warp_test = warp_resized.copy()
            if perm is not None:
                warp_test = np.transpose(warp_test, perm)
            warp_test = warp_test * sign
            try:
                warped_mask1 = warp_image_spatial_transformer(mask1, warp_test, mode='nearest', align_corners=align_corners)
                warped_mask1_bin = (warped_mask1 > 0.5).astype(np.float32)
                dsc = dice_coefficient(mask2, warped_mask1_bin)
                print(f"perm={perm}, sign={sign}, align_corners={align_corners} -> DSC: {dsc:.4f}")
            except Exception as e:
                print(f"perm={perm}, sign={sign}, align_corners={align_corners} -> ERROR: {e}")