import numpy as np
from scipy.io import loadmat
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator

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

# --- Grilla de coordenadas (origen 0 como en MATLAB meshgrid) ---
nx, ny, nz = seg1.shape
x = np.arange(nx)
y = np.arange(ny)
z = np.arange(nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# --- Aplica el warp ---
coords = np.stack([
    X + warp_resized[..., 0],
    Y + warp_resized[..., 1],
    Z + warp_resized[..., 2]
], axis=-1)

# --- Interpolador de la segmentación ---
interp = RegularGridInterpolator(
    (x, y, z),
    seg1,
    method='nearest',
    bounds_error=False,
    fill_value=0
)

# --- Warpea la segmentación ---
coords_flat = coords.reshape(-1, 3)
warped_seg1 = interp(coords_flat).reshape(seg1.shape)

# --- DSC ---
def dice_coefficient(seg1, seg2):
    intersection = np.sum((seg1 > 0) & (seg2 > 0))
    size1 = np.sum(seg1 > 0)
    size2 = np.sum(seg2 > 0)
    if size1 + size2 == 0:
        return 1.0
    return 2.0 * intersection / (size1 + size2)

label = 2
mask2 = (seg2 == label).astype(np.float32)
warped_mask1_bin = (warped_seg1 == label).astype(np.float32)
dsc = dice_coefficient(mask2, warped_mask1_bin)
print(f"DSC (etiqueta {label}): {dsc:.4f}")