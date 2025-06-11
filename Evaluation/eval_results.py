import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interpn, RegularGridInterpolator
import os

dataset_path = 'Baseline/NIREP_Matlab/'
patient = 2

# --- Cargar datos ---
data = loadmat('results.mat')
nirep01 = np.squeeze(data['nirep01'])
nirep02 = np.squeeze(data['nirep02'])
warped = np.squeeze(data['warped'])
id_data = data['id']
disp_data = data['disp']

print("nirep01 shape:", nirep01.shape)
print("id_data shape:", id_data.shape)
print("disp_data shape:", disp_data.shape)

# Extraer coordenadas y desplazamientos igual que en MATLAB
X = np.squeeze(id_data[0, 1, :, :, :]).astype(np.float32)
Y = np.squeeze(id_data[0, 0, :, :, :]).astype(np.float32)
Z = np.squeeze(id_data[0, 2, :, :, :]).astype(np.float32)
u = np.squeeze(disp_data[0, 1, :, :, :]).astype(np.float32)
v = np.squeeze(disp_data[0, 0, :, :, :]).astype(np.float32)
w = np.squeeze(disp_data[0, 2, :, :, :]).astype(np.float32)

print("X shape:", X.shape, "min/max:", X.min(), X.max())
print("Y shape:", Y.shape, "min/max:", Y.min(), Y.max())
print("Z shape:", Z.shape, "min/max:", Z.min(), Z.max())
print("u shape:", u.shape, "min/max:", u.min(), u.max())
print("v shape:", v.shape, "min/max:", v.min(), v.max())
print("w shape:", w.shape, "min/max:", w.min(), w.max())

XI = X + u
YI = Y + v
ZI = Z + w

print("XI min/max:", XI.min(), XI.max())
print("YI min/max:", YI.min(), YI.max())
print("ZI min/max:", ZI.min(), ZI.max())

XI_adj = XI + 1
YI_adj = YI + 1
ZI_adj = ZI + 1

print("XI_adj min/max:", XI_adj.min(), XI_adj.max())
print("YI_adj min/max:", YI_adj.min(), YI_adj.max())
print("ZI_adj min/max:", ZI_adj.min(), ZI_adj.max())

points = (np.arange(XI_adj.shape[0]), np.arange(XI_adj.shape[1]), np.arange(XI_adj.shape[2]))

wwarped = interpn(
    points,
    nirep01,
    np.stack((XI_adj.ravel(), YI_adj.ravel(), ZI_adj.ravel()), axis=-1),
    method='linear',
    bounds_error=False,
    fill_value=0
)
wwarped = wwarped.reshape(XI.shape)
print("wwarped shape:", wwarped.shape, "min/max:", wwarped.min(), wwarped.max())

warp = np.zeros((*XI.shape, 3), dtype=np.float32)
warp[..., 0] = u
warp[..., 1] = v
warp[..., 2] = w
print("warp shape:", warp.shape)

# --- Cargar imágenes y segmentaciones ---
moving_data = loadmat(os.path.join(dataset_path, 'NIREP_01-Sub.mat'))
fixed_data = loadmat(os.path.join(dataset_path, f'NIREP_{patient:02d}-Sub.mat'))
moving = moving_data['im']
fixed = fixed_data['im']

print("moving shape:", moving.shape, "min/max:", moving.min(), moving.max())
print("fixed shape:", fixed.shape, "min/max:", fixed.min(), fixed.max())

moving = (moving - moving.min()) / (moving.max() - moving.min())
fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())

seg_data = loadmat(os.path.join(dataset_path, 'NIREP_01-Seg.mat'))
seg = seg_data['seg']
ref_data = loadmat(os.path.join(dataset_path, f'NIREP_{patient:02d}-Seg.mat'))
reference = ref_data['seg']

print("seg shape:", seg.shape, "unique:", np.unique(seg))
print("reference shape:", reference.shape, "unique:", np.unique(reference))

# --- Ajustar tamaño del warp ---
dim_s = seg.shape
dim_p = warp.shape[:3]
factor = np.array(dim_s) / np.array(dim_p)
print("factor:", factor)

x_p = np.arange(dim_p[0])
y_p = np.arange(dim_p[1])
z_p = np.arange(dim_p[2])
X_s, Y_s, Z_s = np.meshgrid(
    np.arange(dim_s[0]), np.arange(dim_s[1]), np.arange(dim_s[2]), indexing='ij'
)
print("X_s shape:", X_s.shape)

# Interpoladores para cada componente del campo de desplazamiento
uu = np.zeros((*dim_s, 3), dtype=np.float32)
for d in range(3):
    interp = interpn(
        (x_p, y_p, z_p),
        warp[..., d] * factor[d],
        np.stack((X_s.ravel() / factor[0], Y_s.ravel() / factor[1], Z_s.ravel() / factor[2]), axis=-1),
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    uu[..., d] = interp.reshape(dim_s)
    print(f"uu[..., {d}] min/max:", uu[..., d].min(), uu[..., d].max())

iphi = np.zeros((*dim_s, 3), dtype=np.float32)
iphi[..., 0] = X_s + uu[..., 0]
iphi[..., 1] = Y_s + uu[..., 1]
iphi[..., 2] = Z_s + uu[..., 2]
print("iphi shape:", iphi.shape)
print("iphi[...,0] min/max:", iphi[...,0].min(), iphi[...,0].max())
print("iphi[...,1] min/max:", iphi[...,1].min(), iphi[...,1].max())
print("iphi[...,2] min/max:", iphi[...,2].min(), iphi[...,2].max())

# --- Interpolación nearest para segmentación usando RegularGridInterpolator y redondeo ---
seg = seg.astype(np.float32)
interp_seg = RegularGridInterpolator(
    (np.arange(dim_s[0]), np.arange(dim_s[1]), np.arange(dim_s[2])),
    seg,
    method='nearest',
    bounds_error=False,
    fill_value=0
)
coords = np.stack([iphi[..., 0].ravel(), iphi[..., 1].ravel(), iphi[..., 2].ravel()], axis=-1)
coords_rounded = np.round(coords)
print("coords_rounded shape:", coords_rounded.shape, "min/max:", coords_rounded.min(), coords_rounded.max())
seg_w = interp_seg(coords_rounded)
seg_w = seg_w.reshape(dim_s)
print("seg_w shape:", seg_w.shape, "unique labels:", np.unique(seg_w))

# --- Calcular DSC y Jaccard ---
dsc = []
jaccard = []
for label in range(1, 33):
    ref_mask = (reference == label)
    seg_mask = (seg_w == label)
    inter = np.logical_and(ref_mask, seg_mask).sum()
    ref_sum = ref_mask.sum()
    seg_sum = seg_mask.sum()
    if ref_sum + seg_sum > 0:
        dsc_val = 2.0 * inter / (ref_sum + seg_sum)
        dsc.append(dsc_val)
        jaccard.append(dsc_val / (2 - dsc_val))
        print(f"Label {label}: DSC={dsc_val:.4f}, Jaccard={jaccard[-1]:.4f}, ref={ref_sum}, seg={seg_sum}, inter={inter}")
    else:
        dsc.append(np.nan)
        jaccard.append(np.nan)
        print(f"Label {label}: No voxels in reference or segmentation.")

print('Mean DSC:', np.nanmean(dsc))
print('Mean Jaccard:', np.nanmean(jaccard))