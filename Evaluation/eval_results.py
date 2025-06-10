from scipy.io import loadmat
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt


def compute_dsc_and_jaccard(segmentation, reference, num_labels=32):
    dsc = np.zeros(num_labels)
    to = np.zeros(num_labels)
    jaccard = np.zeros(num_labels)
    
    for label in range(1, num_labels + 1):
        seg_mask = (segmentation == label)
        ref_mask = (reference == label)
        intersection = np.logical_and(seg_mask, ref_mask).sum()
        seg_vol = seg_mask.sum()
        ref_vol = ref_mask.sum()
        
        if ref_vol + seg_vol > 0:
            dsc[label - 1] = 2.0 * intersection / (ref_vol + seg_vol)
        else:
            dsc[label - 1] = np.nan 
        
        if ref_vol > 0:
            to[label - 1] = intersection / ref_vol
        else:
            to[label - 1] = np.nan 
        
        if (2 - dsc[label - 1]) != 0:
            jaccard[label - 1] = dsc[label - 1] / (2 - dsc[label - 1])
        else:
            jaccard[label - 1] = np.nan

    mean_dsc = np.nanmean(dsc)
    mean_jaccard = np.nanmean(jaccard)
    return dsc, jaccard, to, mean_dsc, mean_jaccard

data = loadmat('results.mat')
nirep01 = np.squeeze(data['nirep01'])
nirep02 = np.squeeze(data['nirep02'])
warped = np.squeeze(data['warped'])
id_grid = data['id']
disp = data['disp']

# id tiene shape [1, 3, D, H, W]
X = np.squeeze(id_grid[0, 1, :, :, :])  # MATLAB: id(1,2,:,:,:)
Y = np.squeeze(id_grid[0, 0, :, :, :])  # MATLAB: id(1,1,:,:,:)
Z = np.squeeze(id_grid[0, 2, :, :, :])  # MATLAB: id(1,3,:,:,:)

u = np.squeeze(disp[0, 1, :, :, :])     # disp(1,2,:,:,:)
v = np.squeeze(disp[0, 0, :, :, :])     # disp(1,1,:,:,:)
w = np.squeeze(disp[0, 2, :, :, :])     # disp(1,3,:,:,:)

XI = X + u
YI = Y + v
ZI = Z + w

# Las coordenadas deben estar en orden (z, y, x) para map_coordinates
coords = np.array([ZI.flatten(), YI.flatten(), XI.flatten()])
wwarped = map_coordinates(nirep01, coords, order=1, mode='constant', cval=0.0)
wwarped = wwarped.reshape(nirep01.shape)

fig, axs = plt.subplots(1, 2, figsize=(16, 4))

# Central slice index
# Axial slice (z axis)
z_idx = nirep01.shape[0] // 2

axs[0].imshow(np.rot90(warped[:, :, z_idx]), cmap='gray', vmin=0, vmax=1)
axs[0].set_title('warped (axial)')
axs[0].axis('off')

diff = np.abs(nirep02[:, :, z_idx] - warped[:, :, z_idx])
axs[1].imshow(np.rot90(diff), cmap='gray', vmin=0, vmax=1)
axs[1].set_title('nirep02 - warped (axial)')
axs[1].axis('off')

plt.tight_layout()
plt.show()

warp = np.zeros(XI.shape + (3,))
warp[..., 0] = u
warp[..., 1] = v
warp[..., 2] = w

# Plot the displacement field
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(np.rot90(u[:, :, z_idx]), cmap='gray', vmin=-1, vmax=1)
axs[0].set_title('Displacement u (axial)')
axs[0].axis('off')
axs[1].imshow(np.rot90(v[:, :, z_idx]), cmap='gray', vmin=-1, vmax=1)
axs[1].set_title('Displacement v (axial)')
axs[1].axis('off')
axs[2].imshow(np.rot90(w[:, :, z_idx]), cmap='gray', vmin=-1, vmax=1)
axs[2].set_title('Displacement w (axial)')
axs[2].axis('off')
plt.tight_layout()
plt.show()

# Quiver plot (en un plano axial)
# step = 2  # para menos flechas
# plt.figure(figsize=(8, 8))
# plt.imshow(np.rot90(nirep01[:, :, z_idx]), cmap='gray', vmin=0, vmax=1)
# plt.quiver(
#     Y[::step, ::step, z_idx], X[::step, ::step, z_idx],
#     v[::step, ::step, z_idx], u[::step, ::step, z_idx],
#     color='r', angles='xy', scale_units='xy', scale=1
# )
# plt.title('Displacement field (axial, quiver)')
# plt.axis('off')
# plt.show()

# plt.figure(figsize=(8, 4))
# plt.hist(u.flatten(), bins=50, alpha=0.5, label='u')
# plt.hist(v.flatten(), bins=50, alpha=0.5, label='v')
# plt.hist(w.flatten(), bins=50, alpha=0.5, label='w')
# plt.legend()
# plt.title('Histogram of displacement components')
# plt.xlabel('Displacement')
# plt.ylabel('Frequency')
# plt.show()

# Línea central en el eje z
# profile_warped = warped[:, :, z_idx].mean(axis=0)
# profile_nirep02 = nirep02[:, :, z_idx].mean(axis=0)
# plt.figure()
# plt.plot(profile_warped, label='warped')
# plt.plot(profile_nirep02, label='nirep02')
# plt.title('Mean intensity profile (axial slice)')
# plt.legend()
# plt.show()

# diff_3d = np.abs(nirep02 - warped)
# plt.figure(figsize=(8, 4))
# plt.imshow(np.max(diff_3d, axis=0), cmap='hot')
# plt.title('Max projection of |nirep02 - warped| (Y-Z plane)')
# plt.axis('off')
# plt.show()

# --NO TENGO MUY CLARO SI ESTO LO TENGO BIEN--

id_grid = data['id']
disp = data['disp']

# Carga las segmentaciones (ajusta el path y formato según tus datos)
dataset_path = 'Baseline/NIREP_Matlab/'
seg = loadmat(dataset_path + 'NIREP_01-Seg.mat')['seg']  # Segmentación moving (NIREP_01)
reference = loadmat(dataset_path + 'NIREP_02-Seg.mat')['seg']  # Segmentación reference (NIREP_02)

output_shape = reference.shape  # (256, 300, 256)
z, y, x = np.meshgrid(
    np.arange(output_shape[0]),
    np.arange(output_shape[1]),
    np.arange(output_shape[2]),
    indexing='ij'
)

coords = np.array([
    z + map_coordinates(u, [z, y, x], order=1, mode='nearest'),
    y + map_coordinates(v, [z, y, x], order=1, mode='nearest'),
    x + map_coordinates(w, [z, y, x], order=1, mode='nearest')
])

seg_warped = map_coordinates(seg, coords, order=0, mode='constant', cval=0)
seg_warped = seg_warped.reshape(seg.shape)

# Calcula DSC y Jaccard
dsc, jaccard, to, mean_dsc, mean_jaccard = compute_dsc_and_jaccard(seg_warped, reference, num_labels=32)

print("DSC por etiqueta:", dsc)
print("Jaccard por etiqueta:", jaccard)
print("To por etiqueta:", to)
print("Mean DSC:", mean_dsc)
print("Mean Jaccard:", mean_jaccard)