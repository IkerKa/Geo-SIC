import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Parámetros del volumen
dim = 64
sphere_radius = 0.4
pixel_shift = 15

# Crear rejilla 3D con coordenadas normalizadas
coords_x, coords_y, coords_z = np.meshgrid(np.linspace(-1, 1, dim),
                                          np.linspace(-1, 1, dim),
                                          np.linspace(-1, 1, dim),
                                          indexing='ij')

dist_from_center = np.sqrt(coords_x**2 + coords_y**2 + coords_z**2)

# Volumen base: esfera binaria
base_volume = (dist_from_center < sphere_radius).astype(float)

# Segmentación: tres regiones concéntricas
label_volume = np.zeros_like(dist_from_center, dtype=int)
label_volume[dist_from_center < sphere_radius] = 1
label_volume[dist_from_center < sphere_radius * 0.7] = 2
label_volume[dist_from_center < sphere_radius * 0.4] = 3

def apply_warp(volume, displacement_field, interp_mode='bilinear'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vol_tensor = torch.tensor(volume, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    disp_tensor = torch.tensor(displacement_field, dtype=torch.float32, device=device)
    
    nx, ny, nz = volume.shape
    grid_x = torch.linspace(-1, 1, nx, device=device)
    grid_y = torch.linspace(-1, 1, ny, device=device)
    grid_z = torch.linspace(-1, 1, nz, device=device)
    grid = torch.stack(torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij'), dim=-1)
    
    # Escalar desplazamiento a [-1, 1]
    scaled_disp = disp_tensor.clone()
    scaled_disp[..., 0] = 2 * scaled_disp[..., 0] / (nx - 1)
    scaled_disp[..., 1] = 2 * scaled_disp[..., 1] / (ny - 1)
    scaled_disp[..., 2] = 2 * scaled_disp[..., 2] / (nz - 1)
    
    warped_grid = grid - scaled_disp
    warped_grid = warped_grid.unsqueeze(0)
    
    warped = F.grid_sample(vol_tensor, warped_grid, mode=interp_mode,
                           padding_mode='zeros', align_corners=False)
    
    return warped.squeeze().cpu().numpy()

# Desplazamiento constante solo en X
displacement = np.zeros((dim, dim, dim, 3), dtype=np.float32)
displacement[..., 0] = pixel_shift

# Aplicar warp
warped_vol = apply_warp(base_volume, displacement)
warped_labels = apply_warp(label_volume.astype(float), displacement, interp_mode='nearest')
warped_labels = np.round(warped_labels).astype(int)

# Crear segmentación desplazada manualmente para referencia
ref_labels = np.zeros_like(label_volume)
shift_int = int(pixel_shift)
ref_labels[shift_int:, :, :] = label_volume[:-shift_int, :, :]

def dice_coefficient(seg_a, seg_b, classes):
    scores = []
    for cls in classes:
        if cls == 0:
            continue
        mask_a = (seg_a == cls)
        mask_b = (seg_b == cls)
        
        valid_area = np.ones_like(seg_a, dtype=bool)
        valid_area[:shift_int, :, :] = False
        
        intersection = np.sum(mask_a & mask_b & valid_area)
        volume_sum = np.sum(mask_a & valid_area) + np.sum(mask_b & valid_area)
        
        score = (2 * intersection / volume_sum) if volume_sum > 0 else 0.0
        scores.append(score)
    return scores

# Visualizar resultados
mid_slice = dim // 2
plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1)
plt.imshow(label_volume[:, :, mid_slice], cmap='jet', vmin=0, vmax=3)
plt.title('Segmentación Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(ref_labels[:, :, mid_slice], cmap='jet', vmin=0, vmax=3)
plt.title('Segmentación Referencia')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(warped_labels[:, :, mid_slice], cmap='jet', vmin=0, vmax=3)
plt.title('Segmentación Warp')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(np.abs(label_volume[:, :, mid_slice] - ref_labels[:, :, mid_slice]), cmap='hot', vmin=0, vmax=3)
plt.title('Diferencia Original - Ref')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(np.abs(ref_labels[:, :, mid_slice] - warped_labels[:, :, mid_slice]), cmap='hot', vmin=0, vmax=3)
plt.title('Diferencia Ref - Warp')
plt.axis('off')

classes = [1, 2, 3]
dice_scores = dice_coefficient(ref_labels, warped_labels, classes)

plt.subplot(2, 3, 6)
plt.bar([f'Clase {c}' for c in classes], dice_scores, color=['blue', 'green', 'red'])
plt.ylim(0, 1.1)
plt.axhline(0.9, color='r', linestyle='--', label='Umbral 0.9')
plt.legend()
plt.title('DSC por Clase')
plt.ylabel('Coeficiente DSC')

plt.tight_layout()
plt.show()

print(f"\nDSC para desplazamiento de {pixel_shift} píxeles:")
for c, sc in zip(classes, dice_scores):
    print(f"Clase {c}: {sc:.4f}")
