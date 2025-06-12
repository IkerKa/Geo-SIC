import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator

# Cargar datos desde results.mat
data = loadmat('results.mat')
nirep01 = np.squeeze(data['nirep01'])
nirep02 = np.squeeze(data['nirep02'])
warped = np.squeeze(data['warped'])
id_data = data['id']
disp_data = data['disp']

# Extraer componentes X, Y, Z y desplazamientos u, v, w
X = np.squeeze(id_data[0, 1, :, :, :])  # MATLAB: id(1,2,:,:,:)
Y = np.squeeze(id_data[0, 0, :, :, :])  # MATLAB: id(1,1,:,:,:)
Z = np.squeeze(id_data[0, 2, :, :, :])  # MATLAB: id(1,3,:,:,:)

u = np.squeeze(disp_data[0, 1, :, :, :])  # MATLAB: disp(1,2,:,:,:)
v = np.squeeze(disp_data[0, 0, :, :, :])  # MATLAB: disp(1,1,:,:,:)
w = np.squeeze(disp_data[0, 2, :, :, :])  # MATLAB: disp(1,3,:,:,:)

# Calcular coordenadas desplazadas
XI = X + u
YI = Y + v
ZI = Z + w

# Interpolación 3D - CORRECCIÓN DE DIMENSIONES
x_coords = np.unique(Y)  # Extraer coordenadas únicas
y_coords = np.unique(X)
z_coords = np.unique(Z)

# Construir interpolador con orden de dimensiones corregido
interp_func = RegularGridInterpolator(
    (x_coords, y_coords, z_coords),
    nirep01.T,  # Transponer para alinear dimensiones
    method='linear',
    bounds_error=False,
    fill_value=0
)

# Preparar puntos para interpolación
points = np.stack([YI.ravel(), XI.ravel(), ZI.ravel()], axis=-1)

# Interpolar y remodelar
wwarped_flat = interp_func(points)
wwarped = wwarped_flat.reshape(XI.shape)

# Función equivalente a NIREP16_GeoSIC_DSC (versión corregida)
def NIREP16_GeoSIC_DSC(patient, warp, warped):
    # Simular carga de datos VTK como arrays desde .mat
    nirep_dir = 'Baseline/NIREP_Matlab/'
    
    # Cargar imágenes móvil y fija
    moving = loadmat(f'{nirep_dir}/NIREP_01-Sub.mat')['im']
    fixed = loadmat(f'{nirep_dir}/NIREP_{patient:02d}-Sub.mat')['im']
    
    # Normalizar imágenes
    moving = (moving - moving.min()) / (moving.max() - moving.min())
    fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())
    
    nx, ny, nz = fixed.shape
    nnx, nny, nnz, _ = warp.shape
    
    # Cargar segmentaciones
    seg = loadmat(f'{nirep_dir}/NIREP_01-Seg.mat')['seg']
    reference = loadmat(f'{nirep_dir}/NIREP_{patient:02d}-Seg.mat')['seg']
    
    # Ajustar dimensiones
    nx, ny, nz = fixed.shape
    nnx, nny, nnz, _ = warp.shape
    
    # Ajustar dominio GeoSIC a NIREP
    nwarp = np.zeros((nx, ny, nz, 3))
    nwarp[:nnx, :nny, :nnz, 0] = warp[:, :, :, 0]
    nwarp[:nnx, :nny, :nnz, 1] = warp[:, :, :, 1]
    nwarp[:nnx, :nny, :nnz, 2] = warp[:, :, :, 2]
    warp = nwarp

    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X_grid, Y_grid, Z_grid = np.meshgrid(x, y, z, indexing='ij')  # (nx, ny, nz)

    # Calcular coordenadas invertidas
    iX = X_grid + warp[:, :, :, 0]
    iY = Y_grid + warp[:, :, :, 1]
    iZ = Z_grid + warp[:, :, :, 2]

    # Interpolador para imagen móvil
    interp_moving = RegularGridInterpolator(
        (x, y, z),
        moving,
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    
    # Interpolar
    points_w = np.stack([iX.ravel(), iY.ravel(), iZ.ravel()], axis=-1)
    w_flat = interp_moving(points_w)
    w = w_flat.reshape(iX.shape)
    
    # Upsample del desplazamiento
    # ...existing code inside NIREP16_GeoSIC_DSC...

    # Upsample del desplazamiento
    dim_s = seg.shape
    dim_p = warp.shape[:3]
    factor = np.array(dim_s) / np.array(dim_p)

    # Grilla original del warp
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

    # Calcular transformación inversa
    iphi = np.zeros_like(uu)
    iphi[..., 0] = Xs + uu[..., 0]
    iphi[..., 1] = Ys + uu[..., 1]
    iphi[..., 2] = Zs + uu[..., 2]

    # Interpolación de la segmentación
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
    segmentation = interp_seg(points_seg).reshape(dim_s)
    seg_w_flat = interp_seg(points_seg)
    segmentation = seg_w_flat.reshape(seg.shape)
    
    # Calcular DSC y Jaccard
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
    
    return dsc, Jaccard, to

# Crear tensor warp con dimensiones correctas
warp = np.stack([u, v, w], axis=-1)
patient = 2

# Calcular métricas
dsc, Jaccard, to = NIREP16_GeoSIC_DSC(patient, warp, warped)
print("DSC:", dsc)
print("Jaccard:", Jaccard)
print("Overlap:", to)

#mean
print("Mean DSC:", np.nanmean(dsc))