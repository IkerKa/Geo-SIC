import numpy as np
import scipy.io
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator

def load_normalized_volume(path, var_name='im'):
    data = scipy.io.loadmat(path)
    volume = data[var_name]
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return volume.astype(np.float32)

def load_segmentation(path, var_name='seg'):
    data = scipy.io.loadmat(path)
    return data[var_name].astype(np.int16)

def interpolate_displacement(u, seg_shape):
    dim_z, dim_y, dim_x = u.shape[:3]
    factor = np.array(seg_shape) / np.array([dim_z, dim_y, dim_x])

    z = np.arange(dim_z)
    y = np.arange(dim_y)
    x = np.arange(dim_x)

    # Crear grilla destino con forma (seg_shape)
    ZI, YI, XI = np.meshgrid(
        np.arange(seg_shape[0]),
        np.arange(seg_shape[1]),
        np.arange(seg_shape[2]),
        indexing='ij'
    )

    uu = np.zeros(seg_shape + (3,), dtype=np.float32)

    for i in range(3):
        interp_func = RegularGridInterpolator(
            (z, y, x),
            u[..., i] * factor[i],
            bounds_error=False,
            fill_value=0
        )
        pts = np.stack([ZI.ravel(), YI.ravel(), XI.ravel()], axis=-1)
        uu[..., i] = interp_func(pts).reshape(seg_shape)

    return uu, XI, YI, ZI


def warp_segmentation(seg, phi):
    coords = [phi[..., i] for i in range(3)]
    warped = map_coordinates(seg, coords, order=0, mode='constant', cval=0)
    return warped.astype(np.int16)

def compute_dsc(segmentation, reference, n_labels=32):
    interVol = np.zeros(n_labels)
    refVol = np.zeros(n_labels)
    segVol = np.zeros(n_labels)
    dsc = np.zeros(n_labels)
    to = np.zeros(n_labels)

    for label in range(1, n_labels + 1):
        ref_mask = (reference == label)
        seg_mask = (segmentation == label)
        interVol[label - 1] = np.sum(seg_mask & ref_mask)
        refVol[label - 1] = np.sum(ref_mask)
        segVol[label - 1] = np.sum(seg_mask)

        if refVol[label - 1] + segVol[label - 1] > 0:
            dsc[label - 1] = 2.0 * interVol[label - 1] / (refVol[label - 1] + segVol[label - 1])
            to[label - 1] = interVol[label - 1] / refVol[label - 1]

    jaccard = np.divide(dsc, (2 - dsc), out=np.zeros_like(dsc), where=(2 - dsc) != 0)
    return dsc, jaccard, to

def NIREP16_GeoSIC_DSC(patient, warp, warped):
    dataset_path = 'Baseline/NIREP_Matlab'

    moving = load_normalized_volume(f'{dataset_path}/NIREP_01-Sub.mat')
    fixed = load_normalized_volume(f'{dataset_path}/NIREP_{patient:02d}-Sub.mat')

    nx, ny, nz = fixed.shape
    nnx, nny, nnz, _ = warp.shape

    # Expand warp to full volume if needed
    nwarp = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    nwarp[:nnx, :nny, :nnz, :] = warp
    warp = nwarp

    z, y, x = np.arange(nz), np.arange(ny), np.arange(nx)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    iX = X + warp[..., 0]
    iY = Y + warp[..., 1]
    iZ = Z + warp[..., 2]

    coords = [iZ.ravel(), iY.ravel(), iX.ravel()]
    w = map_coordinates(moving, coords, order=1, mode='constant', cval=0).reshape((nx, ny, nz))

    # Visual difference (opcional, útil para debugging)
    diff = warped - w[:nnx, :nny, :nnz]
    # Aquí podrías visualizar `diff` con matplotlib o nibabel si lo deseas

    seg = load_segmentation(f'{dataset_path}/NIREP_01-Seg.mat')
    reference = load_segmentation(f'{dataset_path}/NIREP_{patient:02d}-Seg.mat')

    # Interpolación del campo de desplazamiento al tamaño de la segmentación
    uu, XI, YI, ZI = interpolate_displacement(warp, seg.shape)

    phi = np.stack([XI + uu[..., 0], YI + uu[..., 1], ZI + uu[..., 2]], axis=-1)

    seg_warped = warp_segmentation(seg, phi)

    dsc, jaccard, to = compute_dsc(seg_warped, reference)

    print("Mean DSC:", np.mean(dsc))
    return dsc, jaccard, to
