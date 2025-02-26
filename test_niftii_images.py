import nibabel as nib               # type: ignore
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt     # type: ignore
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr


def load_nii_gz_images_from_path(path):
    """Carga todas las imágenes NIfTI de un directorio."""
    images = []
    for file in os.listdir(path):
        if file.endswith(".nii.gz"):
            img = nib.load(os.path.join(path, file))
            images.append(img.get_fdata())
    return images


def calculate_metrics(image1, image2):
    """Calcula métricas de comparación entre dos imágenes."""
    # Convertir a flotantes para cálculos
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # Error cuadrático medio (MSE)
    mse = np.mean((image2 - image1) ** 2)

    # Índice de similitud estructural (SSIM)
    ssim_value, ssim_map = ssim(image1, image2, full=True, data_range=image2.max() - image2.min())

    # Coeficiente de correlación de Pearson
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    pearson_corr, _ = pearsonr(image1_flat, image2_flat)

    return mse, ssim_value, ssim_map, pearson_corr


def plot_images(images):
    """Muestra un collage de dos imágenes y sincroniza el zoom."""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    img1 = ax1.imshow(images[0], cmap='gray')
    img2 = ax2.imshow(images[-1], cmap='gray')

    ax1.set_title('Imagen 0')
    ax2.set_title('Imagen i')

    # Sincronizar el zoom
    def on_zoom(event):
        if event.inaxes == ax1:
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim(ax1.get_ylim())
        elif event.inaxes == ax2:
            ax1.set_xlim(ax2.get_xlim())
            ax1.set_ylim(ax2.get_ylim())
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_zoom)
    plt.show()

def plot_differences(I1, I2):
    """Muestra el mapa de diferencias absoluto y un heatmap."""
    diff = np.abs(I1 - I2)
    
    # Normalizar entre 0 y 1
    diff_norm = diff / np.max(diff) if np.max(diff) != 0 else diff

    # Mapa de diferencias en escala de grises
    plt.figure()
    plt.imshow(diff_norm, cmap='gray')
    plt.title('Diferencias (Escala de grises)')
    plt.colorbar()
    plt.show()

    # Mapa de calor (heatmap)
    plt.figure()
    plt.imshow(diff_norm, cmap='hot')
    plt.title('Diferencias (Mapa de calor)')
    plt.colorbar()
    plt.show()

    # Mapa binario con umbral (ejemplo: diferencias > 20% del máximo)
    threshold = 0.2
    binary_diff = (diff_norm > threshold).astype(np.uint8)

    plt.figure()
    plt.imshow(binary_diff, cmap='coolwarm')
    plt.title(f'Diferencias Significativas (Threshold = {threshold * 100}%)')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and analyze nii.gz images from a specified path.')
    parser.add_argument('path', type=str, help='Path to the directory containing nii.gz images')
    args = parser.parse_args()

    images = load_nii_gz_images_from_path(args.path)

    if len(images) >= 2:
        I1, I2 = images[0], images[-1]  # Compara la primera y última imagen

        # Calcular métricas
        mse, ssim_value, ssim_map, pearson_corr = calculate_metrics(I1, I2)

        # Mostrar métricas en consola
        print(f"MSE: {mse:.4f}")
        print(f"SSIM: {ssim_value:.4f}")
        print(f"Pearson Correlation: {pearson_corr:.4f}")

        # Mostrar imágenes y diferencias
        plot_images(images)
        plot_differences(I1, I2)

        # Mostrar el mapa de SSIM
        plt.figure()
        plt.imshow(ssim_map, cmap='coolwarm')
        plt.title('Mapa SSIM (Estructural)')
        plt.colorbar()
        plt.show()

    else:
        print("No hay suficientes imágenes para comparar.")
