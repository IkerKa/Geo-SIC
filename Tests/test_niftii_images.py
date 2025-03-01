import nibabel as nib               # type: ignore
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt     # type: ignore
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import cv2                          # type: ignore


def load_nii_gz_images_from_path(path):
    """Carga todas las imágenes NIfTI de un directorio."""
    images = []
    
    for file in sorted(os.listdir(path), key=lambda x: int(x.split('_')[-1].split('.')[0])):
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

    # for i, img in enumerate(images):
    #     plt.figure()
    #     plt.imshow(img, cmap='gray')
    #     plt.title(f'Imagen {i}')
    #     plt.show()
    
    fig, (ax1, ax2) = plt.subplots(1, 2)

    img_1 = np.rot90(images[0], -1)
    img_2 = np.rot90(images[-1], -1)
    img1 = ax1.imshow(img_1, cmap='gray')
    img2 = ax2.imshow(img_2, cmap='gray')

    ax1.set_title('Imagen 0')
    ax2.set_title('Imagen Final')

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


def evaluate_evolution(images):
    """ Evaluate the evolution of the metrics during the training """
    mse_values, ssim_values, pearson_values = [], [], []

    reference = images[0]  

    for i, img in enumerate(images):
        mse, ssim_value, _, pearson_corr = calculate_metrics(reference, img)
        mse_values.append(mse)
        ssim_values.append(ssim_value)
        pearson_values.append(pearson_corr)

    epochs = np.arange(len(images))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, mse_values, label="MSE", color='red')
    plt.plot(epochs, ssim_values, label="SSIM", color='blue')
    plt.plot(epochs, pearson_values, label="Pearson Corr", color='green')
    plt.xlabel("Época")
    plt.ylabel("Valor de la métrica")
    plt.title("Evolución de las métricas durante el entrenamiento")
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_stability(images):
    """Atlas stability evaluation"""
    stability_mse, stability_ssim = [], []

    for i in range(1, len(images)):
        mse, ssim_value, _, _ = calculate_metrics(images[i-1], images[i])
        stability_mse.append(mse)
        stability_ssim.append(ssim_value)

    # Graficar la estabilidad
    epochs = np.arange(1, len(images))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, stability_mse, label="MSE entre épocas", color='red')
    plt.plot(epochs, stability_ssim, label="SSIM entre épocas", color='blue')
    plt.xlabel("Época")
    plt.ylabel("Cambio entre épocas consecutivas")
    plt.title("Estabilidad del atlas a lo largo del entrenamiento")
    plt.legend()
    plt.grid()
    plt.show()

def calculate_variability(images):
    """Calcula la desviación estándar en cada voxel/píxel a lo largo de las épocas."""
    images_array = np.stack(images, axis=-1)  # Apila las imágenes a lo largo de la última dimensión
    std_map = np.std(images_array, axis=-1)  # Calcula la desviación estándar voxel a voxel

    plt.figure()
    plt.imshow(np.rot90(std_map, -1), cmap='hot')
    plt.title("Mapa de Variabilidad del Atlas")
    plt.colorbar()
    plt.show()

def plot_frequency_spectrum(image1, image2):
    """Muestra el espectro de frecuencias de una imagen."""
    f_transform = np.fft.fftshift(np.fft.fft2(image1))
    f_transform2 = np.fft.fftshift(np.fft.fft2(image2))

    magnitude_spectrum1 = np.log(np.abs(f_transform) + 1)
    magnitude_spectrum2 = np.log(np.abs(f_transform2) + 1)

    ax1 = plt.subplot(121)
    ax1.imshow(magnitude_spectrum1, cmap='gray')

    ax2 = plt.subplot(122)
    ax2.imshow(magnitude_spectrum2, cmap='gray')

    ax1.set_title('FS Atlas inicial')
    ax2.set_title('FS Atlas final')

    plt.show()
    


def plot_differences(I1, I2):
    """Muestra el mapa de diferencias absoluto y un heatmap."""
    diff = np.abs(I1 - I2)
    sign_diff  = (I1 - I2) / (np.max(np.abs(I1 - I2)) + 1e-8)

    
    # Normalizar entre 0 y 1
    diff_norm = diff / np.max(diff) if np.max(diff) != 0 else diff

    #rotate 90 clockwise
    sign_diff = np.rot90(sign_diff, -1)
    diff_norm = np.rot90(diff_norm, -1)
    
    plt.figure()
    plt.imshow(sign_diff, cmap='seismic', vmin=-1, vmax=1)
    plt.title('Diferencias con Signo')
    plt.colorbar()
    plt.show()

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
        plt.imshow(np.rot90(ssim_map, -1), cmap='hot')
        plt.title('Mapa SSIM (Estructural)')
        plt.colorbar()
        plt.show()

        # Metric evolution
        evaluate_evolution(images)

        # Atlas stability
        evaluate_stability(images)

        # Variability map
        calculate_variability(images)

        # Frequency spectrum
        plot_frequency_spectrum(I1, I2)





    else:
        print("No hay suficientes imágenes para comparar.")
