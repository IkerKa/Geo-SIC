import os
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Forzar un backend interactivo
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# Paths
atlas_snapshots_path = './atlas_snapshots/'
original_image_path = './datasets/yovani.jpg'

# Load atlas images (assuming they are PNG) and convert to grayscale
atlas_images = []
for file_name in sorted(os.listdir(atlas_snapshots_path)):
    if file_name.endswith('.png'):
        file_path = os.path.join(atlas_snapshots_path, file_name)
        atlas_images.append(Image.open(file_path).convert('L'))

if not atlas_images:
    raise ValueError("No se encontraron imágenes en atlas_snapshots_path.")

# Convert images to arrays (assuming all have the same size)
frames = [np.array(img) for img in atlas_images]

# Load original image, convert to grayscale
original_image = Image.open(original_image_path).convert('L')
original_array = np.array(original_image)

# Resize frames to the same size as the original image
frames_resized = [np.array(img.resize(original_array.shape[::-1])) for img in atlas_images]

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=100)

# Display the original image in the first subplot
ax1.imshow(original_array, cmap='gray')
ax1.axis('off')
ax1.set_title("Original Image")

# Configure the second subplot for the animation
im = ax2.imshow(frames_resized[0], cmap='gray')
ax2.axis('off')
title = ax2.set_title("Animated Atlas")

def update(frame_idx):
    frame = frames_resized[frame_idx]
    im.set_data(frame)
    # Calculate SSIM and MSE between the original image and the current frame
    # Update the title with epoch ID, SSIM value, and MSE value
    title.set_text(f"Epoch: {frame_idx + 1}")
    return im, title

# Create the animation; use blit=False for mayor compatibilidad
ani = FuncAnimation(fig, update, frames=len(frames_resized), interval=500, blit=False, repeat=True)

# Opcional: guardar la animación como GIF (requiere ImageMagick o FFmpeg)
# ani.save('final_result.gif', writer='imagemagick', fps=2)

plt.show()
