import os
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Forzar un backend interactivo
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Paths
atlas_snapshots_path = './backup/b6'
original_image_path = './datasets/images/chloe.jpg'

# Load atlas images (assuming they are PNG) and convert to grayscale
atlas_images = []
for i, file in enumerate(os.listdir(atlas_snapshots_path)):
    if not file.endswith('.png'):
        continue
    number = int(file.split('_')[-1].split('.')[0])
    atlas_images.append((number, Image.open(os.path.join(atlas_snapshots_path, file)).convert('L')))

# Sort images by epoch number
atlas_images.sort(key=lambda x: x[0])
atlas_images = [img for _, img in atlas_images]

   


if not atlas_images:
    raise ValueError("No se encontraron imágenes en atlas_snapshots_path.")

# Convert images to arrays (assuming all have the same size)
frames = [np.array(img) for img in atlas_images]

# Load original image, convert to grayscale
original_image = Image.open(original_image_path).convert('L')
original_array = np.array(original_image)

# Resize frames to the same size as the original image
frames_resized = [np.array(img.resize(original_array.shape[::-1])) for img in atlas_images]

# Create the figure
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

# Configure the subplot for the animation
im = ax.imshow(frames_resized[0], cmap='gray')
ax.axis('off')
title = ax.set_title("Animated Atlas")

def update(frame_idx):
    frame = frames_resized[frame_idx]
    im.set_data(frame)
    # Update the title with epoch ID
    title.set_text(f"Epoch: {frame_idx + 1}")
    return im, title

# Create the animation; use blit=False for mayor compatibilidad
ani = FuncAnimation(fig, update, frames=len(frames_resized), interval=500, blit=False, repeat=True)

# Opcional: guardar la animación como GIF (requiere ImageMagick o FFmpeg)
ani.save('final_gif.gif', writer='imagemagick', fps=30)

plt.show()
