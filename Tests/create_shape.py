import cv2
import numpy as np
import matplotlib.pyplot as plt


# Define the image size
width, height = 256, 256

image = np.zeros((height, width, 3), dtype=np.uint8)

center = (width // 2, height // 2)
radius = 100
color = (255, 255, 255)  # White color
thickness = -1  

cv2.circle(image, center, radius, color, thickness)

# Plot with matplotlib

# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Circle')
# plt.axis('off')  # Hide axes
# plt.show()

cv2.imwrite('./datasets/images/circle.png', image)