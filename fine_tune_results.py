import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gaussian_filter(image_path, output_path, kernel_size=3):
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to open image file {image_path}")
        return
    
    # Apply Gaussian filter
    # filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # Alternative: Apply median filter
    # filtered_image = cv2.medianBlur(image, kernel_size)
    
    # Alternative: Apply bilateral filter
    filtered_image = cv2.bilateralFilter(image, kernel_size, 75, 75)
    
    # cv2.imwrite(output_path, filtered_image)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Filtered Image")
    ax[1].axis("off")
    plt.show()

    print(f"Filtered image saved to {output_path}")

if __name__ == "__main__":
    input_image_path = "./backup/b6/atlas_epoch_28.png"
    output_image_path = "./backup/b6/final_atlas_filtered.png"
    apply_gaussian_filter(input_image_path, output_image_path)