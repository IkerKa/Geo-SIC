import cv2
import numpy as np

def apply_median_filter(image_path, output_path, kernel_size=3):
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to open image file {image_path}")
        return
    
    filtered_image = cv2.medianBlur(image, kernel_size)
    
    cv2.imwrite(output_path, filtered_image)
    print(f"Filtered image saved to {output_path}")

if __name__ == "__main__":
    input_image_path = "./backup/b6/atlas_epoch_28.png"
    output_image_path = "./backup/b6/final_atlas_filtered.png"
    apply_median_filter(input_image_path, output_image_path)