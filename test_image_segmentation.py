import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Unable to read the image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    
    result = cv2.bitwise_and(image, image, mask=mask)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Segmented Image")
    ax[1].axis("off")
    plt.show()




if __name__ == "__main__":
    input_image_path = "datasets/images/guppy.jpg"  # Replace with your input image path
    # output_image_path = "segmented_image.jpg"  # Replace with your desired output image path
    segment_image(input_image_path)