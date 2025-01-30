import os
import numpy as np
import SimpleITK as sitk
import argparse

import matplotlib.pyplot as plt


def load_dataset(path):
    return np.load(path)

def display_samples(data, num_samples):
    for i in range(num_samples):
        reshaped_data = data[i].reshape((28, 28))  # Assuming the images are 28x28 pixels
        plt.imshow(reshaped_data, cmap='gray')
        plt.title(f'Sample {i+1}')
        plt.show()

def convert_to_mhd(data, output_path, num_samples):
    for i in range(num_samples):
        #random idx
        idx = np.random.randint(0, len(data))
        image_3d = data[idx].reshape((1, 28, 28))  # Adding an extra dimension
        #convert the depth to 28
        image_3d = np.repeat(image_3d, 28, axis=0)
        sitk_image = sitk.GetImageFromArray(image_3d)
        sitk.WriteImage(sitk_image, f"{output_path}/image_{i}.mhd")
        print(f"Image {i} saved as MHD")

def main(show_images):
    dataset_path = 'apple.npy'
    data = load_dataset(dataset_path)
    
    if show_images:
        display_samples(data, 5)
    

    #delete the current content from mhd folder that has .raw || .mhd files
    os.system("rm -rf ./mhd/*.raw ./mhd/*.mhd")
    convert_to_mhd(data, './mhd', 5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and display dataset.')
    parser.add_argument('--show', action='store_true', help='Flag to display images')
    args = parser.parse_args()
    
    main(args.show)