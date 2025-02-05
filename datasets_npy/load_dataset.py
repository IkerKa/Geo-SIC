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
        # image_3d = np.repeat(image_3d, 28, axis=0) 
        image_3d = np.repeat(image_3d, 28, axis=0)

        

        image_3d = image_3d.astype(np.float32)

        sitk_image = sitk.GetImageFromArray(image_3d)
        sitk.WriteImage(sitk_image, f"{output_path}/image_{i}.mhd")
        print(f"Image {i} saved as MHD")

def main(args):
    dataset_path = args.path
    folder = args.folder
    num_samples = args.num_samples
    
    data1 = load_dataset(dataset_path)

    if args.show:
        display_samples(data1, num_samples)
    

    #delete the current content from mhd folder that has .raw || .mhd files
    os.system("rm -rf {folder}/*.raw {folder}/*.mhd {folder}/*.csv")
    convert_to_mhd(data1, folder, num_samples)

    #generate a csv with the labels {label},,,
    with open(f"{folder}/label.csv", "w") as f:
        for i in range(num_samples):
            f.write(f"0,,,\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and display dataset.')
    parser.add_argument('--show', action='store_true', help='Flag to display images')
    parser.add_argument('--path', type=str, help='Path to the dataset')
    parser.add_argument('--folder', type=str, help='Folder to save the images')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to display')
    args = parser.parse_args()
    
    main(args)