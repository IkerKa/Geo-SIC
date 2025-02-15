import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import SimpleITK as sitk
from PIL import Image

# Set up argument parser
parser = argparse.ArgumentParser(description="Process and save images from an .ndjson file.")
parser.add_argument("--file", type=str, help="Path to the .ndjson file")
parser.add_argument("--samples", type=int, default=100, help="Number of samples to process")
parser.add_argument("--resize", type=bool, default=False, help="Resize images to 128x128 if True")
#if resize, add another parameter to the parser that will be the size of the image
parser.add_argument("--size", type=int, default=128, help="Size of the image")
parser.add_argument("--two_dims", type=bool, default=False, help="If True, the images will be 2D")
args = parser.parse_args()

#clear all the folders
# os.system("rm -rf jsons/*")
os.system("rm -rf MHD_*")
os.system(f"rm -rf {args.file}/*")


# Paths
ndjson_file = "jsons/" + args.file + ".ndjson"
output_folder = args.file
mhd_folder = "MHD_" + output_folder
twoD_folder = "2D_" + output_folder

os.makedirs(output_folder, exist_ok=True)
os.makedirs(mhd_folder, exist_ok=True)

def convert_to_mhd(image_path, output_path, i):
    """
    Convert a 2D image to a 3D MHD image by stacking it along a new axis.
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image_array = np.array(image)  # Shape: (256, 256)
    #resize the image to 128x128
    if args.resize:
        image = image.resize((args.size, args.size))
        image_array = np.array(image)  # Shape: (128, 128)
        
    # Expand dimensions to make it (256, 256, 256)
    image_3d = np.repeat(image_array[np.newaxis, :, :], [args.size if args.resize else 256], axis=0)
    print(f"Image shape: {image_3d.shape}")  # (256, 256, 256)

    # Convert to SimpleITK image
    sitk_image = sitk.GetImageFromArray(image_3d.astype(np.float32))
    sitk.WriteImage(sitk_image, f"{output_path}/image_{i}.mhd")
    print(f"Image {i} saved as MHD")


def visualize_mhd_image(mhd_file_path):
    """
    Visualize a 3D MHD image by displaying its middle slice.
    """
    # Read the MHD image
    sitk_image = sitk.ReadImage(mhd_file_path)
    image_array = sitk.GetArrayFromImage(sitk_image)  # Shape: (256, 256, 256)
    
    #print the shape of the image
    print(f"Image shape: {image_array.shape}")
    # Get the middle slice
    #get the middle slice of the image
    middle_slice = image_array[args.size//2 if args.resize else 256//2, :, :]  # Shape: (256, 256)
    
    # Display the middle slice
    plt.imshow(middle_slice, cmap="gray")
    plt.title("Middle Slice of MHD Image")
    plt.axis("off")
    plt.show()

 
# Load and read the lines from the file
with open(ndjson_file, "r") as f:
    drawings = [json.loads(line) for line in f]

# Process and save images
for i, drawing in enumerate(drawings[:args.samples]):
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)  # 256x256 pixels
    ax.set_xlim(0, 255)  # Fijar el tamaño
    ax.set_ylim(0, 255)
    ax.set_xticks([])  # Ocultar ejes
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_facecolor("white")  # Fondo blanco

    for stroke in drawing["drawing"]:
        x, y = stroke[0], stroke[1]
        ax.plot(x, 255 - np.array(y), color="black", linewidth=2)  # Invertir Y

    plt.savefig(f"{output_folder}/circle_{i}.png", dpi=100, pad_inches=0)
    plt.close()


    # Convert the saved image to MHD
    if not args.two_dims:
        convert_to_mhd(f"{output_folder}/circle_{i}.png", mhd_folder, i)
        #visualize a random mhd image
        random_idx = np.random.randint(0, args.samples)
        visualize_mhd_image(f"{mhd_folder}/image_{random_idx}.mhd")

        with open(f"{mhd_folder}/label.csv", "w") as f:
            for i in range(args.samples):
                f.write(f"0,,,\n")
    else:  
        with open(f"{output_folder}/label.csv", "w") as f:
            for i in range(args.samples):
                f.write(f"0,,,\n")



if args.two_dims:
    print(f"Images saved in {output_folder} in 2D format")
else:
    print(f"Images saved in {output_folder} in 2D format and converted to MHD in {mhd_folder}")


