import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from PIL import Image

# Set up argument parser
parser = argparse.ArgumentParser(description="Process and save images from an .ndjson file.")
parser.add_argument("--file", type=str, help="Path to the .ndjson file")
parser.add_argument("--samples", type=int, default=100, help="Number of samples to process")
parser.add_argument("--resize", type=bool, default=False, help="Resize images to 128x128 if True")
parser.add_argument("--size", type=int, default=128, help="Size of the image")
parser.add_argument("--two_dims", type=bool, default=False, help="If True, the images will be 2D")
args = parser.parse_args()

# Clear all the folders
os.system("rm -rf MHD_*")
os.system(f"rm -rf {args.file}/*")

# Paths
ndjson_file = "jsons/" + args.file + ".ndjson"
output_folder = args.file
twoD_folder = "2D_" + output_folder

os.makedirs(output_folder, exist_ok=True)

def resize_image(image_path, output_path, i):
    """
    Resize a 2D image if required.
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    if args.resize:
        image = image.resize((args.size, args.size))
    image.save(f"{output_path}/circle_{i}.png")
    print(f"Image {i} saved as PNG")

# Load and read the lines from the file
with open(ndjson_file, "r") as f:
    drawings = [json.loads(line) for line in f]

# Process and save images
for i, drawing in enumerate(drawings[:args.samples]):
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)  # 256x256 pixels
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_xticks([])  # Hide axes
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_facecolor("white")  # White background

    for stroke in drawing["drawing"]:
        x, y = stroke[0], stroke[1]
        ax.plot(x, 255 - np.array(y), color="black", linewidth=2)  # Invert Y

    plt.savefig(f"{output_folder}/circle_{i}.png", dpi=100, pad_inches=0)
    plt.close()

    # Resize the saved image if required
    resize_image(f"{output_folder}/circle_{i}.png", output_folder, i)

    with open(f"{output_folder}/label.csv", "w") as f:
        for i in range(args.samples):
            f.write(f"0,,,\n")

print(f"Images saved in {output_folder} in 2D format")
