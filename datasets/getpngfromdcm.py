import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk

def dcm_to_png(dcm_file, png_file, slice_idx, resize=None):
    # Leer el archivo DICOM
    itk_image = sitk.ReadImage(dcm_file)
    image_np = sitk.GetArrayFromImage(itk_image)

    slice_np = image_np[slice_idx]
    
    image = Image.fromarray(slice_np).convert("L")
    if resize is not None:
        image = image.resize(resize)

    # Guardar la imagen como archivo PNG
    image.save(png_file)

def select_slice(dcm_file):
    itk_image = sitk.ReadImage(dcm_file)
    slices = sitk.GetArrayFromImage(itk_image)

    while True:
        plt.imshow(slices[0], cmap='gray')
        plt.title(f"Slice 1/{slices.shape[0]}")
        plt.show(block=False)
        plt.close()

        user_input = input(f"Enter slice number (1-{slices.shape[0]}) or (Q)uit: ").strip().lower()

        if user_input == 'q':
            print("Exiting without saving.")
            exit(0)
        elif user_input.isdigit():
            slice_idx = int(user_input) - 1
            if 0 <= slice_idx < slices.shape[0]:
                return slice_idx
            else:
                print(f"Invalid slice number. Please enter a number between 1 and {slices.shape[0]}.")
        else:
            print("Invalid input. Please enter a valid slice number or 'Q' to quit.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Uso: python getpngfromdcm.py <input_dcm_file> <output_png_file>")
        sys.exit(1)

    input_dcm_file = sys.argv[1]
    output_png_file = sys.argv[2]

    selected_slice = select_slice(input_dcm_file)
    dcm_to_png(input_dcm_file, output_png_file, selected_slice)

    print(f"Slice {selected_slice + 1} saved as {output_png_file}")
