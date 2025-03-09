import nibabel as nib
import numpy as np
import vedo

# Cargar imagen NIfTI
nii_img = nib.load("./nirep/nifti/na15.nii.gz")
data = nii_img.get_fdata()
data = np.array(data, dtype=np.float32)

# Parámetros del volumen
num_slices = data.shape[2] 
current_slice = num_slices // 2

# Crear ventana de visualización
plotter = vedo.Plotter(axes=1, bg="white")

def show_slice(slice_idx):
    """Función para actualizar y mostrar un slice específico"""
    if 0 <= slice_idx < num_slices:
        img_slice = data[:, :, slice_idx] 
        plotter.clear() 
        
        image = vedo.Image(img_slice) 

        plotter.show(image, text, interactive=False)  

show_slice(current_slice)

while True:
    user_input = input(f"Ingrese un número de slice (0-{num_slices-1}) o 'q' para salir: ")
    if user_input.lower() == 'q':
        break
    try:
        new_slice = int(user_input)
        show_slice(new_slice)
    except ValueError:
        print("Entrada inválida. Por favor ingrese un número.")
