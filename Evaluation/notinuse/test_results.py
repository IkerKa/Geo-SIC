




import config  # Este es el archivo que contiene todas las importaciones
from Evaluation.evaluate_validation_test import load_all_data, convert_to_tensor, compute_segmentation


def plot_results(moving_image, fixed_image, moving_segmentation, fixed_segmentation, phi_inv, device):
    
    #for every possible pair

    I1 = moving_image
    I2 = fixed_image
    I1_seg = moving_segmentation
    I2_seg = fixed_segmentation 

    phi_inv = phi_inv.to(device)
    warped_img, fixed_img, dice_score = compute_segmentation(I1_seg, phi_inv, I2_seg, device)

    phi_inv = phi_inv.unsqueeze(0)  # Add batch dimension to phi_inv
    y_src = config.F.grid_sample(I1, phi_inv, mode='bilinear', padding_mode='border')
    y_src = y_src.squeeze(0)  # Remove batch dimension


    fig, axes = config.plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(I1.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[0].set_title('Source Image (I1)')
    axes[1].imshow(I2.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[1].set_title('Target Image (I2)')
    axes[2].imshow(y_src.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[2].set_title('Registered Image (y_src)')
    error = I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()
    axes[3].imshow(error, cmap='gray')
    axes[3].set_title('Error (I2 - y_src)')

    # Plotting the deformation grid for phi_inv
    ax = axes[4]
    interval = 2
    phi_inv = phi_inv.squeeze(0).cpu().detach().numpy()
    phi_inv[:, :, 1] = -phi_inv[:, :, 1]  # Flip along the Y-axis
    for row in range(0, phi_inv.shape[0], interval):
        ax.plot(phi_inv[row, :, 0],
                phi_inv[row, :, 1],
                'm')

    for col in range(0, phi_inv.shape[1], interval):
        ax.plot(phi_inv[:, col, 0],
                phi_inv[:, col, 1],
                'm')

    ax.set_title("Deformation Grid")

    #plot the same for the shapes
    fig, axes = config.plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(I1_seg.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[0].set_title('Source Segmentation (I1_seg)')
    axes[1].imshow(I2_seg.squeeze().cpu().detach().numpy(), cmap='gray')
    axes[1].set_title('Target Segmentation (I2_seg)')
    axes[2].imshow(warped_img, cmap='gray')
    axes[2].set_title('Warped Segmentation (I1_seg -> I2_seg)')

    # Calculate and plot the shape error
    shape_error = I2_seg.squeeze().cpu().detach().numpy() - warped_img
    axes[3].imshow(shape_error, cmap='gray')
    axes[3].set_title('Shape Error (I2_seg - Warped)')

    config.plt.tight_layout()
    config.plt.show()

    #save the I1, I2, and y_src images as nifti files
    config.nib.save(config.nib.Nifti1Image(I1.squeeze().cpu().detach().numpy(), config.np.eye(4)), 'I1.nii.gz')
    config.nib.save(config.nib.Nifti1Image(I2.squeeze().cpu().detach().numpy(), config.np.eye(4)), 'I2.nii.gz')
    config.nib.save(config.nib.Nifti1Image(y_src.squeeze().cpu().detach().numpy(), config.np.eye(4)), 'I1toI2.nii.gz')
    #same for the segmentations
    config.nib.save(config.nib.Nifti1Image(I1_seg.squeeze().cpu().detach().numpy(), config.np.eye(4)), 'I1_seg.nii.gz')
    config.nib.save(config.nib.Nifti1Image(I2_seg.squeeze().cpu().detach().numpy(), config.np.eye(4)), 'I2_seg.nii.gz')
    config.nib.save(config.nib.Nifti1Image(warped_img, config.np.eye(4)), 'I1_seg_to_I2_seg.nii.gz')




# Results
    #     -/Sagital
    #         -/metrics.json
    #         -/Phis
    #            -/phi_inv_slice_0.npy
    #            -/phi_inv_slice_1.npy
    #     -/Axial
    #         -/metrics.json
    #         -/Phis
    #            -/phi_inv_slice_0.npy
    #            -/phi_inv_slice_1.npy
    #     -/Coronal
    #         -/metrics.json
    #...

    # the metrics will have the following structure:
    # metrics = {
    #     "slice_idx": {
    #         "average_dice": 0.5,
    #         "average_rmse": 0.5,
    #         "average_ssim": 0.5,
    #         }


# We will do two different evaluations:
# 1. Plot the metrics for each slice
# 2. Use the Phi values over the corresponding slice to plot the warped image and the target image


def main():

    device = config.torch.device("cuda" if config.torch.cuda.is_available() else "cpu")

    slices = ['Sagital', 'Axial', 'Coronal']
    
    slice_selection = input("Select the slice to evaluate (Sagital, Axial, Coronal): ")
    if slice_selection not in slices:
        print(f"Invalid slice selection. Please choose from {slices}.")
        return
    
    if slice_selection == 'Sagital': view = 1
    elif slice_selection == 'Axial': view = 3
    elif slice_selection == 'Coronal': view = 2

    results_path = 'Evaluation/Results'
    slice_folder = config.os.path.join(results_path, slice_selection)

    metrics_path = config.os.path.join(slice_folder, 'metrics.json')
    
    # Load the metrics from the json file
    with open(metrics_path, 'r') as f:
        metrics = config.json.load(f)

    # Extract slice indices and corresponding metrics
    slice_indices = list(metrics.keys())
    dice_scores = [metrics[idx]["DICE"] for idx in slice_indices]
    rmse_scores = [metrics[idx]["RMSE"] for idx in slice_indices]
    ssim_scores = [metrics[idx]["SSIM"] for idx in slice_indices]

    # Plot the metrics
    config.plt.figure(figsize=(10, 6))
    config.plt.plot(slice_indices, dice_scores, label="DICE", marker='o')
    config.plt.plot(slice_indices, rmse_scores, label="RMSE", marker='o')
    config.plt.plot(slice_indices, ssim_scores, label="SSIM", marker='o')
    config.plt.xlabel("Slice Index")
    config.plt.ylabel("Metric Value")
    config.plt.title(f"Metrics for {slice_selection} Slices")
    config.plt.legend()
    config.plt.grid(True)
    config.plt.show()

    # Now we are going to select an slice to take its particular phi value
    slice_idx = int(input(f"Select the slice index to evaluate: "))
    if str(slice_idx) not in slice_indices:
        print(f"Invalid slice index. Please choose from the available indices: {', '.join(slice_indices)}.")
        return
    
    phi_path = config.os.path.join(slice_folder, 'Phis', f'phi_inv_slice_{slice_idx}.npy')
    phi = config.np.load(phi_path)
    phi = config.torch.tensor(phi, dtype=config.torch.float32)

    # Load the corresponding image and target image
    all_data = load_all_data(slice_index=slice_idx, view=view)
    validation_images = all_data[:2]

    moving_image = convert_to_tensor(validation_images[0][0], device)
    fixed_image = convert_to_tensor(validation_images[1][0], device)
    # ---
    moving_segmentation = convert_to_tensor(validation_images[0][1], device)
    fixed_segmentation = convert_to_tensor(validation_images[1][1], device)

    plot_results(moving_image, fixed_image, moving_segmentation, fixed_segmentation, phi, device)

   

if __name__ == "__main__":
    main()