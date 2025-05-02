


import config  # Este es el archivo que contiene todas las importaciones
from evaluate_validation_test import load_all_data, convert_to_tensor, exhaustive_train_model_with_validation, compute_segmentation
from Run_Atlas_trainer import initialize_network_optimizer2D, read_yaml

import torch
from networks import UnetDense

def custom_training(net, optimizer, num_epochs, train_dataset, train_segmentations, device, flag = 'SVF'):
    loss_total = 0
    phi_inv = None
    ssim_per_epoch, loss_per_epoch, val_dice_scores = [], [], []


    batch_size = 4
    acc_loss = 0
    
    plot_phis = []

    tqdm_epochs = config.tqdm.tqdm(total=num_epochs, desc="Training Progress", leave=False)
    for epoch in range(num_epochs):
        tqdm_epochs.update(1)

        net.train()
        
        # Per each epoch, train with all possible pairs of images? can be that done? i think it would improve
        tqdm_epochs.set_description(f'Epoch {epoch + 1}/{num_epochs}')

        pair_loss = []
        pair_ssim = []
        pair_dice = []

        pairs = [(i, j) for i in range(len(train_dataset)) for j in range(i + 1, len(train_dataset))]
        pairs = config.random.sample(pairs, k=min(len(pairs), 100))  # Limit to 100 pairs for training

        for pair_idx, (i,j) in enumerate(pairs):

            I1 = train_dataset[i]
            I2 = train_dataset[j]

            I1_seg = train_segmentations[i]
            I2_seg = train_segmentations[j]

            b, c, w, h = I1.shape

            I1, I2 = I1.to(device).float(), I2.to(device).float()
            y_src, momentum,  _, new_locs = net(I1, I2, registration=True, shooting="SVF", return_phi=True)
            _,_, dice_score = compute_segmentation(I1_seg, new_locs[0,...], I2_seg, device)

            # momentum = momentum.permute(0, 3, 1, 2)  # Permute to [batch, 2, height, width]
            # momentum_neg = momentum_neg.permute(0, 3, 1, 2)  # Permute to [batch, 2, height, width]


            if flag == 'SVF':
                # Dist = (NCC().loss(I2, y_src)) # + NCC().loss(y_tgt, I1))
                Dist = config.NCC().loss(I2, y_src)
                # Dist = config.MSE().loss(I2, y_src)
                Reg = config.Grad(penalty='l2')
                Reg_loss = Reg.loss2D(momentum)
                # Dice_loss = DiceLoss(device=device)

                loss_total = (1 * Dist + 0.01 * Reg_loss)
                acc_loss += loss_total

            if (pair_idx + 1) % batch_size == 0 or pair_idx == len(pairs) - 1:
                avg_loss = acc_loss / batch_size
                avg_loss.backward()
                optimizer.step()      # Update model parameters
                optimizer.zero_grad() # Reset gradients
                acc_loss = 0

            # Update the loss and ssim values
            pair_loss.append(loss_total.item())
            pair_ssim.append(config.ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(),
                                    data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min()))
            pair_dice.append(dice_score)

            with torch.no_grad():
                phi_inv = new_locs[0,...]

            
        
        mean_loss = config.np.mean(pair_loss)
        mean_ssim = config.np.mean(pair_ssim)
        mean_dice = config.np.mean(pair_dice)
        loss_per_epoch.append(mean_loss)
        ssim_per_epoch.append(mean_ssim)
        val_dice_scores.append(mean_dice)


        tqdm_epochs.set_postfix(loss=mean_loss, ssim=mean_ssim, dice=mean_dice)

def custom_test(net, moving_image, fixed_image, moving_segmentation, fixed_segmentation, device, flag='SVF'):
    # with the trained model, test the images (phi inverted)

    # can be also use the net?
  

    with torch.no_grad():
        net.eval()
        
        I1 = moving_image
        I2 = fixed_image
        I1_seg = moving_segmentation
        I2_seg = fixed_segmentation

        I1 = I1.to(device).float()
        I2 = I2.to(device).float()

        y_src, _, _, new_locs = net(I1, I2, registration=True, shooting=flag, return_phi=True)
        phi_inv = new_locs[0,...]
        _, _, dice_score = compute_segmentation(I1_seg, phi_inv, I2_seg, device)


        #obtain the metrics and save them
        ssim_score = config.ssim(y_src.squeeze().cpu().detach().numpy(), I2.squeeze().cpu().detach().numpy(), data_range=I2.squeeze().cpu().detach().numpy().max() - I2.squeeze().cpu().detach().numpy().min())
        rmse_score = config.np.sqrt(config.np.mean((I2.squeeze().cpu().detach().numpy() - y_src.squeeze().cpu().detach().numpy()) ** 2))
        mean_dice_score = config.np.mean(dice_score)


        # print(f'Test - SSIM: {ssim_score:.4f}, RMSE: {rmse_score:.4f}, Dice: {mean_dice_score:.4f}')

        # save_metrics('output', I2, y_src, ssim_score, rmse_score, mean_dice_score)

        #print the average
        print(f'Test - Results - SSIM: {ssim_score:.4f}, RMSE: {rmse_score:.4f}, Dice: {mean_dice_score:.4f}')
        
        # plot_results(test_images, test_segs, phiinvs, y_srcs, device, _save = True)
        return ssim_score, rmse_score, mean_dice_score, phi_inv
    
def save_results(slice_idx, slice_folder, metrics, phi_inv):
    metrics = {key: float(value) for key, value in metrics.items()}

    # Save metrics in a JSON file
    metrics_path = config.os.path.join(slice_folder, 'metrics.json')
    if config.os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_data = config.json.load(f)
    else:
        metrics_data = {}

    metrics_data[slice_idx] = metrics
    with open(metrics_path, 'w') as f:
        config.json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved for slice {slice_idx}")

    # Save phi_inv in a folder (NPY format)
    phi_inv_path = config.os.path.join(slice_folder, 'Phis', f'phi_inv_slice_{slice_idx}.npy')
    if not config.os.path.exists(config.os.path.dirname(phi_inv_path)):
        config.os.makedirs(config.os.path.dirname(phi_inv_path))
        print(f"Created folder: {config.os.path.dirname(phi_inv_path)}")
    config.np.save(phi_inv_path, phi_inv.cpu().numpy())
    print(f"Phi_inv saved for slice {slice_idx}")

def main():


    # We are going to do a slice evaluation. So we will launch the code for every slice we want to evaluate.
    # Each slice will have 150/300 epochs of training
    # Dataset length is 16, we will keep the first 2 images to the test and the rest for training.

    # The results will be stored in a folder "Results"

    # Structure of the folder:
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


    output_folder = 'Evaluation/Results'

    slices = ['Sagital', 'Axial', 'Coronal']
    
    slice_selection = input("Select the slice to evaluate (Sagital, Axial, Coronal): ")
    if slice_selection not in slices:
        print(f"Invalid slice selection. Please choose from {slices}.")
        return
    

    # We will create the folder for the selected slice
    slice_folder = config.os.path.join(output_folder, slice_selection)
    if not config.os.path.exists(slice_folder):
        config.os.makedirs(slice_folder)
        print(f"Created folder: {slice_folder}")
    else:
        print(f"Folder already exists: {slice_folder}.")
    
    if slice_selection == 'Sagital': view = 1
    if slice_selection == 'Axial': view = 3
    if slice_selection == 'Coronal': view = 2
    
    # We have 16 volumes, and we have to choose N slices from each volume.
    # The volumes are (256 x 300 x 256) so depending on the view, we have to limit to 256 or 300 slices.
    # For sagital views: [slice_idx, :, :], for coronal views: [:, slice_idx, :], for axial views: [:, :, slice_idx]

    # But we will take the middle range of slices, from idx 100 to 200 for example and with a M jump step

    n_epochs = 300
    jump_step = 5
    start_slice_idx = 90
    end_slice_idx = 210

    slice_indices = list(range(start_slice_idx, end_slice_idx, jump_step))
    print(f"Slice indices to evaluate: {slice_indices}")

    # Take the current data for every slice

    evaluation_data = [] # This will be a vector of vectors 16xN, where N is the number of slices we want to evaluate.
    # each element contains the image and the segmentation
    for i, slice_idx in enumerate(slice_indices):
        
        data_comb = load_all_data(slice_index=slice_idx, view=view)
        evaluation_data.append(data_comb)
        print(f"Loaded data for slice {slice_idx}", end='\r')

    print("\nData loaded for all slices")
    print(f"Evaluation data structure: {type(evaluation_data)} with {len(evaluation_data)} elements")
    print("First element structure:", type(evaluation_data[0]), "with length:", len(evaluation_data[0]) if evaluation_data else "N/A")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    para = read_yaml('parameters.yml')

    print(f"Using device: {device}")
        
    num_training_images = len(evaluation_data) - 2 # We will keep the first two images for testing.
    print(f"Number of training images: {num_training_images}")
    num_validation_images = 2 # We will keep the first two images for testing.
    print(f"Number of validation images: {num_validation_images}")

    for i, slice_idx in enumerate(slice_indices):
        print(f"Evaluating slice {slice_idx}")

        # We will take the first two images for testing and the rest for training.
        net, _, optimizer = initialize_network_optimizer2D(128, 128, para, device)

        
        # The images are in the format [moving_image, fixed_image]--> TEST
        moving_image = convert_to_tensor(evaluation_data[i][0][0], device)
        fixed_image = convert_to_tensor(evaluation_data[i][1][0], device)
        # ---
        moving_segmentation = convert_to_tensor(evaluation_data[i][0][1], device)
        fixed_segmentation = convert_to_tensor(evaluation_data[i][1][1], device)

        # Training data from the 2nd to the last image
        training_data = evaluation_data[i][2:]
        training_images = [convert_to_tensor(data[0], device) for data in training_data]
        training_segmentations = [convert_to_tensor(data[1], device) for data in training_data]

        # plot the images and segmentations
        # for j, data in enumerate(training_data):
        #     config.plt.subplot(1, 2, 1)
        #     config.plt.imshow(data[0][0], cmap='gray')
        #     config.plt.title(f"Training Image {j}")
        #     config.plt.subplot(1, 2, 2)
        #     config.plt.imshow(data[1][0], cmap='gray')
        #     config.plt.title(f"Training Segmentation {j}")
        #     config.plt.show()

        ### Training the model ###
        custom_training(net, optimizer, n_epochs, training_images, training_segmentations, device, flag='SVF')
        ### Testing the model ###
        SSIM, RMSE, DICE, phi_inv = custom_test(net, moving_image, fixed_image, moving_segmentation, fixed_segmentation, device, flag='SVF')

        # Save the results in the corresponding folder
        # 1st the metrics on the JSON for the current slice_idx
        metrics = {
            "DICE": DICE,
            "RMSE": RMSE,
            "SSIM": SSIM,
        }
        save_results(slice_idx, slice_folder, metrics, phi_inv)





        
        



if __name__ == "__main__":
    main()