


import config  # Este es el archivo que contiene todas las importaciones
from evaluate_validation_test import load_all_data, convert_to_tensor, exhaustive_train_model_with_validation, compute_segmentation, net_test_model
from Run_Atlas_trainer import initialize_network_optimizer2D, read_yaml

import torch
from networks import UnetDense

# We will use only 2 images for train and test the model with different number of epochs


def main():

    epochs = 2000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    para = read_yaml('parameters.yml')

    all_dataset = load_all_data(slice_index = 128, view=3)

    # Load only two first images
    moving_image = convert_to_tensor(all_dataset[0][0], device=device)
    fixed_image = convert_to_tensor(all_dataset[1][0], device=device)
    moving_segmentation = convert_to_tensor(all_dataset[0][1], device=device)
    fixed_segmentation = convert_to_tensor(all_dataset[1][1], device=device)

    train_images = [moving_image, fixed_image]
    train_seg = [moving_segmentation, fixed_segmentation]

    val_images = [moving_image, fixed_image]
    val_seg = [moving_segmentation, fixed_segmentation]


    net, _, optimizer = initialize_network_optimizer2D(128, 128, para, device)

    shooting_flag = 'SVF'

    print(f"Before training...")
    net_test_model(net, val_images, val_seg, shooting_flag, device)

    input("Training with all possible pairs of images...")
    time_init = config.time.time()
    phi_inv, _, loss, ssim = exhaustive_train_model_with_validation(net, optimizer, epochs, train_images, train_seg, device, shooting_flag)
    time_end = config.time.time()
    print(f'Training time: {time_end - time_init} seconds')


    print(f"After training...")
    net_test_model(net, val_images, val_seg, shooting_flag, device)

    


if __name__ == "__main__":
    main()
