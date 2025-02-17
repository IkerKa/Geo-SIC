import argparse
from datasets.datasetloader3d import DataLoaderHandler
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test MHD2DDataset and DataLoaderHandler")
    args = parser.parse_args()

    # Initialize DataLoaderHandler
    data_loader_handler = DataLoaderHandler(
        mhd_folder='./datasets/dcm/',
    )

    # Show an example image
    data_loader_handler.show_example()

    # Save the DataLoader
    # data_loader_handler.save_dataloader(file_path='dataloader.pt')