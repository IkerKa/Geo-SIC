import argparse
from datasets.createDataset import DataLoaderHandler

def main(image_path, samples, batch_size):
    data_handler = DataLoaderHandler(image_path, samples=samples, batch_size=batch_size)
    for i in range(10):
        data_handler.show_example()
    # data_handler.save_dataloader()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ImageTransformDataset and DataLoaderHandler")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--samples", type=int, default=100, help="Number of distorted samples to generate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader")
    
    args = parser.parse_args()
    main(args.image_path, args.samples, args.batch_size)