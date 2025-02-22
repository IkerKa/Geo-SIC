from datasets.shapedsloader import ShapesDataLoaderHandler

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shape dataset loader")
    parser.add_argument("--shape", type=str, default="circle", help="Name of the shape to load")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")

    args = parser.parse_args()
    
    shapes_folder = "datasets/shapes"
    object_name = args.shape
    
    handler = ShapesDataLoaderHandler(shapes_folder, object_name, batch_size=8, resize=128, samples=args.samples)
    
    # Show an example
    handler.show_example()
    
    # Get all images in a tensor
    all_images = handler.get_all_images_tensor()
    print(f"Total images loaded for {object_name}: {all_images.shape[0]}")