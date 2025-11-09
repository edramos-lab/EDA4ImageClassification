import os
import sys

def generate_dataset_info(dataset_path):
    """
    Generate dataset information CSV file from a given dataset path.
    Handles both flat structure (dataset/class1, dataset/class2) and 
    nested structure (dataset/train/class1, dataset/test/class1).
    
    Args:
        dataset_path (str): Path to the dataset directory containing class subdirectories
    """
    # Get the absolute path of the dataset directory
    dataset_path = os.path.abspath(dataset_path)
    
    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return
    
    # Check if it's a directory
    if not os.path.isdir(dataset_path):
        print(f"Error: '{dataset_path}' is not a directory.")
        return
    
    # Create the output CSV file path in the dataset directory
    output_csv_path = os.path.join(dataset_path, 'dataset_info.csv')
    
    with open(output_csv_path, 'w') as f:
        f.write("path,class\n")
        
        # First, check if we have a nested structure (train/test subdirectories)
        subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        # Check if we have train/test subdirectories
        has_train_test = any(subdir.lower() in ['train', 'test', 'validation', 'val'] for subdir in subdirs)
        
        if has_train_test:
            print("Detected nested structure with train/test subdirectories")
            # Handle nested structure: dataset/train/class1, dataset/test/class1
            for subdir in sorted(subdirs):
                subdir_path = os.path.join(dataset_path, subdir)
                if os.path.isdir(subdir_path):
                    print(f"Processing subdirectory: {subdir}")
                    for class_name in sorted(os.listdir(subdir_path)):
                        class_dir = os.path.join(subdir_path, class_name)
                        if os.path.isdir(class_dir):
                            print(f"  Processing class: {class_name}")
                            for filename in sorted(os.listdir(class_dir)):
                                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                                    full_path = os.path.join(class_dir, filename)
                                    # Use absolute path for the image
                                    absolute_path = os.path.abspath(full_path)
                                    f.write(f"{absolute_path},{class_name}\n")
        else:
            print("Detected flat structure with direct class subdirectories")
            # Handle flat structure: dataset/class1, dataset/class2
            for class_name in sorted(os.listdir(dataset_path)):
                class_dir = os.path.join(dataset_path, class_name)
                if os.path.isdir(class_dir):
                    print(f"Processing class: {class_name}")
                    for filename in sorted(os.listdir(class_dir)):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                            full_path = os.path.join(class_dir, filename)
                            # Use absolute path for the image
                            absolute_path = os.path.abspath(full_path)
                            f.write(f"{absolute_path},{class_name}\n")
    
    print(f"Dataset information generated successfully from: {dataset_path}")
    print(f"Output saved to: {output_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_dataset_info.py <dataset-path>")
        print("Example: python generate_dataset_info.py /path/to/dataset")
        print("Supports both flat and nested structures:")
        print("  Flat: dataset/class1, dataset/class2")
        print("  Nested: dataset/train/class1, dataset/test/class1")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    generate_dataset_info(dataset_path) 