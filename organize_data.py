import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def create_directory_structure():
    """Create the necessary directory structure"""
    # Create main directories
    directories = [
        'data/train',
        'data/valid',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def organize_data(csv_path, images_dir, train_ratio=0.8):
    """
    Organize images based on CSV classifications
    
    Parameters:
    csv_path: Path to the _classes.csv file
    images_dir: Directory containing the images
    train_ratio: Ratio of images to use for training (default: 0.8)
    """
    # Create directory structure
    create_directory_structure()
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Split into train and validation sets
    train_df, valid_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=42
    )
    
    # Function to copy images
    def copy_images(dataframe, subset_type):
        for _, row in dataframe.iterrows():
            filename = row['filename']
            src = os.path.join(images_dir, filename)
            dst = os.path.join(f'data/{subset_type}', filename)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"Copied {filename} to {subset_type} set")
            else:
                print(f"Warning: Could not find {filename}")
    
    # Copy images to train and valid directories
    copy_images(train_df, 'train')
    copy_images(valid_df, 'valid')
    
    # Save split CSVs for reference
    train_df.to_csv('data/train/_classes.csv', index=False)
    valid_df.to_csv('data/valid/_classes.csv', index=False)

if __name__ == "__main__":
    csv_path = "_classes.csv"  # Path to your CSV file
    images_dir = "images"      # Directory containing your images
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found!")
    elif not os.path.exists(images_dir):
        print(f"Error: Images directory '{images_dir}' not found!")
    else:
        organize_data(csv_path, images_dir)
        print("\nData organization complete!")
        print("\nCreated structure:")
        print("data/")
        print("├── train/")
        print("│   ├── _classes.csv")
        print("│   └── [images]")
        print("└── valid/")
        print("    ├── _classes.csv")
        print("    └── [images]") 