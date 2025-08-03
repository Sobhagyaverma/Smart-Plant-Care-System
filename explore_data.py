import os

# Set the path to your training directory
train_dir = 'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'

# Loop through each folder (each class)
for folder_name in os.listdir(train_dir):
    # Ignore hidden files like .DS_Store
    if not folder_name.startswith('.'):
        folder_path = os.path.join(train_dir, folder_name)
        num_images = len(os.listdir(folder_path))
        print(f"Class: {folder_name}, Images: {num_images}")