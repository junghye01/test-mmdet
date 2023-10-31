import os
import shutil
from pathlib import Path

def copy_images(txt_file, src_dir, dest_dir):
    with open(txt_file, 'r', encoding='UTF-8') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('#'):
            continue
        elif line.endswith('.jpg\n') or line.endswith('.jpeg\n'):
            img_name = line.strip()
            img_name=img_name.replace('jpg','xml')
            src_path = find_image_path(src_dir, img_name)
            if src_path:
                dest_path = os.path.join(dest_dir, img_name)
                shutil.copy(src_path, dest_path)
                print(f'{src_path} -> {dest_path}')
            else:
                print(f'{img_name} not found in {src_dir}')

def find_image_path(src_dir, img_name):
    for root, dirs, files in os.walk(src_dir):
        if img_name in files:
            return os.path.join(root, img_name)
    return None

src_dir = 'data_completed/Label'
train_dest_dir = 'data_completed/origin/train/Label'
val_dest_dir = 'data_completed/origin/val/Label'

# Create destination directories if they don't exist
Path(train_dest_dir).mkdir(parents=True, exist_ok=True)
Path(val_dest_dir).mkdir(parents=True, exist_ok=True)

# Copy images based on the lists in train.txt and val.txt
copy_images('data_completed/annotations/train.txt', src_dir, train_dest_dir)
copy_images('data_completed/annotations/val.txt', src_dir, val_dest_dir)