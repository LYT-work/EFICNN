import os
import random
from shutil import copy2

img_folder = "../datasets/Whu/A"
train_output_file = "../splits/Whu/train.txt"
val_output_file = "../splits/Whu/val.txt"
test_output_file = "../splits/Whu/test.txt"

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

img_files = os.listdir(img_folder)

file_paths = {}
for img_file in img_files:
    img_prefix = img_file.split("_")[0]
    file_paths[img_prefix] = {"img_file": img_file}

num_files = len(file_paths)
num_train = int(num_files * train_ratio)
num_val = int(num_files * val_ratio)
num_test = num_files - num_train - num_val

train_keys = random.sample(list(file_paths.keys()), num_train)
remaining_keys = list(set(file_paths.keys()) - set(train_keys))
val_keys = random.sample(remaining_keys, k=num_val)
test_keys = list(set(remaining_keys) - set(val_keys))

def save_files(keys, folder, output_file):
    with open(output_file, "w") as file:
        for i, key in enumerate(keys):
            new_filename = f"{i + 1:05d}.png"
            original_file_path = os.path.join(img_folder, file_paths[key]["img_file"])
            new_file_path = os.path.join(folder, new_filename)
            copy2(original_file_path, new_file_path)
            file.write(f"{new_file_path}\n")

print("Image randomization and renaming are complete!")
