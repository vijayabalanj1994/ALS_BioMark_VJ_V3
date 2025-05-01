import os
import shutil
from PIL import Image

import pandas as pd
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from config.config import config

def get_train_test(fold):

    print(f"\tsplitting into train and test for flod: {fold}")
    # Loading the image key's with fold file
    keys_path = os.path.join(config.dataset_dir_path, "image_keys_with_fold")
    df = pd.read_csv(keys_path)

    # Label mapping
    label_map = {
        "Control": 0,
        "Concordant": 1,
        "Discordant": 2
    }

    # Splitting based on fold
    train_df = df[df["fold"] != fold]
    val_df = df[df["fold"] == fold]

    # Creating image paths and labels
    def make_paths_and_labels(sub_df):
        image_paths = [os.path.join(config.dataset_dir_path,"images",f"{int(row['Image No'])}.tif") for _,row in sub_df.iterrows()]
        labels = [label_map[row["Category"]] for _,row in sub_df.iterrows()]
        case_ids = [row["Case ID"] for _,row in sub_df.iterrows()]
        return image_paths, labels, case_ids

    train_image_paths, train_labels, train_case_ids = make_paths_and_labels(train_df)
    val_image_paths, val_labels, val_case_ids = make_paths_and_labels(val_df)

    if any(item in set(val_case_ids) for item in set(train_case_ids)):
        print("\tthere is overlap between training and testing data.")
    else:
        print("\tno overlap between training and testing data.")

    return train_image_paths, val_image_paths, train_labels, val_labels

def compute_rgb_mean_std(image_paths):

    print("\tcalculating the mean and std of original training data to normalize the images")

    # transform fun to convert images to tensors
    to_tensor = transforms.ToTensor()

    # accumulators
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    num_pixels = 0

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = to_tensor(img)

        channel_sum += img_tensor.sum(dim=(1,2))
        channel_squared_sum += (img_tensor**2).sum(dim=(1,2))
        num_pixels += img_tensor.shape[1]* img_tensor.shape[2]

    mean = channel_sum / num_pixels
    std = (channel_squared_sum / num_pixels - mean**2).sqrt()

    return mean.tolist(), std.tolist()

def transform_function(mean, std):
    return transforms.Compose([
        transforms.Resize((400,400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])

class ALSDataset(Dataset):

    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

def save_augmented_image(folder, base_name, suffix, image, train_image_paths, train_labels, label):
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, f"{base_name}_{suffix}.tif")
    cv2.imwrite(out_path, image)
    train_image_paths.append(out_path)
    train_labels.append(label)

def augment_images(img_paths, labels):
    data_folder = os.path.join(config.dataset_dir_path, 'augmented_training_data')
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
        print(f"\tDeleted folder: {data_folder}")
    else:
        print(f"\tFolder does not exist: {data_folder}")

    print("\taugmenting training data")

    train_image_paths = []
    train_labels = []

    for img_path, label in zip(img_paths, labels):
        img_color = cv2.imread(img_path)
        if img_color is None:
            print(f"Warning: Skipping invalid image {img_path}")
            continue

        h, w = img_color.shape[:2]
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # 1. Original
        save_augmented_image(os.path.join(data_folder, "original"), base_name, "orig", img_color, train_image_paths, train_labels, label)

        # 2. Horizontal Flip
        hflip = cv2.flip(img_color, 1)
        save_augmented_image(os.path.join(data_folder, "hflip"), base_name, "hflip", hflip, train_image_paths, train_labels, label)

        # 3. Vertical Flip
        vflip = cv2.flip(img_color, 0)
        save_augmented_image(os.path.join(data_folder, "vflip"), base_name, "vflip", vflip, train_image_paths, train_labels, label)

        # 4. Rotation ±15°
        for angle in [-15, 15]:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(img_color, M, (w, h))
            save_augmented_image(os.path.join(data_folder, "rotated"), base_name, f"rot{angle}", rotated, train_image_paths, train_labels, label)

        # 5. Brightness/Contrast
        bright = cv2.convertScaleAbs(img_color, alpha=1.1, beta=10)
        dark = cv2.convertScaleAbs(img_color, alpha=0.9, beta=-10)
        save_augmented_image(os.path.join(data_folder, "brightness"), base_name, "bright", bright, train_image_paths, train_labels, label)
        save_augmented_image(os.path.join(data_folder, "brightness"), base_name, "dark", dark, train_image_paths, train_labels, label)

        # 6. Blur
        blurred = cv2.GaussianBlur(img_color, (5, 5), 0)
        save_augmented_image(os.path.join(data_folder, "blur"), base_name, "blur", blurred, train_image_paths, train_labels, label)

        # 7. Gaussian Noise
        noise = np.random.normal(0, 10, img_color.shape).astype(np.uint8)
        noisy = cv2.add(img_color, noise)
        save_augmented_image(os.path.join(data_folder, "noise"), base_name, "noise", noisy, train_image_paths, train_labels, label)

        # 8. Scaling (0.9x)
        scaled = cv2.resize(img_color, (int(w * 0.9), int(h * 0.9)))
        scaled_full = cv2.copyMakeBorder(scaled, 0, h - scaled.shape[0], 0, w - scaled.shape[1], cv2.BORDER_CONSTANT, value=0)
        save_augmented_image(os.path.join(data_folder, "scaled"), base_name, "scaled", scaled_full, train_image_paths, train_labels, label)

        # 9. Cropping
        cropped = img_color[10:-10, 10:-10]
        cropped_resized = cv2.resize(cropped, (w, h))
        save_augmented_image(os.path.join(data_folder, "cropped"), base_name, "crop", cropped_resized, train_image_paths, train_labels, label)

    return train_image_paths, train_labels