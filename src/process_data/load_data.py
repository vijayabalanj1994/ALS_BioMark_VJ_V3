import os

from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
from torch.utils.data import DataLoader

from config.config import config
from src.process_data.utils import get_train_test, compute_rgb_mean_std, transform_function, ALSDataset, augment_images

def prepare_folds():

    # Loading the image key's excel file
    keys_path = os.path.join(config.dataset_dir_path, "image_keys.xlsx")
    df = pd.read_excel(keys_path, sheet_name="Sheet1", header=1)
    df = df[["Image No", "Case ID", "Category"]].copy()

    # Initializing StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=config.no_of_folds, shuffle=True, random_state=42)

    # Creating a new colum in the DataFrame to store fold assignments
    df["fold"] = -1

    # Applying the split and assign folds
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X=df, y=df["Category"], groups=df["Case ID"])):
        df.loc[val_idx, "fold"] = fold

    # Saving the DataFrame
    keys_path = os.path.join(config.dataset_dir_path,"image_keys_with_fold")
    df.to_csv(keys_path, index=False)


def get_dataloaders(fold):

    print("\tinstantiating the train and val dataloaders")
    # getting the train and test dats for the fold
    train_image_paths, val_image_paths, train_labels, val_labels = get_train_test(fold)

    # function to normalize the images before passing into model
    mean, std = compute_rgb_mean_std(train_image_paths)
    transform =  transform_function(mean,std)

    # augmenting the train images
    train_image_paths, train_labels = augment_images(train_image_paths, train_labels)

    # instantiating the torch datasets
    train_dataset = ALSDataset(train_image_paths, train_labels, transform)
    val_dataset = ALSDataset(val_image_paths, val_labels, transform)

    # instantiating the torch dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader
