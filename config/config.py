import torch

class Config():

    dataset_dir_path = r"C:\Users\vijay\Neuro_BioMark\ALS_BioMark_VJ_V3\dataset"
    saved_models_dir_path = r"C:\Users\vijay\Neuro_BioMark\ALS_BioMark_VJ_V3\saved_models"
    logs_dir_path = r"C:\Users\vijay\Neuro_BioMark\ALS_BioMark_VJ_V3\logs"

    no_of_folds = 5
    fold_no = -1

    lr = 1e-4
    batch_size = 32
    no_of_epoch = 10
    weight_decay = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing {device} device")

config = Config()