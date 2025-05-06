from src.process_data.load_data import prepare_folds
from src.models.utils import run_folds
from config.config import config


device = config.device

prepare_folds()

for fold in range(config.no_of_folds):

    config.fold_no = fold
    print(f"\nFold: {fold}")
    run_folds(fold)





