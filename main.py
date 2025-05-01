from src.process_data.load_data import prepare_folds, get_dataloaders

prepare_folds()

train_loader, val_loader =  get_dataloaders(fold=0)



