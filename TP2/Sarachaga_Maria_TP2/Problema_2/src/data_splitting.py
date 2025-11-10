import numpy as np
import pandas as pd

def split_val_train(dataset, target_column, test_size=0.2):
    np.random.seed(42)
    indices = np.random.permutation(dataset.index)
    val_size = int(len(dataset) * test_size)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    val_set = dataset.loc[val_indices]
    train_set = dataset.loc[train_indices]
    X_train = train_set.drop(columns=[target_column])
    y_train = train_set[target_column]
    X_val = val_set.drop(columns=[target_column])
    y_val = val_set[target_column]
    return X_train, X_val, y_train, y_val