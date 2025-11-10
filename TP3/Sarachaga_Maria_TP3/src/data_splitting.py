import numpy as np
import pandas as pd

def split_val_train(X_data, y_data, test_size=0.2):
    TARGET_COL_NAME = 'Target'
    if y_data.ndim > 1:
        y_data = y_data.ravel() 
    df_features = pd.DataFrame(X_data)
    y_series = pd.Series(y_data, name=TARGET_COL_NAME)
    dataset = pd.concat([df_features, y_series], axis=1)
    np.random.seed(42)
    indices = np.random.permutation(dataset.index)
    val_size = int(len(dataset) * test_size)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    val_set = dataset.loc[val_indices]
    train_set = dataset.loc[train_indices]
    X_train = train_set.drop(columns=[TARGET_COL_NAME])
    y_train = train_set[TARGET_COL_NAME]
    X_val = val_set.drop(columns=[TARGET_COL_NAME])
    y_val = val_set[TARGET_COL_NAME]
    return X_train.values, X_val.values, y_train.values, y_val.values