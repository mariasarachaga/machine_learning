import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def split_and_plot_classes(data, target_column='Diagnosis', test_size=0.2):
    X_train, X_val, y_train, y_val = split_val_train(
        data, target_column=target_column, test_size=test_size
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    train_counts = y_train.value_counts(normalize=True)
    val_counts = y_val.value_counts(normalize=True)
    df_plot = pd.DataFrame({"Train": train_counts, "Validation": val_counts}).T
    df_plot.plot(kind='bar', figsize=(6,4))
    plt.title("Proporción de clases en Train y Validation")
    plt.ylabel("Proporción")
    plt.xticks(rotation=0)
    plt.ylim(0,1)
    plt.show()
    return X_train, X_val, y_train, y_val, train_df, val_df
