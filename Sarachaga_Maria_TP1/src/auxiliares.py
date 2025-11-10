import numpy as np
from src.metrics import mse
import pandas as pd
import matplotlib.pyplot as plt

def kfold_indices(n, k=5, shuffle=True, random_state=None):
    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n)
    else:
        indices = np.arange(n)
    
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1
    current = 0
    folds = []

    for fold_size in fold_sizes:
        val_idx = indices[current:current + fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        folds.append((train_idx, val_idx))
        current += fold_size

    return folds

def cross_validation_ecm(modelo_cls, X, y, lambdas, k=5, modo="ridge"):
    n = len(X)
    folds = kfold_indices(n, k=k, shuffle=True, random_state=42)
    errores = []

    for lam in lambdas:
        mse_folds = []
        for train_idx, val_idx in folds:
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

            mu_X, sigma_X = X_train.mean(), X_train.std()
            X_train_scaled = (X_train - mu_X) / sigma_X
            X_val_scaled = (X_val - mu_X) / sigma_X

            if modo == "lasso":
                mu_y, sigma_y = y_train.mean(), y_train.std()
                y_train_scaled = (y_train - mu_y) / sigma_y
            else:
                y_train_scaled = y_train

            modelo = modelo_cls(X_train_scaled, y_train_scaled)
            modelo.array()
            if modo == "ridge":
                modelo.pseudoinversa(lambda2=lam)
            else: 
                modelo.descenso_gradiente(alpha=0.05, iteraciones=2000, lambda1=lam)

            y_pred = np.dot(
                np.hstack([np.ones((X_val_scaled.shape[0], 1)), X_val_scaled]),
                modelo.coef
            )
            
            if modo == "lasso":
                y_pred = y_pred * sigma_y + mu_y

            mse_folds.append(mse(y_val, y_pred))
        errores.append(np.mean(mse_folds))
    return np.array(errores)

def generar_features(X_base, features, max_cols=50):
    X = pd.DataFrame(index=X_base.index)
    for i in range(len(features)):
        for j in range(i, len(features)):
            col_name = f"{features[i]}*{features[j]}"
            X[col_name] = X_base[features[i]] * X_base[features[j]]
    for f in features:
        X[f"{f}^3"] = X_base[f] ** 3

    return X.iloc[:, :max_cols].copy()


def entrenar_y_evaluar(modelo, X_train, y_train, X_val, y_val):
    theta_gd, _ = modelo.descenso_gradiente(alpha=0.001, iteraciones=10000)
    modelo.coef = theta_gd
    mse_train_gd = mse(y_train, np.dot(np.hstack([np.ones((X_train.shape[0], 1)), X_train]), modelo.coef))
    mse_val_gd = mse(y_val, np.dot(np.hstack([np.ones((X_val.shape[0], 1)), X_val]), modelo.coef))  
    modelo.pseudoinversa()
    mse_train_pi = mse(y_train, np.dot(np.hstack([np.ones((X_train.shape[0], 1)), X_train]), modelo.coef))
    mse_val_pi = mse(y_val, np.dot(np.hstack([np.ones((X_val.shape[0], 1)), X_val]), modelo.coef))

    return mse_train_gd, mse_val_gd, mse_train_pi, mse_val_pi

def learning_curve(modelo_class, X_train, y_train, X_val, y_val, step=50, metodo="pseudoinversa", **kwargs):
    train_errors, val_errors, sizes = [], [], []
    
    for m in range(step, X_train.shape[0]+1, step):
        X_sub = X_train.iloc[:m, :] if isinstance(X_train, pd.DataFrame) else X_train[:m, :]
        y_sub = y_train[:m]

        modelo = modelo_class(X_sub, y_sub)
        modelo.array()

        if metodo == "pseudoinversa":
            modelo.pseudoinversa()
        elif metodo == "gradiente":
            theta_gd, _ = modelo.descenso_gradiente(
                alpha=kwargs.get("alpha", 0.01),
                iteraciones=kwargs.get("iteraciones", 1000)
            )
            modelo.coef = theta_gd
        y_train_pred = modelo.predecir(X_sub)
        y_val_pred   = modelo.predecir(X_val)
        mse_train = mse(y_sub, y_train_pred)
        mse_val   = mse(y_val, y_val_pred)

        train_errors.append(mse_train)
        val_errors.append(mse_val)
        sizes.append(m)

    return sizes, train_errors, val_errors


def plot_learning_curve(modelo_class, X_train, y_train, X_val, y_val, titulo, metodo="pseudoinversa", **kwargs):
    sizes, train_err, val_err = learning_curve(modelo_class, X_train, y_train, X_val, y_val, metodo=metodo, **kwargs)

    plt.figure(figsize=(8,6))
    plt.plot(sizes, train_err, label="ECM (train)", marker="o")
    plt.plot(sizes, val_err, label="ECM (validation)", marker="o")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("ECM")
    plt.title(f"Learning Curve - {titulo} ({metodo})")
    plt.legend()
    plt.grid(True)
    plt.show()
