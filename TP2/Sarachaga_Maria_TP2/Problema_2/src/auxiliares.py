from src.metrics import calcular_metricas, matriz_confusion, accuracy
from src.models import logistic_regression, softmax, RandomForest
import matplotlib.pyplot as plt
import numpy as np


def scores_ovr(y_val_idx, y_pred_idx, clases_unicas):
    y_score = np.zeros((len(y_val_idx), len(clases_unicas)))
    for i, _ in enumerate(clases_unicas):
        y_score[:, i] = (y_pred_idx == i).astype(float)
    return y_score

def metricas(y_val_idx, y_pred_idx, y_score, clases_unicas):
    resultados = {}

    for i, c in enumerate(clases_unicas):
        y_true_bin = (y_val_idx == i).astype(int)
        y_pred_bin = (y_pred_idx == i).astype(int)
        y_score_bin = y_score[:, i]

        metrics_i = calcular_metricas(y_true_bin, y_pred_bin, y_score_bin)
        resultados[c] = metrics_i

    return resultados


def graficar_curvas(y_val_idx, y_score, clases_unicas, title_suffix=""):
    plt.figure(figsize=(12,5))

    # ==== ROC ====
    plt.subplot(1,2,1)
    for i, c in enumerate(clases_unicas):
        y_true_bin = (y_val_idx == i).astype(int)
        y_score_bin = y_score[:, i]

        sorted_idx = np.argsort(y_score_bin)[::-1]
        y_sorted = y_true_bin[sorted_idx]

        P = np.sum(y_sorted)
        N = len(y_sorted) - P
        tp = fp = 0
        tpr_list, fpr_list = [], []

        for val in y_sorted:
            if val == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp/P if P>0 else 0)
            fpr_list.append(fp/N if N>0 else 0)

        plt.plot(fpr_list, tpr_list, label=f"Clase {c}")

    plt.title(f"ROC Curve (OvR) {title_suffix}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(True)

    # ==== PR ====
    plt.subplot(1,2,2)
    for i, c in enumerate(clases_unicas):
        y_true_bin = (y_val_idx == i).astype(int)
        y_score_bin = y_score[:, i]

        sorted_idx = np.argsort(y_score_bin)[::-1]
        y_sorted = y_true_bin[sorted_idx]

        P = np.sum(y_sorted)
        tp = fp = 0
        precision_list, recall_list = [], []

        for val in y_sorted:
            if val == 1:
                tp += 1
            else:
                fp += 1
            precision_list.append(tp/(tp+fp) if (tp+fp)>0 else 0)
            recall_list.append(tp/P if P>0 else 0)

        plt.plot(recall_list, precision_list, label=f"Clase {c}")

    plt.title(f"Precision-Recall Curve (OvR) {title_suffix}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.show()


def LDA(X_train, y_train, X_test):
    clases = np.unique(y_train)
    n_features = X_train.shape[1]
    mu = {c: X_train[y_train==c].mean(axis=0).reshape(-1,1) for c in clases}
    Σ = np.zeros((n_features, n_features))
    for c in clases:
        Xc = X_train[y_train==c] - mu[c].ravel()
        Σ += Xc.T @ Xc
    Σ /= len(X_train)
    Σ_inv = np.linalg.pinv(Σ)
    priors = {c: np.mean(y_train==c) for c in clases}
    y_pred = []
    for x in X_test:
        scores = []
        for c in clases:
            m = mu[c].ravel()
            s = x @ Σ_inv @ m - 0.5 * m @ Σ_inv @ m + np.log(priors[c])
            scores.append(s)
        y_pred.append(clases[np.argmax(scores)])
    return np.array(y_pred), mu, Σ_inv, clases


def entrenar_LDA(X_train, y_train, X_val, y_val, dataset_name=""):
    print(f"Dataset: {dataset_name}")
    y_pred, mu, Σ_inv, clases = LDA(X_train, y_train, X_val)
    clases_unicas = np.unique(np.concatenate([y_train, y_val]))
    clase_to_idx = {c:i for i,c in enumerate(clases_unicas)}
    y_val_idx = np.array([clase_to_idx[c] for c in y_val])
    y_pred_idx = np.array([clase_to_idx[c] for c in y_pred])
    print("\nMatriz de Confusion:")
    print(matriz_confusion(y_val_idx, y_pred_idx))
    acc = accuracy(y_val_idx, y_pred_idx)
    print(f"Accuracy LDA: {acc:.4f}")
    acc = np.mean(y_pred_idx == y_val_idx)
    y_score = scores_ovr(y_val_idx, y_pred_idx, clases_unicas)
    metrics = metricas(y_val_idx, y_pred_idx, y_score, clases_unicas)
    graficar_curvas(y_val_idx, y_score, clases_unicas, title_suffix=f"LDA {dataset_name}")
    return metrics



def entrenar_logreg(X_train, y_train, X_val, y_val, dataset_name=""):
    print(f"\n=== Evaluacion LogReg: {dataset_name} ===")
    clases_unicas = np.unique(np.concatenate([y_train, y_val]))
    class_to_idx = {c:i for i,c in enumerate(clases_unicas)}
    index_to_class = {i:c for i,c in enumerate(clases_unicas)}
    y_train_idx = np.array([class_to_idx[c] for c in y_train])
    y_val_idx = np.array([class_to_idx[c] for c in y_val])

    W, b, y_pred_idx = logistic_regression(X_train, y_train_idx, X_val,
                                           lr=0.1, epochs=2000, reg_lambda=0.01)
    y_pred = np.array([index_to_class[i] for i in y_pred_idx])
    y_score = softmax(X_val @ W + b)

    print("\nMatriz de Confusion:")
    print(matriz_confusion(y_val_idx, y_pred_idx))
    acc = accuracy(y_val_idx, y_pred_idx)
    print(f"Accuracy Regresion: {acc:.4f}")
    metrics = metricas(y_val_idx, y_pred_idx, y_score, clases_unicas)
    graficar_curvas(y_val_idx, y_score, clases_unicas, title_suffix=f"LogReg {dataset_name}")
    return metrics

def entrenar_RF(X_train, y_train, X_val, y_val, dataset_name=""):
    print(f"\n=== Evaluación RF: {dataset_name} ===")
    clases_unicas = np.unique(np.concatenate([y_train, y_val]))
    class_to_idx = {c:i for i,c in enumerate(clases_unicas)}
    y_val_idx = np.array([class_to_idx[c] for c in y_val])

    rf = RandomForest(n_trees=10, max_depth=5)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    y_pred_idx = np.array([class_to_idx[c] for c in y_pred])

    y_score = rf.predict_proba(X_val)

    print("\nMatriz de Confusion:")
    print(matriz_confusion(y_val_idx, y_pred_idx))
    acc = accuracy(y_val_idx, y_pred_idx)
    print(f"Accuracy Random Forest: {acc:.4f}")
    metrics = metricas(y_val_idx, y_pred_idx, y_score, clases_unicas)
    graficar_curvas(y_val_idx, y_score, clases_unicas, title_suffix=f"RF {dataset_name}")
    return metrics