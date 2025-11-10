import matplotlib.pyplot as plt
import numpy as np

def matriz_confusion(y_true, y_pred, num_classes=None):
    if num_classes is None:
        num_classes = len(np.unique(y_true))
    conf = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    return conf

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    classes = np.unique(y_true)
    precisions = []
    for c in classes:
        tp = np.sum((y_true==c) & (y_pred==c))
        fp = np.sum((y_true!=c) & (y_pred==c))
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return np.mean(precisions)

def recall(y_true, y_pred):
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        tp = np.sum((y_true==c) & (y_pred==c))
        fn = np.sum((y_true==c) & (y_pred!=c))
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return np.mean(recalls)

def f1_score_manual(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-10)

def auc_trapecio(x, y):
    n = min(len(x), len(y))
    return np.trapz(y[:n], x[:n])


def plot_curves(y_true, y_score, num_classes):
    plt.figure(figsize=(12,5))

    #curva roc
    plt.subplot(1,2,1)
    for i in range(num_classes):
        fpr, tpr = [], []
        thresholds = np.sort(np.unique(y_score[:,i]))[::-1]
        P = np.sum(y_true==i)
        N = len(y_true)-P
        for t in thresholds:
            y_pred = (y_score[:,i]>=t).astype(int)
            tp = np.sum((y_true==i) & (y_pred==1))
            fp = np.sum((y_true!=i) & (y_pred==1))
            fpr.append(fp/N if N>0 else 0)
            tpr.append(tp/P if P>0 else 0)
        plt.plot(fpr, tpr, label=f"Clase {i}")
        auc = auc_trapecio(fpr, tpr)
        print(f"AUC-ROC Clase {i}: {auc:.4f}")
    plt.title("ROC Curve (OvR)")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(True)

    #cruva precision-recall
    plt.subplot(1,2,2)
    for i in range(num_classes):
        precision, recall = [], []
        thresholds = np.sort(np.unique(y_score[:,i]))[::-1]
        P = np.sum(y_true==i)
        for t in thresholds:
            y_pred = (y_score[:,i]>=t).astype(int)
            tp = np.sum((y_true==i) & (y_pred==1))
            fp = np.sum((y_true!=i) & (y_pred==1))
            precision.append(tp/(tp+fp) if (tp+fp)>0 else 1)
            recall.append(tp/P if P>0 else 0)
        plt.plot(recall, precision, label=f"Clase {i}")
        auc = auc_trapecio(recall, precision)
        print(f"AUC-PR Clase {i}: {auc:.4f}")
    plt.title("Precision-Recall Curve (OvR)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)

    plt.show()
