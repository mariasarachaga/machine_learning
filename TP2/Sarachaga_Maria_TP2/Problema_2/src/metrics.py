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

def auc_trapecio(y_true, prob_pred):
    y_true_sorted = np.array(y_true)
    prob_sorted = np.array(prob_pred)
    sorted_idx = np.argsort(prob_sorted)[::-1]
    y_true_sorted = y_true_sorted[sorted_idx]

    #roc
    P = np.sum(y_true_sorted)
    N = len(y_true_sorted) - P
    tpr_list, fpr_list = [], []
    tp = fp = 0
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1: tp += 1
        else: fp += 1
        tpr_list.append(tp / P if P > 0 else 0)
        fpr_list.append(fp / N if N > 0 else 0)
    auc_roc = np.trapz(tpr_list, fpr_list)

    #pr
    precision_list, recall_list = [], []
    tp = fp = 0
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1: tp += 1
        else: fp += 1
        precision_list.append(tp / (tp+fp) if (tp+fp) > 0 else 0)
        recall_list.append(tp / P if P > 0 else 0)
    auc_pr = np.trapz(precision_list, recall_list)

    return auc_roc, auc_pr

def calcular_metricas(y_true_bin, y_pred_bin, prob_bin):
    auc_roc, auc_pr = auc_trapecio(y_true_bin, prob_bin)
    return {
        "precision": precision(y_true_bin, y_pred_bin),
        "recall": recall(y_true_bin, y_pred_bin),
        "f1": f1_score_manual(y_true_bin, y_pred_bin),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr
    }


