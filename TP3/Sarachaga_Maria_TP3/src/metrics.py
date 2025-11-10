import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_cross_entropy(Y_hat, Y_one_hot):
    m = Y_one_hot.shape[1] 
    Y_hat = np.clip(Y_hat, 1e-12, 1.0 - 1e-12)
    cost = (-1 / m) * np.sum(Y_one_hot * np.log(Y_hat))
    return float(cost)

def compute_accuracy(Y_hat, Y_true):
    predictions = np.argmax(Y_hat, axis=0)
    Y_true_flat = np.array(Y_true).flatten()
    correct_predictions = np.sum(predictions == Y_true_flat)
    return float(correct_predictions / len(Y_true_flat))

def compute_confusion_matrix_and_f1(Y_hat, Y_true, num_classes):
    Y_pred = np.argmax(Y_hat, axis=0)
    Y_true_flat = np.array(Y_true).flatten()
    m = len(Y_true_flat)

    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(m):
        true_label = int(Y_true_flat[i])
        pred_label = int(Y_pred[i])
        conf_matrix[true_label, pred_label] += 1

    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP

    precision = np.where((TP + FP) != 0, TP / (TP + FP), 0)
    recall = np.where((TP + FN) != 0, TP / (TP + FN), 0)
    f1_scores = np.where((precision + recall) != 0, 2 * (precision * recall) / (precision + recall), 0)
    f1_macro = np.mean(f1_scores)

    return conf_matrix, float(f1_macro)

def report_performance(model, X, Y_true, num_classes):
    Y_hat, _ = model.forward_propagation(X)
    Y_true_flat = np.array(Y_true).flatten()
    m = len(Y_true_flat)
    Y_true_OH = np.zeros((num_classes, m))
    Y_true_OH[Y_true_flat.astype(int), np.arange(m)] = 1

    conf_matrix, f1_macro = compute_confusion_matrix_and_f1(Y_hat, Y_true_flat, num_classes)

    metrics = {
        'Accuracy': compute_accuracy(Y_hat, Y_true_flat),
        'Cross-Entropy': compute_cross_entropy(Y_hat, Y_true_OH),
        'Confusion Matrix': conf_matrix,
        'F1-Score Macro': f1_macro
    }
    return metrics

def print_metrics(metrics, dataset_name="Dataset"):
    # 1. Definir cm (Matriz de Confusión)
    cm = metrics['Confusion Matrix']
    
    print(f" MÉTRICAS DE PERFORMANCE - {dataset_name.upper()}")
    print(f"  Accuracy:        {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
    print(f"  Cross-Entropy:   {metrics['Cross-Entropy']:.6f}")
    print(f"  F1-Score Macro:  {metrics['F1-Score Macro']:.4f}")
    
    # 2. Usar cm
    print(f"  Total predicciones correctas: {np.trace(cm)} / {np.sum(cm)}")

def plot_confusion_matrix(conf_matrix, title="Matriz de Confusión", normalize=False, figsize=(14,12)):
    plt.figure(figsize=figsize)
    if normalize:
        conf_matrix_plot = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_plot = np.nan_to_num(conf_matrix_plot)
        fmt = '.2f'
        vmax = 1.0
    else:
        conf_matrix_plot = conf_matrix
        fmt = 'd'
        vmax = None

    sns.heatmap(
        conf_matrix_plot,
        annot=False,
        fmt=fmt,
        cmap='Blues',
        cbar=True,
        square=True,
        linewidths=0,
        vmax=vmax,
        cbar_kws={'label': 'Proporción' if normalize else 'Cantidad de predicciones'}
    )

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Clase Real', fontsize=13)
    plt.xlabel('Clase Predicha', fontsize=13)

    num_classes = conf_matrix.shape[0]
    tick_positions = range(0, num_classes, max(1, num_classes // 10))
    tick_labels = range(0, num_classes, max(1, num_classes // 10))
    plt.xticks(tick_positions, tick_labels, rotation=0)
    plt.yticks(tick_positions, tick_labels, rotation=0)

    plt.tight_layout()
    plt.show()

    errors_per_class = np.sum(conf_matrix, axis=1) - np.diag(conf_matrix)
    worst_classes = np.argsort(errors_per_class)[-5:][::-1]

    print(f"\n--- Análisis de Errores: {title} ---")
    print("\nTop 5 clases con más errores:")
    for i, class_idx in enumerate(worst_classes, 1):
        total = np.sum(conf_matrix[class_idx, :])
        correct = conf_matrix[class_idx, class_idx]
        errors = errors_per_class[class_idx]
        accuracy_class = (correct / total * 100) if total > 0 else 0
        print(f"  {i}. Clase {class_idx}: {errors}/{total} errores (accuracy: {accuracy_class:.1f}%)")

    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and conf_matrix[i, j] > 0:
                confusions.append((i, j, conf_matrix[i, j]))
    confusions.sort(key=lambda x: x[2], reverse=True)
    print("\nTop 10 confusiones más frecuentes (Real → Predicho):")
    for i, (real, pred, count) in enumerate(confusions[:10], 1):
        print(f"  {i}. Clase {real} → Clase {pred}: {count} veces")

def plot_learning_curves(train_costs, val_costs, title="Learning Curves", learning_rate=0.01):
    plt.figure(figsize=(10,6))
    plt.plot(train_costs, label='Costo Entrenamiento')
    plt.plot(val_costs, label='Costo Validación')
    plt.title(f"{title} (lr={learning_rate})")
    plt.xlabel("Épocas")
    plt.ylabel("Costo")
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_accuracy_torch(y_true, y_pred):
    return np.mean(y_true == y_pred)


def calculate_f1_macro_torch(y_true, y_pred, num_classes):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true_flat)):
        true_label = int(y_true_flat[i])
        pred_label = int(y_pred_flat[i])
        conf_matrix[true_label, pred_label] += 1
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    
    for i in range(num_classes):
        if (TP[i] + FP[i]) > 0:
            precision[i] = TP[i] / (TP[i] + FP[i])
        else:
            precision[i] = 0.0
            
        if (TP[i] + FN[i]) > 0:
            recall[i] = TP[i] / (TP[i] + FN[i])
        else:
            recall[i] = 0.0
    f1_scores = np.zeros(num_classes)
    for i in range(num_classes):
        if (precision[i] + recall[i]) > 0:
            f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1_scores[i] = 0.0
    
    f1_macro = np.mean(f1_scores)
    classes_with_samples = (TP + FN) > 0
    if np.sum(classes_with_samples) > 0:
        f1_macro = np.mean(f1_scores[classes_with_samples])
    else:
        f1_macro = 0.0
    
    return float(f1_macro)

def evaluate_pytorch_model(model, X_data, y_true, num_classes):
    model.eval()
    with torch.no_grad():
        X_data = X_data.to(DEVICE)
        outputs = model(X_data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, y_true.to(DEVICE)).item()
        _, predicted_classes = torch.max(outputs.cpu(), 1)
        y_pred = predicted_classes.numpy()
        y_true_np = y_true.numpy()
        acc = calculate_accuracy_torch(y_true_np, y_pred)
        f1_macro = calculate_f1_macro_torch(y_true_np, y_pred, num_classes)
        conf_matrix = calculate_confusion_matrix_torch(y_true_np, y_pred, num_classes)
        
        return {
            'Accuracy': acc,
            'Cross-Entropy': loss,
            'F1-Score Macro': f1_macro,
            'Confusion Matrix': conf_matrix
        }

def calculate_confusion_matrix_torch(y_true, y_pred, num_classes):
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        conf_matrix[true_label, pred_label] += 1
    return conf_matrix
