import numpy as np
import matplotlib.pyplot as plt
import time
from model import MLP_Base
from metrics import report_performance


def visualize_sample_images(X, y, n_samples=6, figsize=(15, 3)):
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    plt.figure(figsize=figsize)
    
    for i, idx in enumerate(indices):
        img_flat = X[idx]
        label = y[idx]
        img_2d = img_flat.reshape(28, 28)
        
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(img_2d, cmap='gray')
        plt.title(f"Clase: {label}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_learning_curves(train_costs, val_costs, title="Evolución del Costo", 
                         figsize=(12, 6), learning_rate=None):
    plt.figure(figsize=figsize)
    epochs_range = range(1, len(train_costs) + 1)
    
    plt.plot(epochs_range, train_costs, 
             label='Costo de Entrenamiento', marker='o', markersize=4, linewidth=2)
    plt.plot(epochs_range, val_costs, 
             label='Costo de Validación', marker='s', markersize=4, linewidth=2)
    
    if learning_rate:
        title = f"{title} (LR={learning_rate})"
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def compare_train_val_metrics(train_metrics, val_metrics, threshold=0.05):
    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO: TRAIN vs VALIDATION")
    print("="*80)
    
    metrics_to_compare = ['Accuracy', 'Cross-Entropy', 'F1-Score Macro']
    
    print(f"\n{'Métrica':<20} {'Entrenamiento':>15} {'Validación':>15} {'Diferencia':>15}")
    print("-"*80)
    
    for metric in metrics_to_compare:
        train_val = train_metrics[metric]
        val_val = val_metrics[metric]
        diff = train_val - val_val
        
        if metric == 'Cross-Entropy':
            print(f"{metric:<20} {train_val:>15.6f} {val_val:>15.6f} {diff:>+15.6f}")
        else:
            print(f"{metric:<20} {train_val:>15.4f} {val_val:>15.4f} {diff:>+15.4f}")
    
    acc_diff = train_metrics['Accuracy'] - val_metrics['Accuracy']

def run_experiment(name, architecture, description="", save_key=None, **kwargs):
    print(f" {name}")
    
    if description:
        print(f"\nDescripción: {description}")
    
    print(f"Arquitectura: {architecture}")
    print(f"\nHiperparámetros:")
    for key, value in kwargs.items():
        if callable(value):
            print(f"  {key}: {value.__name__}")
        else:
            print(f"  {key}: {value}")
    
    model = MLP_Base(architecture)
    
    start_time = time.time()
    
    train_costs, val_costs = model.train(
        kwargs.get('X_train_T'), 
        kwargs.get('y_train'), 
        kwargs.get('X_val_T'), 
        kwargs.get('y_val'),
        epochs=kwargs.get('epochs', kwargs.get('EPOCHS_EXP', 50)),
        learning_rate=kwargs.get('learning_rate', kwargs.get('LR_BASE', 0.01)),
        **{k: v for k, v in kwargs.items() if k not in ['epochs', 'learning_rate', 
                                                          'X_train_T', 'y_train', 
                                                          'X_val_T', 'y_val', 
                                                          'NUM_CLASSES', 'trained_models',
                                                          'EPOCHS_EXP', 'LR_BASE']}
    )
    
    elapsed_time = time.time() - start_time
    

    print(f"Evaluando performance...")
    val_metrics = report_performance(model, kwargs.get('X_val_T'), 
                                     kwargs.get('y_val'), 
                                     kwargs.get('NUM_CLASSES'))
    
    plt.figure(figsize=(12, 5))
    epochs_range = range(1, len(train_costs) + 1)
    plt.plot(epochs_range, train_costs, label='Train Loss', 
             marker='o', markersize=3, linewidth=2, alpha=0.8)
    plt.plot(epochs_range, val_costs, label='Validation Loss', 
             marker='s', markersize=3, linewidth=2, alpha=0.8)
    plt.title(f'{name}\nTiempo: {elapsed_time:.2f}s | Val Acc: {val_metrics["Accuracy"]:.4f}',
              fontsize=12, fontweight='bold')
    plt.xlabel('Época', fontsize=11)
    plt.ylabel('Cross-Entropy Loss', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    


    print(f"RESULTADOS FINALES:")
    print(f"  Accuracy (Val):     {val_metrics['Accuracy']:.4f} ({val_metrics['Accuracy']*100:.2f}%)")
    print(f"  F1-Score Macro:     {val_metrics['F1-Score Macro']:.4f}")
    print(f"  Cross-Entropy:      {val_metrics['Cross-Entropy']:.6f}")
    print(f"  Tiempo:             {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
    print(f"  Épocas completadas: {len(train_costs)}/{kwargs.get('epochs', kwargs.get('EPOCHS_EXP', 50))}")
    print("-"*80)
    
    if save_key and 'trained_models' in kwargs:
        kwargs['trained_models'][save_key] = model
    
    return model, val_metrics, elapsed_time, len(train_costs)


def train_all_models(configs, X_train, y_train, X_val, y_val):
    resultados = {}
    mejor_modelo = None
    mejor_val_loss = float('inf')
    
    for nombre, cfg in configs.items():
        print("\n" + "="*90)
        print(f"Entrenando {nombre}")
        print(cfg["description"])
        print("="*90)
        
        model = MLP_Base(cfg["layer_dims"])
        
        start_time = time.time()
        train_costs, val_costs = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            lr_schedule=cfg.get("lr_schedule", None),
            decay_rate=cfg.get("decay_rate", None),
            min_lr_factor=cfg.get("min_lr_factor", None),
            optimizer=cfg["optimizer"],
            beta1=cfg["beta1"],
            beta2=cfg["beta2"],
            epsilon=cfg["epsilon"],
            lambda_l2=cfg["lambda_l2"],
            patience=cfg["patience"]
        )
        duracion = time.time() - start_time
        
        resultados[nombre] = {
            "model": model,
            "train_costs": train_costs,
            "val_costs": val_costs,
            "tiempo": duracion
        }
        
        plt.figure(figsize=(10, 5))
        epochs_range = range(1, len(train_costs) + 1)
        plt.plot(epochs_range, train_costs, label='Train Loss', 
                 marker='o', markersize=3, linewidth=2, alpha=0.8)
        plt.plot(epochs_range, val_costs, label='Validation Loss', 
                 marker='s', markersize=3, linewidth=2, alpha=0.8)
        plt.title(f'{nombre}\nTiempo: {duracion:.2f}s', 
                  fontsize=12, fontweight='bold')
        plt.xlabel('Época', fontsize=11)
        plt.ylabel('Cross-Entropy Loss', fontsize=11)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
        print(f"\n{nombre} finalizado en {duracion:.2f} segundos")
        print(f"Último costo de entrenamiento: {train_costs[-1]:.4f}")
        print(f"Último costo de validación: {val_costs[-1]:.4f}")
        
        if val_costs[-1] < mejor_val_loss:
            mejor_val_loss = val_costs[-1]
            mejor_modelo = nombre
    
    print(f"El modelo con mejor desempeño en validación es: {mejor_modelo}")
    
    return resultados, mejor_modelo

def convert_to_pytorch_tensors(X, y):
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    return X_tensor, y_tensor

