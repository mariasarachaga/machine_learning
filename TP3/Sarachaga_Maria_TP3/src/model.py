import numpy as np
import math
import time
import torch
from torch import nn

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0 
    return dZ

def one_hot_encode(Y, num_classes):
    m = Y.shape[0] if Y.ndim == 1 else Y.shape[1]
    Y_one_hot = np.zeros((num_classes, m))
    if Y.ndim == 1:
        Y_one_hot[Y.astype(int), np.arange(m)] = 1
    else:
        Y_one_hot[Y.astype(int).flatten(), np.arange(m)] = 1
    return Y_one_hot

def to_one_hot(Y, num_classes):
    return one_hot_encode(Y, num_classes)
def backward_activation(dA, Z):
    return relu_backward(dA, Z)



def linear_schedule(initial_lr, epoch, total_epochs, min_lr_factor=0.1):
    progress = epoch / total_epochs
    lr = initial_lr * (1 - (1 - min_lr_factor) * progress)
    return max(lr, initial_lr * min_lr_factor)

def exponential_schedule(initial_lr, epoch, decay_rate=0.95):
    return initial_lr * (decay_rate ** epoch)


class MLP_Base:
    def __init__(self, layer_dims):
        self.L = len(layer_dims) - 1
        self.parameters = {}
        for l in range(1, self.L + 1):
            n_in = layer_dims[l-1]
            n_out = layer_dims[l]
            self.parameters['W' + str(l)] = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)
            self.parameters['b' + str(l)] = np.zeros((n_out, 1))
            
        # print(f"MLP inicializado con {self.L - 1} capas ocultas. Arquitectura: {layer_dims}")

    def forward_propagation(self, X):
        cache = {}
        A = X 
        for l in range(1, self.L): 
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A = relu(Z)
            cache['A' + str(l-1)] = A_prev
            cache['Z' + str(l)] = Z
        W = self.parameters['W' + str(self.L)]
        b = self.parameters['b' + str(self.L)]
        A_prev = A
        Z = np.dot(W, A_prev) + b
        Y_hat = softmax(Z)
        cache['A' + str(self.L - 1)] = A_prev
        cache['Z' + str(self.L)] = Z
        return Y_hat, cache

    def compute_cost(self, Y_hat, Y_one_hot, lambda_l2=0.0):
        m = Y_one_hot.shape[1] 
        Y_hat = np.clip(Y_hat, 1e-12, 1.0 - 1e-12)
        cross_entropy_cost = (-1 / m) * np.sum(Y_one_hot * np.log(Y_hat))
        
        l2_regularization_cost = 0
        if lambda_l2 > 0:
            for l in range(1, self.L + 1):
                l2_regularization_cost += np.sum(self.parameters["W" + str(l)] ** 2)
            l2_regularization_cost = (lambda_l2 / (2 * m)) * l2_regularization_cost
        return cross_entropy_cost + l2_regularization_cost
    
    def backward_propagation(self, Y_hat, Y_one_hot, cache, lambda_l2=0.0):
        grads = {}
        m = Y_one_hot.shape[1]
        
        dZ = Y_hat - Y_one_hot 
        A_prev = cache['A' + str(self.L - 1)]
        
        grads['dW' + str(self.L)] = (1 / m) * np.dot(dZ, A_prev.T)
        grads['db' + str(self.L)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        if lambda_l2 > 0:
            grads['dW' + str(self.L)] += (lambda_l2 / m) * self.parameters["W" + str(self.L)]

        for l in reversed(range(1, self.L)):
            dA = np.dot(self.parameters['W' + str(l + 1)].T, dZ)
            Z = cache['Z' + str(l)]
            dZ = backward_activation(dA, Z)
            
            A_prev = cache['A' + str(l - 1)]
            grads['dW' + str(l)] = (1 / m) * np.dot(dZ, A_prev.T)
            grads['db' + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            
            if lambda_l2 > 0:
                grads['dW' + str(l)] += (lambda_l2 / m) * self.parameters["W" + str(l)]
            
        return grads
    
    def update_parameters_sgd(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
        return self.parameters
    
    def initialize_adam_moments(self):
        v = {}
        s = {}
        for l in range(1, self.L + 1):
            v["dW" + str(l)] = np.zeros(self.parameters["W" + str(l)].shape)
            v["db" + str(l)] = np.zeros(self.parameters["b" + str(l)].shape)
            s["dW" + str(l)] = np.zeros(self.parameters["W" + str(l)].shape)
            s["db" + str(l)] = np.zeros(self.parameters["b" + str(l)].shape)
        return v, s

    def update_parameters_adam(self, grads, v, s, t, learning_rate, beta1, beta2, epsilon):
        for l in range(1, self.L + 1):
            v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
            v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
            
            s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
            s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)
            
            v_corrected_dw = v["dW" + str(l)] / (1 - beta1 ** t)
            v_corrected_db = v["db" + str(l)] / (1 - beta1 ** t)
            s_corrected_dw = s["dW" + str(l)] / (1 - beta2 ** t)
            s_corrected_db = s["db" + str(l)] / (1 - beta2 ** t)
            
            self.parameters["W" + str(l)] -= learning_rate * v_corrected_dw / (np.sqrt(s_corrected_dw) + epsilon)
            self.parameters["b" + str(l)] -= learning_rate * v_corrected_db / (np.sqrt(s_corrected_db) + epsilon)
        return self.parameters, v, s
    
    def train(self, X_train, Y_train, X_val, Y_val, epochs=50, 
                    learning_rate=0.01, batch_size=128, 
                    lr_schedule=None, decay_rate=None, min_lr_factor=None,
                    optimizer='sgd', beta1=0.9, beta2=0.999, epsilon=1e-8, 
                    lambda_l2=0.0, patience=5):
        
        m = X_train.shape[1]
        if batch_size is None or batch_size >= m:
            batch_size = m
            print("Usando Full Batch Gradient Descent.")
        else:
             print(f"Usando Mini-Batch (TamaÃ±o: {batch_size}).")
        
        num_batches = math.ceil(m / batch_size)
        n_output = self.parameters['W' + str(self.L)].shape[0]
        Y_val_OH = to_one_hot(Y_val, n_output)
        
        if optimizer == 'adam':
            v, s = self.initialize_adam_moments() 
            t = 0 
        
        best_val_cost = float('inf')
        epochs_no_improvement = 0
        
        train_costs = []
        val_costs = []

        for epoch in range(1, epochs + 1):
            permutation = np.random.permutation(m)
            shuffled_X = X_train[:, permutation]
            shuffled_Y = Y_train[permutation] 

            current_lr = learning_rate
            if lr_schedule:
                if lr_schedule.__name__ == 'linear_schedule':
                    current_lr = linear_schedule(learning_rate, epoch, epochs, min_lr_factor=min_lr_factor)
                elif lr_schedule.__name__ == 'exponential_schedule':
                    current_lr = exponential_schedule(learning_rate, epoch, decay_rate=decay_rate)
            
            epoch_cost = 0
            for i in range(num_batches):
                X_batch = shuffled_X[:, i * batch_size: min( (i + 1) * batch_size, m) ]
                Y_batch = shuffled_Y[i * batch_size: min( (i + 1) * batch_size, m) ]
                
                Y_one_hot = to_one_hot(Y_batch, n_output)
                
                Y_hat, caches = self.forward_propagation(X_batch)
                batch_cost = self.compute_cost(Y_hat, Y_one_hot, lambda_l2=lambda_l2)
                epoch_cost += batch_cost * (X_batch.shape[1] / m)

                grads = self.backward_propagation(Y_hat, Y_one_hot, caches, lambda_l2=lambda_l2)
                
                if optimizer == 'adam':
                    t += 1
                    self.parameters, v, s = self.update_parameters_adam(
                        grads, v, s, t, current_lr, beta1, beta2, epsilon
                    )
                else: 
                    self.parameters = self.update_parameters_sgd(
                        grads, current_lr
                    )

            train_costs.append(epoch_cost)
            
            Y_val_hat, _ = self.forward_propagation(X_val)
            val_cost = self.compute_cost(Y_val_hat, Y_val_OH, lambda_l2=lambda_l2)
            val_costs.append(val_cost)
            
            # 5. Early Stopping
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1
                if epochs_no_improvement >= patience:
                    print(f"\n--- Early Stopping activado en la Ã‰poca {epoch} (Paciencia={patience}) ---")
                    break

            if epoch % 10 == 0 or epoch == epochs or epoch == 1:
                print(f"Epoch {epoch}/{epochs} | Costo Train: {epoch_cost:.4f} | Costo Val: {val_cost:.4f} (LR: {current_lr:.6f})")

        return train_costs, val_costs


class MLP_PyTorch(nn.Module):
    def __init__(self, layer_dims, activation=nn.ReLU):
        super(MLP_PyTorch, self).__init__()
        layers = []
        for i in range(len(layer_dims) - 2):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(activation())
        # Capa de salida (sin activación Softmax, ya que CrossEntropyLoss la incluye)
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MLP_PyTorch_Advanced(nn.Module):
    def __init__(self, layer_dims, activation_name='ReLU', dropout_rate=0.0):
        super(MLP_PyTorch_Advanced, self).__init__()
        activation_map = {
            'ReLU': nn.ReLU,
            'LeakyReLU': nn.LeakyReLU,
            'SiLU': nn.SiLU,
            'Swish': nn.SiLU, 
            'GELU': nn.GELU
        }
        
        activation_fn = activation_map.get(activation_name, nn.ReLU)
        layers = []
        
        for i in range(len(layer_dims) - 2):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def add_gaussian_noise_numpy(X, std_dev):
    noise = np.random.normal(0, std_dev, X.shape)
    X_noisy = X + noise
    # Recortar valores para mantenerlos en el rango [0, 1]
    X_noisy = np.clip(X_noisy, 0.0, 1.0)
    return X_noisy

def train_pytorch_model(model, config, X_train, y_train, X_val, y_val, num_classes):
    model.to(DEVICE)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                               weight_decay=config['lambda_l2'])
    else:  # sgd
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], 
                              weight_decay=config['lambda_l2'])
    
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    if config.get('lr_schedule') == 'exponential':
        decay_rate = config.get('decay_rate', 0.95)  # Valor por defecto si no existe
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    elif config.get('lr_schedule') == 'linear':
        def linear_lambda(epoch):
            min_factor = config.get('min_lr_factor', 0.1)  # Valor por defecto si no existe
            total_epochs = config['epochs']
            return max(min_factor, 1.0 - epoch / total_epochs)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lambda)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = copy.deepcopy(model.state_dict())
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()
        val_metrics = evaluate_pytorch_model(model, X_val, y_val, num_classes)
        val_loss = val_metrics['Cross-Entropy']
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', config['epochs'] + 1):
                print(f"Early Stopping en época {epoch + 1}.")
                break
                
    end_time = time.time()
    total_time = end_time - start_time
   
    model.load_state_dict(best_model_state)
    
    return model, total_time, epoch + 1