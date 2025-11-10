import numpy as np
import pandas as pd

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class RegresionLineal:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coef = None

    def array(self):
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.x = np.concatenate([np.ones((self.x.shape[0], 1)), self.x], axis=1)

    def pseudoinversa(self, lambda2=0):
        if lambda2 == 0:
            self.coef = np.linalg.pinv(self.x) @ self.y
        else:
            n_features = self.x.shape[1]
            I = np.eye(n_features)
            I[0, 0] = 0  
            self.coef = np.linalg.inv(self.x.T @ self.x + lambda2 * I) @ (self.x.T @ self.y)

    def funcion_costo(self, coef, lambda1=0, lambda2=0):
        predicciones = self.x @ coef
        ECM = mse(self.y, predicciones)
        if lambda2 > 0:
            ECM += lambda2 * np.sum(coef[1:]**2)
        error = predicciones - self.y
        return ECM, error

    def descenso_gradiente(self, alpha=0.01, iteraciones=10000, tol=1e-6, lambda1=0, lambda2=0):
        coef = np.zeros(self.x.shape[1])
        m = self.y.size
        for i in range(iteraciones):
            predicciones = self.x @ coef
            error = predicciones - self.y
            gradiente = (1/m) * (self.x.T @ error)
            if lambda2 > 0:
                gradiente[1:] += (lambda2 / m) * coef[1:]
            coef -= alpha * gradiente
            if lambda1 > 0:
                thresh = alpha * (lambda1 / m)
                coef[1:] = np.sign(coef[1:]) * np.maximum(np.abs(coef[1:]) - thresh, 0)
            if np.linalg.norm(gradiente, ord=2) < tol:
                break

        self.coef = coef
        return self.coef, None
    
    def predecir(self, X):
        X_mat = np.array(X)
        X_mat = np.concatenate([np.ones((X_mat.shape[0], 1)), X_mat], axis=1) 
        return X_mat @ self.coef

    def imprimir_coef(self, nombres_columnas=None):
        nombres = ["Termino independiente"] + list(nombres_columnas) if nombres_columnas is not None else [f"x{i}" for i in range(len(self.coef))]
        df_coef = pd.DataFrame({'Feature': nombres, 'Coeficiente': self.coef})
        print(df_coef)

###############################################################################
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

def logistic_regression(X_train, y_train, X_test, lr=0.01, epochs=10000, reg_lambda=0.0, sample_weights=None):
    X_train = np.array(X_train)
    y_train = np.array(y_train).astype(int)
    X_test = np.array(X_test)
    n_samples, n_features = X_train.shape
    num_classes = len(np.unique(y_train))
    W = np.zeros((n_features, num_classes))
    b = np.zeros((1, num_classes))
    y_one_hot = one_hot(y_train, num_classes)
    if sample_weights is None:
        sample_weights = np.ones(n_samples)
    sample_weights = sample_weights.reshape(-1, 1) 
    for _ in range(epochs):
        scores = np.dot(X_train, W) + b
        probs = softmax(scores)
        weights_exp = sample_weights.reshape(-1, 1)
        dW = (1/n_samples) * np.dot(X_train.T, (probs - y_one_hot) * weights_exp) + (reg_lambda/n_samples) * W
        db = (1/n_samples) * np.sum((probs - y_one_hot) * weights_exp, axis=0, keepdims=True)
        W -= lr * dW
        b -= lr * db
    y_pred = np.argmax(softmax(np.dot(X_test, W) + b), axis=1)

    def probabilidades(X):
        X = np.array(X)
        scores = np.dot(X, W) + b
        return softmax(scores) 

    return W, b, y_pred, probabilidades

###############################################################################################################
def SMOTE(X, y, clase_minoritaria=0, random_state=42):
    np.random.seed(random_state)
    X_minor = X[y == clase_minoritaria]
    y_minor = y[y == clase_minoritaria]
    X_mayor = X[y != clase_minoritaria]
    y_mayor = y[y != clase_minoritaria]
    n_samples_generar = len(X_mayor) - len(X_minor)
    synthetic_samples = []

    for _ in range(n_samples_generar):
        idx1, idx2 = np.random.randint(0, X_minor.shape[0], size=2)
        x_i = X_minor[idx1]
        x_j = X_minor[idx2]
    
        gap = np.random.rand()
        x_new = x_i + gap * (x_j - x_i)
        synthetic_samples.append(x_new)
    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.full(len(X_synthetic), clase_minoritaria)
    X_res = np.vstack([X_mayor, X_minor, X_synthetic])
    y_res = np.hstack([y_mayor, y_minor, y_synthetic])
    
    return X_res, y_res