import numpy as np
import pandas as pd
from src.metrics import mse

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
            I[0, 0] = 0  # no regularizamos el término independiente
            self.coef = np.linalg.inv(self.x.T @ self.x + lambda2 * I) @ (self.x.T @ self.y)

    def funcion_costo(self, coef, lambda1=0, lambda2=0):
        predicciones = self.x @ coef
        ECM = mse(self.y, predicciones)
        # regularización
        if lambda2 > 0:
            ECM += lambda2 * np.sum(coef[1:]**2)
        if lambda1 > 0:
            ECM += lambda1 * np.sum(np.abs(coef[1:]))
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
        nombres = ["Término independiente"] + list(nombres_columnas) if nombres_columnas is not None else [f"x{i}" for i in range(len(self.coef))]
        df_coef = pd.DataFrame({'Feature': nombres, 'Coeficiente': self.coef})
        print(df_coef)

