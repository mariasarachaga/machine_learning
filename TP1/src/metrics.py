import numpy as np

def mse(y,y_pred):
    return ((y - y_pred)**2).mean()

def rmse(y,y_pred):
    return np.sqrt(((y - y_pred)**2).mean())
    
def mae(y,y_pred):
    return (abs(y-y_pred)).mean()