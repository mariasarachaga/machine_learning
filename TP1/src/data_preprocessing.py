import pandas as pd
import numpy as np

SQFT_TO_M2 = 0.092903
COLUMN_ORDER = [
    'metros_cubiertos', 'Área', 'pisos', 'ambientes', 'edad', 'pileta',
    'cubiertos_ratio', 'cubiertos_x_pisos', 'log_area', 'sqrt_area',
    'area_x_ambientes', 'ambientes_x_pisos', 'tipo_depto', 'tipo_ph'
]

def preprocess_and_engineer(df, edad_mean=None):
    df = df.dropna(subset=['precio'])
    df = df[df['precio'] != 0].copy()

    df['pisos'] = df['pisos'].fillna(1)
    if edad_mean is None:
        edad_mean = df['edad'].mean()
    df['edad'] = df['edad'].fillna(edad_mean)
    
    df['pileta'] = df['pileta'].map({False: 0, True: 1})
    mask_sqft = df['unidades'] == 'sqft'
    df.loc[mask_sqft, 'Área'] = df.loc[mask_sqft, 'Área'] * SQFT_TO_M2
    df = df.drop(columns=['unidades'])
    
    df["cubiertos_ratio"] = df["metros_cubiertos"] / df["Área"]
    df["cubiertos_x_pisos"] = df["metros_cubiertos"] * df["pisos"]
    df["log_area"] = np.log1p(df["Área"])
    df["sqrt_area"] = np.sqrt(df["Área"])
    df["area_x_ambientes"] = df["Área"] * df["ambientes"]
    df["ambientes_x_pisos"] = df["ambientes"] * df["pisos"]
    
    df_encoded = pd.get_dummies(df, columns=['tipo'], drop_first=True)
    df_encoded = df_encoded.drop(columns=['lat', 'lon'], errors='ignore')

    return df_encoded, edad_mean

def z_score_normalize(X_train, X_test):
    mu = X_train.mean()
    sigma = X_train.std()

    sigma[sigma == 0] = 1 

    X_train_normalized = (X_train - mu) / sigma
    X_test_normalized = (X_test - mu) / sigma
    
    return X_train_normalized, X_test_normalized, mu, sigma