import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import RegresionLineal, logistic_regression
from src.metrics import f1_score_manual

def codificar_categoricas(df):
    df = df.copy()
    df['GeneticMutation'] = df['GeneticMutation'].map({'Absnt': 0, 'Presnt': 1})
    df['CellType'] = df['CellType'].map({'Epthlial': 0, 'Mesnchymal': 1, '???': 2})
    return df

def tratamiento_nan(train_df, val_df, target_column='Diagnosis'):
    train_df_proc = codificar_categoricas(train_df.copy()).dropna().reset_index(drop=True)
    val_df_proc   = codificar_categoricas(val_df.copy()).dropna().reset_index(drop=True)
    cols_num = train_df_proc.select_dtypes(include=[np.number]).columns
    cols_norm = [c for c in cols_num if c not in ['CellType','GeneticMutation', target_column]]
    mean_train = train_df_proc[cols_norm].mean()
    std_train  = train_df_proc[cols_norm].std().replace(0,1)
    train_df_proc[cols_norm] = (train_df_proc[cols_norm] - mean_train) / std_train
    val_df_proc[cols_norm]   = (val_df_proc[cols_norm] - mean_train) / std_train

    return train_df_proc, val_df_proc, mean_train, std_train

def entrenar_con_lambda_optimo(X_train, y_train, X_val, y_val,lr=0.01, epochs=10000,lambdas=np.linspace(0.0001,10,100),sample_weights=None):
    best_lambda, best_f1 = None, -1
    for lam in lambdas:
        W_tmp, b_tmp, y_pred_val, predict_proba_val = logistic_regression(X_train, y_train, X_val, lr=lr, epochs=epochs, reg_lambda=lam,sample_weights=sample_weights)
        f1 = f1_score_manual(y_val, y_pred_val)
        if f1 > best_f1:
            best_f1, best_lambda = f1, lam
    print(f"\nMejor lambda: {best_lambda}, F1={best_f1:.4f}")
    W, b, y_pred, predict_proba_fn = logistic_regression(X_train, y_train, X_val, lr=lr, epochs=epochs, reg_lambda=best_lambda,sample_weights=sample_weights)
    y_score = predict_proba_fn(X_val)

    return W, b, y_pred, predict_proba_fn, y_score, best_lambda

def imputar_dataset(X_train, X_val, corr_matrix, k=2):
    X_train_imp = X_train.copy()
    X_val_imp   = X_val.copy()

    cols_moda = ["CellType", "GeneticMutation"]
    cols_mediana = ["NucleusDensity", "MitosisRate"]

    for col in X_train.columns:
        if X_train[col].isna().any() or X_val[col].isna().any():
            if col in cols_moda:
                moda_train = X_train[col].mode().iloc[0] if not X_train[col].mode().empty else 0
                X_train_imp[col].fillna(moda_train, inplace=True)
                X_val_imp[col].fillna(moda_train, inplace=True)
                continue
            if col in cols_mediana:
                mediana_train = X_train[col].median()
                X_train_imp[col].fillna(mediana_train, inplace=True)
                X_val_imp[col].fillna(mediana_train, inplace=True)
                continue
            correlaciones = corr_matrix[col].drop(col).abs().sort_values(ascending=False)
            top_features = correlaciones.head(k).index.tolist()
            mask_train_fit = X_train[col].notna() & X_train[top_features].notna().all(axis=1)
            if mask_train_fit.sum() == 0:
                continue
            X_fit = X_train.loc[mask_train_fit, top_features]
            y_fit = X_train.loc[mask_train_fit, col]
            modelo = RegresionLineal(X_fit.values, y_fit.values)
            modelo.array()
            modelo.pseudoinversa()
            mask_missing_train = X_train_imp[col].isna()
            if mask_missing_train.any():
                X_pred_train = X_train_imp.loc[mask_missing_train, top_features].fillna(X_train_imp[top_features].mean())
                X_train_imp.loc[mask_missing_train, col] = modelo.predecir(X_pred_train.values)
            mask_missing_val = X_val_imp[col].isna()
            if mask_missing_val.any():
                X_pred_val = X_val_imp.loc[mask_missing_val, top_features].fillna(X_train_imp[top_features].mean())
                X_val_imp.loc[mask_missing_val, col] = modelo.predecir(X_pred_val.values)

    return X_train_imp, X_val_imp


def preprocesar_imputado(train, test, numericas, categoricas, target='Diagnosis', k=1):
    Q1 = train[numericas].quantile(0.25)
    Q3 = train[numericas].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    train_num_clean = train[numericas].mask((train[numericas] < lower) | (train[numericas] > upper))
    test_num_clean  = test[numericas].mask((test[numericas] < lower) | (test[numericas] > upper))
    X_train_outliers = pd.concat([train_num_clean, train[categoricas[:-1]]], axis=1)
    X_test_outliers  = pd.concat([test_num_clean,  test[categoricas[:-1]]], axis=1)
    train_outliers = pd.concat([X_train_outliers, train[target]], axis=1)
    test_outliers  = pd.concat([X_test_outliers,  test[target]], axis=1)
    train_outliers = codificar_categoricas(train_outliers)
    test_outliers  = codificar_categoricas(test_outliers)

    X_train, X_test = train_outliers.drop(columns=[target]), test_outliers.drop(columns=[target])
    y_train, y_test = train_outliers[target], test_outliers[target]
    corr_matrix = X_train.corr(numeric_only=True)
    X_train_imp, X_test_imp = imputar_dataset(X_train, X_test, corr_matrix, k=k)

    train_imputado = pd.concat([X_train_imp, y_train], axis=1).dropna().reset_index(drop=True)
    test_imputado  = pd.concat([X_test_imp,  y_test],  axis=1).dropna().reset_index(drop=True)
    cols_num = train_imputado.select_dtypes(include=[float, int]).columns
    cols_norm = [c for c in cols_num if c not in categoricas + [target]]
    mean_train, std_train = train_imputado[cols_norm].mean(), train_imputado[cols_norm].std()
    train_imputado[cols_norm] = (train_imputado[cols_norm] - mean_train) / std_train
    test_imputado[cols_norm]  = (test_imputado[cols_norm] - mean_train) / std_train
    X_train_imp = train_imputado.drop(columns=[target]).values
    y_train_imp = train_imputado[target].values.astype(int)
    X_test_imp  = test_imputado.drop(columns=[target]).values
    y_test_imp  = test_imputado[target].values.astype(int)

    return X_train_imp, y_train_imp, X_test_imp, y_test_imp, train_imputado, test_imputado

def correlaciones_altas(corr, umbral=0.8):
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            val = corr.iloc[i, j]
            if abs(val) > umbral:
                high_corr.append((corr.columns[i], corr.columns[j], val))
    return high_corr

def graficar_corr(df, redondeo=2, features_num=None, target='Diagnosis'):
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns
    corr = df[num_cols].corr().round(redondeo)

    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Matriz de correlación entre features numéricos")
    plt.show()

    if features_num is not None:
        sample = df.sample(500) if len(df) > 500 else df
        sns.pairplot(sample, vars=num_cols[:features_num], hue=target, palette='Set1')
        plt.show()

def visualizar_distribuciones(data, categoricas=None, columnas=3):
    if categoricas is None:
        categoricas = ["CellType", "GeneticMutation", "Diagnosis"]

    features = list(data.columns)
    filas = (len(features) + columnas - 1) // columnas
    fig, axes = plt.subplots(filas, columnas, figsize=(15, 5*filas))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        es_numerica = feature not in categoricas
        sns.histplot(data[feature], bins=100, ax=axes[i], color=(224/255,0,123/255), stat="density")
        if es_numerica:
            sns.kdeplot(data[feature].dropna(), ax=axes[i], color=(0,0,1))
        axes[i].set_title(feature)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

    numericas = [f for f in features if f not in categoricas]
    filas_box = (len(numericas) + columnas - 1) // columnas
    fig, axes = plt.subplots(filas_box, columnas, figsize=(15, 5*filas_box))
    axes = axes.flatten()
    for i, feature in enumerate(numericas):
        sns.boxplot(y=data[feature], ax=axes[i], color=(224/255,0,123/255))
        axes[i].set_title(feature)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, len(categoricas), figsize=(18,5))
    if len(categoricas) == 1: axes = [axes]
    for i, cat in enumerate(categoricas):
        sns.countplot(x=cat, data=data, ax=axes[i], palette='Set2')
        axes[i].set_title(f'Conteo de {cat}')
    plt.tight_layout()
    plt.show()

def calc_roc_pr(y_true, y_score):
    y_true, y_score = np.asarray(y_true).astype(int), np.asarray(y_score).astype(float)
    desc = np.argsort(y_score)[::-1]
    y_sorted = y_true[desc]
    tp = np.cumsum(y_sorted==1)
    fp = np.cumsum(y_sorted==0)
    P, N = np.sum(y_true==1), np.sum(y_true==0)
    tpr = np.concatenate(([0.0], tp/P)) if P>0 else np.zeros(len(tp)+1)
    fpr = np.concatenate(([0.0], fp/N)) if N>0 else np.zeros(len(fp)+1)
    recall = np.concatenate(([0.0], tp/P)) if P>0 else np.array([0.0,1.0])
    precision = np.concatenate(([1.0], tp/(tp+fp))) if P>0 and N>0 else np.array([1.0,0.0])
    tpr = np.append(tpr,1.0) if tpr[-1]!=1.0 else tpr
    fpr = np.append(fpr,1.0) if fpr[-1]!=1.0 else fpr
    recall = np.append(recall,1.0) if recall[-1]!=1.0 else recall
    precision = np.append(precision, P/(P+N) if P>0 and N>0 else 1.0) if precision[-1]!=1.0 else precision
    return fpr, tpr, recall, precision

def get_prob_positiva(probs):
    probs = np.array(probs)
    if probs.ndim == 1:
        return probs
    elif probs.ndim == 2 and probs.shape[1] == 2:
        return probs[:, 1]
