def normalizar_dataset(df, excluir_cols=None):
    if excluir_cols is None:
        excluir_cols = []
    df_normalizado = df.copy()
    num_cols = [c for c in df_normalizado.select_dtypes(include='number').columns if c not in excluir_cols]
    for col in num_cols:
        mean = df_normalizado[col].mean()
        std = df_normalizado[col].std()
        if std != 0:
            df_normalizado[col] = (df_normalizado[col] - mean) / std
    return df_normalizado
