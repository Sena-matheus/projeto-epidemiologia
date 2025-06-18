# src/ml_pipeline.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

AGG_PATH = Path("data/processed/aggregated")
MODEL_PATH = Path("data/models")
MODEL_PATH.mkdir(exist_ok=True)

def carregar_dados():
    arquivos = sorted(AGG_PATH.glob("casos_por_estado_mes_*.csv"), reverse=True)
    if not arquivos:
        raise FileNotFoundError("Nenhum arquivo agregado encontrado em data/processed/aggregated")
    df = pd.read_csv(arquivos[0])
    df['ano_mes'] = pd.to_datetime(df['ano_mes'], format='%Y-%m')
    df = df.sort_values(['sg_uf_not', 'ano_mes'])
    return df

def criar_features_lags(df, n_lags=3):
    df = df.copy()
    # Criar colunas de lag para cada estado
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df.groupby('sg_uf_not')['casos'].shift(lag)
    # Feature temporal
    df['mes'] = df['ano_mes'].dt.month
    df['ano'] = df['ano_mes'].dt.year
    # Drop linhas com NA (devido a lags)
    df = df.dropna()
    return df

def preparar_dados(df):
    X = df[['sg_uf_not', 'ano', 'mes'] + [f'lag_{i}' for i in range(1, 4)]].copy()
    y = df['casos']

    # Encoding simples do estado (label encoding manual)
    estados = sorted(X['sg_uf_not'].unique())
    estado_map = {e: i for i, e in enumerate(estados)}
    X['sg_uf_not'] = X['sg_uf_not'].map(estado_map)

    return X, y, estado_map

def treinar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")

    return model

def salvar_modelo(model, estado_map):
    joblib.dump(model, MODEL_PATH / "mpox_casos_rf.joblib")
    joblib.dump(estado_map, MODEL_PATH / "estado_map.joblib")
    print("[INFO] Modelo e mapa de estados salvos em data/models/")

def main():
    df = carregar_dados()
    df = criar_features_lags(df)
    X, y, estado_map = preparar_dados(df)
    model = treinar_modelo(X, y)
    salvar_modelo(model, estado_map)

if __name__ == "__main__":
    main()
