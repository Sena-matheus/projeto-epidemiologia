import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import dump
import numpy as np

DATA_PROCESSED_PATH = Path("data/processed/")
MODEL_PATH = Path("data/models/")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

def carregar_dados(nome_arquivo: str) -> pd.DataFrame:
    path = DATA_PROCESSED_PATH / nome_arquivo
    df = pd.read_csv(path)
    return df

def preparar_dataset_agrupado(df: pd.DataFrame):
    # Agrupando casos por ano_mes e sg_uf_not
    agrup = df.groupby(['ano_mes', 'sg_uf_not']).size().reset_index(name='casos')
    
    # Criar features categóricas - aqui só ano_mes e sg_uf_not, poderia adicionar outras
    X = agrup[['ano_mes', 'sg_uf_not']]
    y = agrup['casos']
    
    # Codificar categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['ano_mes', 'sg_uf_not'])
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    return train_test_split(X_processed, y, test_size=0.2, random_state=42), preprocessor

def treinar_modelo(X_train, y_train):
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def avaliar_modelo(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse:.2f}')
    print(f'R²: {r2:.2f}')

def salvar_modelo(model, preprocessor, nome_arquivo="rf_regressor_casos.joblib"):
    dump({'model': model, 'preprocessor': preprocessor}, MODEL_PATH / nome_arquivo)
    print(f'Modelo salvo em {MODEL_PATH / nome_arquivo}')

def main():
    nome_arquivo = "mpox_processed_20250618_161802.csv"
    df = carregar_dados(nome_arquivo)
    
    (X_train, X_test, y_train, y_test), preprocessor = preparar_dataset_agrupado(df)
    
    print(f"Dados preparados: treino={X_train.shape[0]}, teste={X_test.shape[0]}")
    
    model = treinar_modelo(X_train, y_train)
    
    avaliar_modelo(model, X_test, y_test)
    
    salvar_modelo(model, preprocessor)

if __name__ == "__main__":
    main()
