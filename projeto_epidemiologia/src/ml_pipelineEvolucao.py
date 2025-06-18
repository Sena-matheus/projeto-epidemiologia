# src/ml_pipeline.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
import logging

logging.basicConfig(level=logging.INFO, format='[INFO] %(message)s')

DATA_PROCESSED_PATH = Path("data/processed/")
MODEL_PATH = Path("data/models/")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

def carregar_dados(nome_arquivo: str) -> pd.DataFrame:
    path = DATA_PROCESSED_PATH / nome_arquivo
    logging.info(f"Carregando dados: {path}")
    df = pd.read_csv(path)
    return df

def preparar_dados(df: pd.DataFrame):
    # Definir features e target
    target = "evolucao"  # classe que queremos prever

    features_cat = [
        "cs_sexo", "cs_gestant", "sg_uf_not", "faixa_idade", "cs_raca", "tratamento_monkeypox", "clado"
    ]
    features_num = [
        "nu_idade_n",
    ]

    # Verifica colunas
    cols_necessarias = features_cat + features_num + [target]
    for col in cols_necessarias:
        if col not in df.columns:
            raise KeyError(f"Coluna {col} não encontrada no DataFrame")

    # Remover linhas com NaN nas colunas escolhidas
    df = df.dropna(subset=cols_necessarias)

    X_cat = df[features_cat].astype(str)  # garantir string para categóricas
    X_num = df[features_num]
    y = df[target].astype(int)  # garante que target é int (classe)

    # Codificar categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat)
        ],
        remainder="passthrough"  # mantém numéricas
    )

    X_processed = preprocessor.fit_transform(pd.concat([X_cat, X_num], axis=1))

    return train_test_split(X_processed, y, test_size=0.2, random_state=42), preprocessor

def treinar_modelo(X_train, y_train):
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def avaliar_modelo(model, X_test, y_test):
    y_pred = model.predict(X_test)
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Acurácia: {acc:.3f}")

def salvar_modelo(model, preprocessor, nome_arquivo="rf_classifier_evolucao.joblib"):
    dump({"model": model, "preprocessor": preprocessor}, MODEL_PATH / nome_arquivo)
    logging.info(f"Modelo salvo em {MODEL_PATH / nome_arquivo}")

def main():
    nome_arquivo = "mpox_processed_20250618_161802.csv" 
    df = carregar_dados(nome_arquivo)
    (X_train, X_test, y_train, y_test), preprocessor = preparar_dados(df)
    logging.info(f"Dados preparados: treino {X_train.shape[0]}, teste {X_test.shape[0]}")

    model = treinar_modelo(X_train, y_train)
    avaliar_modelo(model, X_test, y_test)
    salvar_modelo(model, preprocessor)

if __name__ == "__main__":
    main()
