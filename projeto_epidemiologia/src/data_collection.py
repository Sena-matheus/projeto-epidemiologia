# src/data_collection.py
import os
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

API_URL = "https://apidadosabertos.saude.gov.br/vigilancia-e-meio-ambiente/mpox"
HEADERS = {"accept": "application/json"}

RAW_DATA_PATH = Path("data/raw/")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

def obter_dados_mpox_api(limit=1000, max_records=5000):
    """Obtém dados de Mpox da API com paginação."""
    all_casos = []
    offset = 0

    print(f"[INFO] Buscando dados da API (limite: {limit}, máximo: {max_records})")

    while True:
        try:
            params = {"limit": limit, "offset": offset}
            response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()

            data = response.json()
            casos_atuais = data.get("mpox", [])

            if not casos_atuais:
                print("[INFO] Fim dos dados disponíveis.")
                break

            all_casos.extend(casos_atuais)
            print(f"[INFO] Registros obtidos até agora: {len(all_casos)}")

            if len(all_casos) >= max_records or len(casos_atuais) < limit:
                break

            offset += limit

        except requests.exceptions.RequestException as e:
            print(f"[ERRO] Requisição falhou: {e}")
            break

    print(f"[INFO] Total de registros obtidos: {len(all_casos)}")
    return all_casos

def salvar_csv(df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RAW_DATA_PATH / f"mpox_data_{timestamp}.csv"
    df.to_csv(path, index=False)
    print(f"[INFO] Dados salvos em {path}")

def main():
    casos = obter_dados_mpox_api()
    if not casos:
        print("[WARN] Nenhum dado carregado.")
        return

    df = pd.json_normalize(casos)
    df = df.dropna(axis=1, how='all')  # Remove colunas totalmente vazias

    # Se houver coluna dataNotificacao, remover registros sem essa data
    if 'dataNotificacao' in df.columns:
        df = df.dropna(subset=['dataNotificacao'])

    salvar_csv(df)

if __name__ == "__main__":
    main()
