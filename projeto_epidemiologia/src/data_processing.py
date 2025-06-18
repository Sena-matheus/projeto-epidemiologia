# src/data_processing.py

import pandas as pd
from datetime import datetime
from pathlib import Path

# Diretórios
RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# Mapeamentos de categorias
SEXO_MAP = {
    1: 'Masculino', 2: 'Feminino', 3: 'Não Binário', 9: 'Ignorado',
    'M': 'Masculino', 'F': 'Feminino', 'I': 'Ignorado'
}

GESTANTE_MAP = {
    1: '1º Trimestre', 2: '2º Trimestre', 3: '3º Trimestre',
    4: 'Idade gestacional ignorada', 5: 'Não gestante',
    6: 'Idade gestacional não se aplica', 9: 'Ignorado'
}

SIM_NAO_MAP = {
    1: 'Sim', 2: 'Não', 9: 'Ignorado',
    'S': 'Sim', 'N': 'Não'
}

EVOLUCAO_MAP = {
    1: 'Cura', 2: 'Óbito por Mpox', 3: 'Óbito por outras causas',
    4: 'Em acompanhamento', 9: 'Ignorado'
}

def carregar_arquivo_csv():
    arquivos = sorted(RAW_PATH.glob("mpox_data_*.csv"), reverse=True)
    if not arquivos:
        print("[ERRO] Nenhum arquivo CSV encontrado em data/raw/")
        return None
    print(f"[INFO] Carregando dados: {arquivos[0]}")
    return pd.read_csv(arquivos[0])

def processar_dados(df):
    print("[INFO] Shape do df original:", df.shape)
    df = df.dropna(how='all', axis=1)

    colunas_data = [col for col in df.columns if col.startswith(('dt_', 'data'))]
    for col in colunas_data:
        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

    if 'dt_notific' in df.columns:
        df['ano_mes'] = df['dt_notific'].dt.to_period('M').astype(str)

    if 'nu_idade_n' in df.columns:
        df['nu_idade_n'] = pd.to_numeric(df['nu_idade_n'], errors='coerce')
        bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 120]
        labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
        df['faixa_idade'] = pd.cut(df['nu_idade_n'], bins=bins, labels=labels, right=True)
        df['faixa_idade'] = df['faixa_idade'].cat.add_categories('Ignorado').fillna('Ignorado')

    mapeamentos = {
        'cs_sexo': SEXO_MAP,
        'cs_gestant': GESTANTE_MAP,
        'evolucao': EVOLUCAO_MAP,
        'ist_ativa': SIM_NAO_MAP,
        'hiv': SIM_NAO_MAP,
        'vacina': SIM_NAO_MAP,
        'estrangeiro': SIM_NAO_MAP,
        'profis_saude': SIM_NAO_MAP
    }

    for col, mapa in mapeamentos.items():
        if col in df.columns:
            df[col] = df[col].astype(str).replace(mapa).fillna('Ignorado')

    colunas_categoricas = ['cs_raca', 'orienta_sexual', 'classi_fin', 'clado', 'sg_uf_not']
    for col in colunas_categoricas:
        if col in df.columns:
            df[col] = df[col].fillna('Ignorado')

    df = df.dropna(subset=['dt_notific'])

    return df

def salvar_dados(df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"mpox_processed_{timestamp}.csv"
    caminho = PROCESSED_PATH / nome_arquivo
    df.to_csv(caminho, index=False)
    print(f"[INFO] Dados processados salvos em: {caminho}")

def main():
    df = carregar_arquivo_csv()
    if df is None:
        return

    df = processar_dados(df)
    salvar_dados(df)

if __name__ == "__main__":
    main()
