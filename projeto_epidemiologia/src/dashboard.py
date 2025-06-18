import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

# Caminhos
AGG_PATH = Path("data/processed/aggregated")
MODEL_PATH = Path("data/models")

# Carregar dados agregados
def carregar_dados():
    arquivo = sorted(AGG_PATH.glob("casos_por_estado_mes_*.csv"), reverse=True)[0]
    df = pd.read_csv(arquivo)
    df['ano_mes'] = pd.to_datetime(df['ano_mes'], format='%Y-%m')
    return df

# Criar features de previsão
def criar_features_para_prever(df, estado_map, n_lags=3):
    previsoes = []

    for estado in df['sg_uf_not'].unique():
        dados_estado = df[df['sg_uf_not'] == estado].sort_values("ano_mes")
        if len(dados_estado) < n_lags:
            continue  # Não há dados suficientes

        # Extrair últimos lags
        ultimos = dados_estado.tail(n_lags).copy()
        lags = list(ultimos['casos'].values)[-n_lags:]

        # Definir data da próxima previsão
        ultima_data = dados_estado['ano_mes'].max()
        proxima_data = (ultima_data + pd.DateOffset(months=1)).replace(day=1)
        mes = proxima_data.month
        ano = proxima_data.year

        # Montar input
        entrada = {
            'sg_uf_not': estado_map.get(estado, -1),
            'ano': ano,
            'mes': mes,
            **{f'lag_{i+1}': lags[-(i+1)] for i in range(n_lags)}
        }
        previsoes.append({
            'sg_uf_not': estado,
            'ano_mes': proxima_data,
            'casos_previstos': modelo.predict(pd.DataFrame([entrada]))[0]
        })

    return pd.DataFrame(previsoes)

# Carregar modelo e mapa
modelo = joblib.load(MODEL_PATH / "mpox_casos_rf.joblib")
estado_map = joblib.load(MODEL_PATH / "estado_map.joblib")

# Reverter o encoding
estado_map_inv = {v: k for k, v in estado_map.items()}

# Dados históricos e previsão
df = carregar_dados()
df_prev = criar_features_para_prever(df, estado_map)

# App
app = dash.Dash(__name__)
app.title = "Previsão de Casos de Mpox"

# Layout
app.layout = html.Div([
    html.H1("📈 Previsão de Casos de Mpox no Brasil", style={"textAlign": "center"}),

    html.Label("Escolha o Estado:", style={"marginTop": "20px"}),
    dcc.Dropdown(
        options=[{"label": uf, "value": uf} for uf in sorted(df["sg_uf_not"].unique())],
        value="SP",
        id="estado-dropdown"
    ),

    dcc.Graph(id="grafico-casos")
])

# Callback
@app.callback(
    Output("grafico-casos", "figure"),
    Input("estado-dropdown", "value")
)
def atualizar_grafico(estado):
    df_estado = df[df["sg_uf_not"] == estado].copy()
    df_prev_estado = df_prev[df_prev["sg_uf_not"] == estado].copy()

    # Unir previsão com dados reais
    df_prev_estado = df_prev_estado.rename(columns={"casos_previstos": "casos"})
    df_prev_estado["tipo"] = "Previsto"
    df_estado["tipo"] = "Histórico"

    df_total = pd.concat([df_estado, df_prev_estado], ignore_index=True)

    fig = px.line(df_total, x="ano_mes", y="casos", color="tipo",
                  markers=True, title=f"Série Temporal - {estado}",
                  labels={"ano_mes": "Ano/Mês", "casos": "Casos de Mpox", "tipo": "Origem"})

    fig.update_layout(transition_duration=500)
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
