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
    df['ano'] = df['ano_mes'].dt.year
    df['mes'] = df['ano_mes'].dt.month
    return df

# Criar features para previsão
def criar_features_para_prever(df, estado_map, modelo, n_lags=3, meses_prev=3):
    previsoes = []
    for estado in df['sg_uf_not'].unique():
        dados_estado = df[df['sg_uf_not'] == estado].sort_values("ano_mes")
        if len(dados_estado) < n_lags:
            continue
        lags = list(dados_estado['casos'].values)[-n_lags:]
        ultima_data = dados_estado['ano_mes'].max()
        for i in range(meses_prev):
            proxima_data = (ultima_data + pd.DateOffset(months=1)).replace(day=1)
            entrada = {
                'sg_uf_not': estado_map.get(estado, -1),
                'ano': proxima_data.year,
                'mes': proxima_data.month,
                **{f'lag_{j+1}': lags[-(j+1)] for j in range(n_lags)}
            }
            pred = modelo.predict(pd.DataFrame([entrada]))[0]
            previsoes.append({
                'sg_uf_not': estado,
                'ano_mes': proxima_data,
                'casos': pred,
                'tipo': 'Previsto'
            })
            lags.append(pred)
            lags = lags[-n_lags:]
            ultima_data = proxima_data
    return pd.DataFrame(previsoes)

# Carregar modelo
modelo = joblib.load(MODEL_PATH / "mpox_casos_rf.joblib")
estado_map = joblib.load(MODEL_PATH / "estado_map.joblib")

# Dados históricos + previsão
df = carregar_dados()
df_prev = criar_features_para_prever(df, estado_map, modelo, meses_prev=3)
df_prev['ano'] = df_prev['ano_mes'].dt.year
df_prev['mes'] = df_prev['ano_mes'].dt.month
df_hist = df.copy()
df_hist['tipo'] = 'Histórico'
df_total = pd.concat([df_hist, df_prev])

# Inicializar app
app = dash.Dash(__name__, title="Dashboard Mpox Brasil", suppress_callback_exceptions=True)

# Layout
app.layout = html.Div([
    html.H1("📊 Dashboard de Casos de Mpox no Brasil", style={"textAlign": "center"}),

    dcc.Tabs([
        dcc.Tab(label="📈 Evolução Temporal + Previsão", children=[
            dcc.Dropdown(
                id="estado-predicao",
                options=[{"label": uf, "value": uf} for uf in sorted(df_total["sg_uf_not"].unique())],
                value="SP"
            ),
            dcc.Graph(id="grafico-casos")
        ]),

        dcc.Tab(label="📊 Ranking por Estado", children=[
            dcc.Dropdown(
                id="ano-ranking",
                options=[{"label": str(ano), "value": ano} for ano in sorted(df["ano"].unique())],
                value=df["ano"].max()
            ),
            dcc.Graph(id="grafico-ranking")
        ]),

        dcc.Tab(label="🗺️ Mapa de Casos", children=[
            html.Div([
                dcc.Dropdown(
                    id="filtro-ano-mapa",
                    options=[{"label": str(ano), "value": ano} for ano in sorted(df["ano"].unique())],
                    value=df["ano"].max(),
                    style={"width": "200px"}
                ),
                dcc.Dropdown(
                    id="filtro-estados-mapa",
                    options=[{"label": uf, "value": uf} for uf in sorted(df["sg_uf_not"].unique())],
                    value=sorted(df["sg_uf_not"].unique()),
                    multi=True,
                    style={"width": "400px"}
                )
            ], style={"display": "flex", "gap": "20px", "padding": "10px"}),

            dcc.Graph(id="grafico-mapa")
        ])
    ])
])

# Callbacks
@app.callback(
    Output("grafico-casos", "figure"),
    Input("estado-predicao", "value")
)
def atualizar_grafico_predicao(estado):
    df_estado = df_total[df_total["sg_uf_not"] == estado]
    fig = px.line(df_estado, x="ano_mes", y="casos", color="tipo", markers=True,
                  title=f"Série Temporal com Previsão - {estado}",
                  labels={"ano_mes": "Data", "casos": "Casos", "tipo": "Tipo"})
    return fig

@app.callback(
    Output("grafico-ranking", "figure"),
    Input("ano-ranking", "value")
)
def atualizar_ranking(ano):
    df_ano = df[df["ano"] == ano]
    df_ranking = df_ano.groupby("sg_uf_not")["casos"].sum().reset_index().sort_values("casos", ascending=False)
    fig = px.bar(df_ranking, x="sg_uf_not", y="casos", title=f"Ranking de Casos - {ano}")
    return fig

@app.callback(
    Output("grafico-mapa", "figure"),
    [Input("filtro-ano-mapa", "value"),
     Input("filtro-estados-mapa", "value")]
)
def atualizar_mapa(ano, estados):
    df_mapa = df[(df["ano"] == ano) & (df["sg_uf_not"].isin(estados))]
    df_mapa_agg = df_mapa.groupby("sg_uf_not")["casos"].sum().reset_index()
    fig = px.choropleth(
        df_mapa_agg,
        geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson",
        locations="sg_uf_not",
        featureidkey="properties.sigla",
        color="casos",
        color_continuous_scale="Reds",
        title=f"Mapa de Casos de Mpox - {ano}",
        scope="south america"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    return fig

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)

