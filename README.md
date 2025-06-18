#Projeto Epidemiologia - Previsão de Casos de Mpox no Brasil
Este projeto tem como objetivo coletar, processar e analisar dados epidemiológicos de Mpox no Brasil para realizar previsões de casos futuros por estado, utilizando técnicas de Machine Learning. Também disponibiliza um dashboard interativo para visualização dos dados históricos, ranking de estados e mapas.

📁 Estrutura do Projeto
bash
Copiar
Editar
projeto_pyspark/
├── data/
│   ├── raw/             # Dados brutos obtidos das APIs
│   ├── processed/       # Dados processados e agregados
│   └── models/          # Modelos treinados salvos
├── src/
│   ├── data_collection.py   # Coleta de dados da API Mpox
│   ├── data_processing.py   # Processamento e limpeza dos dados
│   ├── ml_pipeline.py       # Treinamento e predição com ML
│   └── dashboard.py         # Dashboard interativo com Dash
├── notebooks/
│   ├── eda.ipynb            # Análise exploratória dos dados
│   └── model_evaluation.ipynb # Avaliação do modelo ML
├── reports/
│   └── analise_preditiva.pdf # Relatório final
└── requirements.txt         # Dependências do projeto
🚀 Como rodar localmente
Pré-requisitos
Python 3.8 ou superior

Pip instalado

Instalar dependências
bash
Copiar
Editar
pip install -r requirements.txt
Executar coleta e processamento de dados
bash
Copiar
Editar
python src/data_collection.py
python src/data_processing.py
Treinar modelo e gerar predição
bash
Copiar
Editar
python src/ml_pipeline.py
Rodar dashboard interativo
bash
Copiar
Editar
python src/dashboard.py
O dashboard estará disponível em: http://127.0.0.1:8050

🧰 Tecnologias utilizadas
Python (pandas, scikit-learn, joblib)

Dash / Plotly para visualização interativa

Requests para coleta de dados via API

Machine Learning: Random Forest Regressor para predição de casos

Controle de versão via Git/GitHub

📊 Funcionalidades do Dashboard
Visualização da série histórica de casos por estado

Previsão de casos para os próximos 3 meses

Ranking anual de estados por número de casos

Mapa interativo filtrável por estados e ano para análise geográfica

📈 Modelo de Machine Learning
O modelo é baseado em Random Forest Regressor e utiliza lags temporais (dados dos últimos 3 meses) para prever os casos futuros de Mpox por estado e mês.

🤝 Contribuição
Contribuições são bem-vindas! Para sugerir melhorias ou reportar problemas, abra uma issue ou envie um pull request.

📄 Licença
Este projeto está sob a licença MIT — veja o arquivo LICENSE para detalhes.

