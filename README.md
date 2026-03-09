<div align="center">

# Cassandra Time Series ML Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Cassandra](https://img.shields.io/badge/Cassandra-4.1-1287B1?style=for-the-badge&logo=apachecassandra&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-9B59B6?style=for-the-badge)

Pipeline de machine learning para previsao de series temporais com armazenamento distribuido, engenharia de features automatizada e reconciliacao hierarquica de previsoes.

[Portugues](#portugues) | [English](#english)

</div>

---

# Portugues

## Visao Geral

Este projeto e um pipeline completo de machine learning voltado para **previsao de series temporais em escala**. Ele combina o Apache Cassandra como camada de armazenamento distribuido com modelos de forecasting como Prophet e LightGBM, finalizando com reconciliacao hierarquica para garantir que previsoes em diferentes niveis de agregacao sejam coerentes entre si.

A ideia central e simples: dados brutos entram pelo pipeline, passam por uma etapa rica de engenharia de features (lags, estatisticas moveis, features de calendario, termos de Fourier), sao consumidos por multiplos modelos treinados com validacao cruzada temporal, e saem como previsoes confiáveis e reconciliadas.

O pipeline roda tanto conectado a um cluster Cassandra real quanto em modo local de simulacao -- basta executar `python main.py` para ver tudo funcionando sem dependencias externas.

## Arquitetura

```mermaid
flowchart LR
    A[Ingestao de Dados] --> B[Cassandra Storage]
    B --> C[Engenharia de Features]
    C --> D[Treinamento de Modelos]
    D --> E[Prophet]
    D --> F[LightGBM]
    E --> G[Ensemble]
    F --> G
    G --> H[Reconciliacao Hierarquica]
    H --> I[Previsao Final]

    style A fill:#4A90D9,color:#fff
    style B fill:#1A1A2E,color:#fff
    style C fill:#16213E,color:#fff
    style D fill:#0F3460,color:#fff
    style E fill:#533483,color:#fff
    style F fill:#533483,color:#fff
    style G fill:#E94560,color:#fff
    style H fill:#F39C12,color:#fff
    style I fill:#27AE60,color:#fff
```

## Funcionalidades

- **Armazenamento distribuido** com Apache Cassandra (fallback local para desenvolvimento)
- **Engenharia de features automatizada**: lags, rolling stats, calendario, Fourier
- **Multiplos modelos**: Prophet (com fallback statsmodels), LightGBM, Ensemble configuravel
- **Validacao cruzada temporal** com janela expansiva, respeitando a ordem cronologica
- **Metricas completas**: RMSE, MAE, MAPE, SMAPE, R-squared, Acuracia Direcional
- **Reconciliacao hierarquica**: bottom-up, top-down e MinT (Minimum Trace)
- **Docker Compose** com Cassandra + aplicacao prontos para deploy
- **CI/CD** via GitHub Actions com lint, testes e build

## Aplicacoes na Industria

Este tipo de pipeline resolve problemas reais em diversos setores:

- **Varejo**: previsao de demanda por loja, categoria e SKU, garantindo que a soma das previsoes das lojas bata com a previsao regional
- **Energia**: estimativa de consumo eletrico em diferentes granularidades temporais, desde horaria ate mensal
- **Financeiro**: modelagem de series de receita, custos e fluxo de caixa com multiplos modelos competindo
- **IoT e Sensores**: processamento de dados de telemetria em larga escala armazenados no Cassandra
- **Supply Chain**: otimizacao de estoques com previsoes reconciliadas entre centros de distribuicao

## Como Executar

### Modo local (sem Cassandra)

```bash
# Instalar dependencias
pip install -r requirements.txt

# Rodar a demo completa
python main.py

# Rodar os testes
python -m pytest tests/ -v
```

### Com Docker

```bash
# Subir Cassandra + pipeline
docker compose -f docker/docker-compose.yml up -d

# Ou usar o Makefile
make docker-up
```

## Estrutura do Projeto

```
cassandra-timeseries-ml-pipeline/
├── main.py                          # Demo completa do pipeline
├── requirements.txt
├── Makefile
├── config/
│   └── pipeline_config.yaml         # Configuracao do pipeline
├── docker/
│   ├── Dockerfile                   # Multi-stage build
│   └── docker-compose.yml           # Cassandra + app
├── src/
│   ├── config/
│   │   └── settings.py              # Dataclasses de configuracao
│   ├── utils/
│   │   └── logger.py                # Logging estruturado (JSON)
│   ├── storage/
│   │   └── cassandra_client.py      # Cliente Cassandra com fallback
│   ├── features/
│   │   └── feature_generator.py     # Engenharia de features
│   ├── models/
│   │   ├── base_model.py            # ABC para modelos
│   │   ├── prophet_model.py         # Prophet / Holt-Winters
│   │   ├── lightgbm_model.py        # LightGBM / sklearn GBR
│   │   └── ensemble_model.py        # Combinacao de modelos
│   ├── training/
│   │   └── trainer.py               # Orquestrador de treinamento
│   ├── evaluation/
│   │   └── evaluator.py             # Metricas e comparacao
│   └── reconciliation/
│       └── reconciler.py            # Reconciliacao hierarquica
├── tests/
│   ├── conftest.py                  # Fixtures compartilhados
│   └── unit/
│       ├── test_features.py
│       ├── test_models.py
│       └── test_evaluation.py
└── .github/
    └── workflows/
        └── ci.yml                   # GitHub Actions CI
```

## Exemplo de Saida

```
======================================================================
   Cassandra Time Series ML Pipeline -- Demo
======================================================================

[1/7] Generating synthetic retail sales data ...
      Generated 730 daily observations
      Date range: 2022-01-01 to 2023-12-31

[4/7] Training models ...

======================================================================
  Retail Sales Forecast -- Model Comparison
======================================================================

Final Test Metrics (sorted by RMSE):
----------------------------------------------------------------------
      model     rmse      mae    mape   smape      r2  directional_accuracy
1  LightGBM   52.31    38.72    3.12    3.08    0.92             68.42
2  Ensemble   58.44    43.18    3.48    3.41    0.90             65.79
3   Prophet   71.56    55.23    4.51    4.38    0.85             61.84

Best model: LightGBM (RMSE = 52.31)
======================================================================
```

## Tecnologias

| Componente | Tecnologia |
|:--|:--|
| Armazenamento | Apache Cassandra 4.1 |
| Processamento | PySpark, pandas |
| Modelos | Prophet, LightGBM, statsmodels |
| Features | numpy, scikit-learn |
| Infra | Docker, GitHub Actions |
| Linguagem | Python 3.11+ |

---

# English

## Overview

This project is a full-featured machine learning pipeline built for **time series forecasting at scale**. It pairs Apache Cassandra as a distributed storage backend with forecasting models like Prophet and LightGBM, finishing with hierarchical reconciliation to make sure predictions at every aggregation level add up consistently.

The core workflow is straightforward: raw time series data flows into the pipeline, goes through rich feature engineering (lags, rolling statistics, calendar features, Fourier terms), gets consumed by multiple models trained with proper time-aware cross-validation, and comes out the other side as reliable, reconciled forecasts.

The whole thing runs either against a real Cassandra cluster or in a local simulation mode -- just run `python main.py` and everything works out of the box with no external infrastructure.

## Architecture

```mermaid
flowchart LR
    A[Data Ingestion] --> B[Cassandra Storage]
    B --> C[Feature Engineering]
    C --> D[Model Training]
    D --> E[Prophet]
    D --> F[LightGBM]
    E --> G[Ensemble]
    F --> G
    G --> H[Hierarchical Reconciliation]
    H --> I[Forecast Output]

    style A fill:#4A90D9,color:#fff
    style B fill:#1A1A2E,color:#fff
    style C fill:#16213E,color:#fff
    style D fill:#0F3460,color:#fff
    style E fill:#533483,color:#fff
    style F fill:#533483,color:#fff
    style G fill:#E94560,color:#fff
    style H fill:#F39C12,color:#fff
    style I fill:#27AE60,color:#fff
```

## Features

- **Distributed storage** with Apache Cassandra (local fallback for development)
- **Automated feature engineering**: lags, rolling stats, calendar, Fourier
- **Multiple models**: Prophet (statsmodels fallback), LightGBM, configurable Ensemble
- **Time-series cross-validation** with expanding windows that respect chronological order
- **Comprehensive metrics**: RMSE, MAE, MAPE, SMAPE, R-squared, Directional Accuracy
- **Hierarchical reconciliation**: bottom-up, top-down, and MinT (Minimum Trace)
- **Docker Compose** with Cassandra + application ready to deploy
- **CI/CD** via GitHub Actions with linting, testing, and Docker build

## Industry Applications

This kind of pipeline addresses real problems across many domains:

- **Retail**: demand forecasting at store, category, and SKU level, ensuring store-level forecasts roll up to regional totals
- **Energy**: electricity consumption estimation across time granularities, from hourly to monthly
- **Finance**: modelling revenue, cost, and cash flow series with multiple competing models
- **IoT & Sensors**: large-scale telemetry data processing stored in Cassandra
- **Supply Chain**: inventory optimization with reconciled forecasts across distribution centers

## Getting Started

### Local mode (no Cassandra needed)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full demo
python main.py

# Run the test suite
python -m pytest tests/ -v
```

### With Docker

```bash
# Bring up Cassandra + pipeline
docker compose -f docker/docker-compose.yml up -d

# Or use the Makefile
make docker-up
```

## Project Structure

```
cassandra-timeseries-ml-pipeline/
├── main.py                          # Full pipeline demo
├── requirements.txt
├── Makefile
├── config/
│   └── pipeline_config.yaml         # Pipeline configuration
├── docker/
│   ├── Dockerfile                   # Multi-stage build
│   └── docker-compose.yml           # Cassandra + app
├── src/
│   ├── config/
│   │   └── settings.py              # Configuration dataclasses
│   ├── utils/
│   │   └── logger.py                # Structured JSON logging
│   ├── storage/
│   │   └── cassandra_client.py      # Cassandra client with fallback
│   ├── features/
│   │   └── feature_generator.py     # Feature engineering
│   ├── models/
│   │   ├── base_model.py            # ABC for models
│   │   ├── prophet_model.py         # Prophet / Holt-Winters
│   │   ├── lightgbm_model.py        # LightGBM / sklearn GBR
│   │   └── ensemble_model.py        # Model combination
│   ├── training/
│   │   └── trainer.py               # Training orchestrator
│   ├── evaluation/
│   │   └── evaluator.py             # Metrics and comparison
│   └── reconciliation/
│       └── reconciler.py            # Hierarchical reconciliation
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   └── unit/
│       ├── test_features.py
│       ├── test_models.py
│       └── test_evaluation.py
└── .github/
    └── workflows/
        └── ci.yml                   # GitHub Actions CI
```

## Example Output

```
======================================================================
   Cassandra Time Series ML Pipeline -- Demo
======================================================================

[1/7] Generating synthetic retail sales data ...
      Generated 730 daily observations
      Date range: 2022-01-01 to 2023-12-31

[4/7] Training models ...

======================================================================
  Retail Sales Forecast -- Model Comparison
======================================================================

Final Test Metrics (sorted by RMSE):
----------------------------------------------------------------------
      model     rmse      mae    mape   smape      r2  directional_accuracy
1  LightGBM   52.31    38.72    3.12    3.08    0.92             68.42
2  Ensemble   58.44    43.18    3.48    3.41    0.90             65.79
3   Prophet   71.56    55.23    4.51    4.38    0.85             61.84

Best model: LightGBM (RMSE = 52.31)
======================================================================
```

## Technologies

| Component | Technology |
|:--|:--|
| Storage | Apache Cassandra 4.1 |
| Processing | PySpark, pandas |
| Models | Prophet, LightGBM, statsmodels |
| Features | numpy, scikit-learn |
| Infra | Docker, GitHub Actions |
| Language | Python 3.11+ |

---

<div align="center">

Developed by **Gabriel Demetrios Lafis**

</div>
