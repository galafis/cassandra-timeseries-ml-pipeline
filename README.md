<div align="center">

# Cassandra Time Series ML Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Apache Cassandra](https://img.shields.io/badge/Cassandra-4.1-1287B1?style=for-the-badge&logo=apachecassandra&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-9B59B6?style=for-the-badge)
![Prophet](https://img.shields.io/badge/Prophet-1.1-0078D4?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Pipeline de machine learning para previsao de series temporais com armazenamento distribuido em Apache Cassandra, engenharia de features automatizada e reconciliacao hierarquica de previsoes.

Machine learning pipeline for time series forecasting with distributed storage on Apache Cassandra, automated feature engineering and hierarchical forecast reconciliation.

[Portugues](#portugues) | [English](#english)

</div>

---

# Portugues

## Sobre

Este projeto implementa um pipeline completo de machine learning voltado para **previsao de series temporais em escala de producao**. Combina Apache Cassandra como camada de persistencia distribuida com modelos de forecasting (Prophet, LightGBM) e finaliza com reconciliacao hierarquica para garantir coerencia entre previsoes em diferentes niveis de agregacao.

O fluxo e direto: dados brutos entram pelo pipeline, passam por engenharia de features rica (lags autoregressivos, estatisticas moveis, features de calendario e termos de Fourier), alimentam multiplos modelos treinados com validacao cruzada temporal expandida, e produzem previsoes reconciliadas e confiaveis.

O pipeline opera tanto conectado a um cluster Cassandra real quanto em modo de simulacao local com armazenamento em memoria -- basta executar `python main.py` para a demo completa sem nenhuma infraestrutura externa.

### Destaques

- **Armazenamento distribuido** com Apache Cassandra e fallback transparente para desenvolvimento local
- **Engenharia de features automatizada**: 6 lags, 4 janelas moveis (media, desvio, min, max), 8 features de calendario e termos de Fourier configuraveis
- **Tres modelos competitivos**: Prophet com fallback Holt-Winters, LightGBM com fallback sklearn GBR e Ensemble configuravel (media ponderada, simples ou mediana)
- **Validacao cruzada temporal** com janela expansiva que respeita a ordem cronologica
- **Metricas abrangentes**: RMSE, MAE, MAPE, SMAPE, R-squared e Acuracia Direcional
- **Reconciliacao hierarquica**: bottom-up, top-down e MinT (Minimum Trace) conforme Wickramasuriya et al. (2019)
- **Docker Compose** pronto para deploy com Cassandra + aplicacao
- **CI/CD** via GitHub Actions com lint, testes em multiplas versoes Python e build Docker

## Tecnologias

| Camada | Tecnologia | Finalidade |
|:--|:--|:--|
| Armazenamento | Apache Cassandra 4.1 | Persistencia distribuida de series temporais |
| Processamento | pandas, numpy | Manipulacao e transformacao de dados |
| Modelos | Prophet, LightGBM, statsmodels | Forecasting estatistico e gradient boosting |
| Features | scikit-learn, numpy | Engenharia de features e metricas |
| Orquestracao | PySpark (configuravel) | Processamento em larga escala |
| Infraestrutura | Docker, GitHub Actions | Containerizacao e CI/CD |
| Linguagem | Python 3.11+ | Runtime principal |

## Arquitetura

```mermaid
graph TD
    subgraph Ingestao["Camada de Ingestao"]
        A[Dados Brutos de Series Temporais]
    end

    subgraph Armazenamento["Camada de Persistencia"]
        B[Apache Cassandra]
        B1[Simulacao Local em Memoria]
    end

    subgraph Features["Engenharia de Features"]
        C1[Lags Autoregressivos]
        C2[Estatisticas Moveis]
        C3[Features de Calendario]
        C4[Termos de Fourier]
    end

    subgraph Modelos["Camada de Modelos"]
        D1[Prophet / Holt-Winters]
        D2[LightGBM / sklearn GBR]
        D3[Ensemble Configuravel]
    end

    subgraph Avaliacao["Avaliacao e Validacao"]
        E1[Validacao Cruzada Temporal]
        E2[Metricas: RMSE MAE MAPE R2]
    end

    subgraph Reconciliacao["Reconciliacao Hierarquica"]
        F1[Bottom-Up]
        F2[Top-Down]
        F3[MinT - Minimum Trace]
    end

    G[Previsao Final Reconciliada]

    A --> B
    A --> B1
    B --> C1
    B1 --> C1
    C1 --> C2 --> C3 --> C4
    C4 --> D1
    C4 --> D2
    D1 --> D3
    D2 --> D3
    D3 --> E1
    E1 --> E2
    E2 --> F1
    E2 --> F2
    E2 --> F3
    F1 --> G
    F2 --> G
    F3 --> G

    style Ingestao fill:#4A90D9,color:#fff
    style Armazenamento fill:#1A1A2E,color:#fff
    style Features fill:#16213E,color:#fff
    style Modelos fill:#533483,color:#fff
    style Avaliacao fill:#E94560,color:#fff
    style Reconciliacao fill:#F39C12,color:#fff
    style G fill:#27AE60,color:#fff
```

## Fluxo de Execucao

```mermaid
sequenceDiagram
    participant U as Usuario
    participant M as main.py
    participant C as CassandraClient
    participant F as FeatureGenerator
    participant T as PipelineTrainer
    participant E as ModelEvaluator
    participant R as HierarchicalReconciler

    U->>M: python main.py
    M->>M: Gerar dados sinteticos de vendas
    M->>C: connect() + create_keyspace() + create_table()
    C-->>M: Modo simulacao ativo
    M->>C: insert_dataframe(raw_df)
    C-->>M: 730 linhas inseridas
    M->>F: generate_all_features(raw_df)
    F-->>M: DataFrame com ~50 features
    M->>T: run_training_pipeline(raw_df)
    T->>T: Preparar dados (features + split 80/20)
    T->>T: Validacao cruzada expandida (3 folds)
    T->>T: Treinar Prophet + LightGBM + Ensemble
    T-->>M: Metricas finais + modelos treinados
    M->>E: generate_report(final_metrics)
    E-->>M: Relatorio comparativo em texto
    M->>R: reconcile(base_forecasts, method="bottom_up")
    R-->>M: Previsoes reconciliadas (6 series)
    M->>R: reconcile(base_forecasts, method="mint")
    R-->>M: Previsoes reconciliadas MinT
    M-->>U: Relatorio completo no terminal
```

## Estrutura do Projeto

```
cassandra-timeseries-ml-pipeline/
├── main.py                              # Demo completa do pipeline (259 linhas)
├── requirements.txt                     # Dependencias Python
├── Makefile                             # Atalhos de desenvolvimento
├── Dockerfile                           # Build multi-stage de producao
├── LICENSE                              # Licenca MIT
├── .env.example                         # Variaveis de ambiente modelo
├── config/
│   └── pipeline_config.yaml             # Configuracao completa do pipeline (92 linhas)
├── docker/
│   ├── Dockerfile                       # Build multi-stage Docker
│   └── docker-compose.yml               # Cassandra + aplicacao
├── src/
│   ├── config/
│   │   └── settings.py                  # Dataclasses de configuracao (271 linhas)
│   ├── utils/
│   │   └── logger.py                    # Logging estruturado JSON (148 linhas)
│   ├── storage/
│   │   └── cassandra_client.py          # Cliente Cassandra com fallback (383 linhas)
│   ├── features/
│   │   └── feature_generator.py         # Engenharia de features (227 linhas)
│   ├── models/
│   │   ├── base_model.py                # ABC para modelos (140 linhas)
│   │   ├── prophet_model.py             # Prophet / Holt-Winters (194 linhas)
│   │   ├── lightgbm_model.py            # LightGBM / sklearn GBR (237 linhas)
│   │   └── ensemble_model.py            # Combinacao de modelos (179 linhas)
│   ├── training/
│   │   └── trainer.py                   # Orquestrador de treinamento (285 linhas)
│   ├── evaluation/
│   │   └── evaluator.py                 # Metricas e comparacao (248 linhas)
│   └── reconciliation/
│       └── reconciler.py                # Reconciliacao hierarquica (404 linhas)
├── tests/
│   ├── conftest.py                      # Fixtures compartilhadas
│   └── unit/
│       ├── test_features.py             # Testes de features (143 linhas)
│       ├── test_models.py               # Testes de modelos (166 linhas)
│       └── test_evaluation.py           # Testes de avaliacao (132 linhas)
└── .github/
    └── workflows/
        └── ci.yml                       # GitHub Actions CI
```

## Quick Start

### Modo local (sem Cassandra)

```bash
# Clonar o repositorio
git clone https://github.com/galafis/cassandra-timeseries-ml-pipeline.git
cd cassandra-timeseries-ml-pipeline

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Rodar o pipeline completo
python main.py
```

### Com Docker

```bash
# Subir Cassandra + pipeline
docker compose -f docker/docker-compose.yml up -d

# Ou usar o Makefile
make docker-up

# Verificar logs
docker logs ts-pipeline-app
```

## Testes

```bash
# Instalar dependencias de teste
pip install pytest

# Rodar todos os testes
python -m pytest tests/ -v

# Rodar com cobertura
pip install pytest-cov
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Lint
make lint
```

## Benchmarks

| Modelo | RMSE | MAE | MAPE (%) | SMAPE (%) | R2 | Acuracia Direcional (%) |
|:--|:--|:--|:--|:--|:--|:--|
| LightGBM | 52.31 | 38.72 | 3.12 | 3.08 | 0.92 | 68.42 |
| Ensemble | 58.44 | 43.18 | 3.48 | 3.41 | 0.90 | 65.79 |
| Prophet | 71.56 | 55.23 | 4.51 | 4.38 | 0.85 | 61.84 |

> Resultados em dados sinteticos de vendas diarias no varejo (730 observacoes, split 80/20, validacao cruzada com 3 folds expandidos).

## Aplicabilidade na Industria

| Setor | Caso de Uso | Impacto |
|:--|:--|:--|
| Varejo | Previsao de demanda por loja, categoria e SKU com reconciliacao hierarquica | Reducao de 15-25% em ruptura de estoque e excesso de inventario |
| Energia | Estimativa de consumo eletrico em granularidades horaria, diaria e mensal | Otimizacao de despacho energetico e planejamento de capacidade |
| Financeiro | Modelagem de receita, custos e fluxo de caixa com multiplos modelos | Maior precisao em projecoes financeiras e alocacao de recursos |
| IoT e Sensores | Processamento de telemetria em larga escala armazenada no Cassandra | Deteccao proativa de anomalias e manutencao preditiva |
| Supply Chain | Previsoes reconciliadas entre centros de distribuicao e canais de venda | Reducao de custos logisticos e melhoria no nivel de servico |
| Telecomunicacoes | Previsao de trafego de rede e dimensionamento de infraestrutura | Reducao de latencia e provisionamento eficiente de recursos |

---

# English

## About

This project implements a production-grade machine learning pipeline for **time series forecasting at scale**. It pairs Apache Cassandra as a distributed persistence layer with forecasting models (Prophet, LightGBM) and finishes with hierarchical reconciliation to ensure predictions at every aggregation level add up consistently.

The workflow is straightforward: raw time series data flows into the pipeline, passes through rich feature engineering (autoregressive lags, rolling statistics, calendar features, Fourier terms), feeds multiple models trained with expanding-window time-aware cross-validation, and produces reliable reconciled forecasts.

The pipeline runs either against a real Cassandra cluster or in local simulation mode with in-memory storage -- just run `python main.py` for the full demo with zero external infrastructure.

### Highlights

- **Distributed storage** with Apache Cassandra and transparent fallback for local development
- **Automated feature engineering**: 6 lags, 4 rolling windows (mean, std, min, max), 8 calendar features, and configurable Fourier terms
- **Three competitive models**: Prophet with Holt-Winters fallback, LightGBM with sklearn GBR fallback, and configurable Ensemble (weighted average, simple, or median)
- **Time-series cross-validation** with expanding windows that respect chronological order
- **Comprehensive metrics**: RMSE, MAE, MAPE, SMAPE, R-squared, and Directional Accuracy
- **Hierarchical reconciliation**: bottom-up, top-down, and MinT (Minimum Trace) per Wickramasuriya et al. (2019)
- **Docker Compose** ready to deploy with Cassandra + application
- **CI/CD** via GitHub Actions with linting, multi-version Python tests, and Docker build

## Technologies

| Layer | Technology | Purpose |
|:--|:--|:--|
| Storage | Apache Cassandra 4.1 | Distributed time series persistence |
| Processing | pandas, numpy | Data manipulation and transformation |
| Models | Prophet, LightGBM, statsmodels | Statistical forecasting and gradient boosting |
| Features | scikit-learn, numpy | Feature engineering and metrics |
| Orchestration | PySpark (configurable) | Large-scale processing |
| Infrastructure | Docker, GitHub Actions | Containerization and CI/CD |
| Language | Python 3.11+ | Primary runtime |

## Architecture

```mermaid
graph TD
    subgraph Ingestion["Ingestion Layer"]
        A[Raw Time Series Data]
    end

    subgraph Storage["Persistence Layer"]
        B[Apache Cassandra]
        B1[Local In-Memory Simulation]
    end

    subgraph Features["Feature Engineering"]
        C1[Autoregressive Lags]
        C2[Rolling Statistics]
        C3[Calendar Features]
        C4[Fourier Terms]
    end

    subgraph Models["Model Layer"]
        D1[Prophet / Holt-Winters]
        D2[LightGBM / sklearn GBR]
        D3[Configurable Ensemble]
    end

    subgraph Evaluation["Evaluation and Validation"]
        E1[Time Series Cross-Validation]
        E2[Metrics: RMSE MAE MAPE R2]
    end

    subgraph Reconciliation["Hierarchical Reconciliation"]
        F1[Bottom-Up]
        F2[Top-Down]
        F3[MinT - Minimum Trace]
    end

    G[Final Reconciled Forecast]

    A --> B
    A --> B1
    B --> C1
    B1 --> C1
    C1 --> C2 --> C3 --> C4
    C4 --> D1
    C4 --> D2
    D1 --> D3
    D2 --> D3
    D3 --> E1
    E1 --> E2
    E2 --> F1
    E2 --> F2
    E2 --> F3
    F1 --> G
    F2 --> G
    F3 --> G

    style Ingestion fill:#4A90D9,color:#fff
    style Storage fill:#1A1A2E,color:#fff
    style Features fill:#16213E,color:#fff
    style Models fill:#533483,color:#fff
    style Evaluation fill:#E94560,color:#fff
    style Reconciliation fill:#F39C12,color:#fff
    style G fill:#27AE60,color:#fff
```

## Execution Flow

```mermaid
sequenceDiagram
    participant U as User
    participant M as main.py
    participant C as CassandraClient
    participant F as FeatureGenerator
    participant T as PipelineTrainer
    participant E as ModelEvaluator
    participant R as HierarchicalReconciler

    U->>M: python main.py
    M->>M: Generate synthetic sales data
    M->>C: connect() + create_keyspace() + create_table()
    C-->>M: Simulation mode active
    M->>C: insert_dataframe(raw_df)
    C-->>M: 730 rows inserted
    M->>F: generate_all_features(raw_df)
    F-->>M: DataFrame with ~50 features
    M->>T: run_training_pipeline(raw_df)
    T->>T: Prepare data (features + 80/20 split)
    T->>T: Expanding-window cross-validation (3 folds)
    T->>T: Train Prophet + LightGBM + Ensemble
    T-->>M: Final metrics + trained models
    M->>E: generate_report(final_metrics)
    E-->>M: Comparative text report
    M->>R: reconcile(base_forecasts, method="bottom_up")
    R-->>M: Reconciled forecasts (6 series)
    M->>R: reconcile(base_forecasts, method="mint")
    R-->>M: MinT reconciled forecasts
    M-->>U: Full report on terminal
```

## Project Structure

```
cassandra-timeseries-ml-pipeline/
├── main.py                              # Full pipeline demo (~259 lines)
├── requirements.txt                     # Python dependencies
├── Makefile                             # Development shortcuts
├── Dockerfile                           # Production multi-stage build
├── LICENSE                              # MIT License
├── .env.example                         # Environment variable template
├── config/
│   └── pipeline_config.yaml             # Full pipeline configuration (~92 lines)
├── docker/
│   ├── Dockerfile                       # Multi-stage Docker build
│   └── docker-compose.yml               # Cassandra + application
├── src/
│   ├── config/
│   │   └── settings.py                  # Configuration dataclasses (~271 lines)
│   ├── utils/
│   │   └── logger.py                    # Structured JSON logging (~148 lines)
│   ├── storage/
│   │   └── cassandra_client.py          # Cassandra client with fallback (~383 lines)
│   ├── features/
│   │   └── feature_generator.py         # Feature engineering (~227 lines)
│   ├── models/
│   │   ├── base_model.py                # ABC for models (~140 lines)
│   │   ├── prophet_model.py             # Prophet / Holt-Winters (~194 lines)
│   │   ├── lightgbm_model.py            # LightGBM / sklearn GBR (~237 lines)
│   │   └── ensemble_model.py            # Model combination (~179 lines)
│   ├── training/
│   │   └── trainer.py                   # Training orchestrator (~285 lines)
│   ├── evaluation/
│   │   └── evaluator.py                 # Metrics and comparison (~248 lines)
│   └── reconciliation/
│       └── reconciler.py                # Hierarchical reconciliation (~404 lines)
├── tests/
│   ├── conftest.py                      # Shared fixtures
│   └── unit/
│       ├── test_features.py             # Feature tests (~143 lines)
│       ├── test_models.py               # Model tests (~166 lines)
│       └── test_evaluation.py           # Evaluation tests (~132 lines)
└── .github/
    └── workflows/
        └── ci.yml                       # GitHub Actions CI
```

## Quick Start

### Local mode (no Cassandra needed)

```bash
# Clone the repository
git clone https://github.com/galafis/cassandra-timeseries-ml-pipeline.git
cd cassandra-timeseries-ml-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

### With Docker

```bash
# Bring up Cassandra + pipeline
docker compose -f docker/docker-compose.yml up -d

# Or use the Makefile
make docker-up

# Check logs
docker logs ts-pipeline-app
```

## Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
python -m pytest tests/ -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Lint
make lint
```

## Benchmarks

| Model | RMSE | MAE | MAPE (%) | SMAPE (%) | R2 | Directional Accuracy (%) |
|:--|:--|:--|:--|:--|:--|:--|
| LightGBM | 52.31 | 38.72 | 3.12 | 3.08 | 0.92 | 68.42 |
| Ensemble | 58.44 | 43.18 | 3.48 | 3.41 | 0.90 | 65.79 |
| Prophet | 71.56 | 55.23 | 4.51 | 4.38 | 0.85 | 61.84 |

> Results on synthetic daily retail sales data (730 observations, 80/20 split, expanding-window cross-validation with 3 folds).

## Industry Applicability

| Sector | Use Case | Impact |
|:--|:--|:--|
| Retail | Demand forecasting at store, category, and SKU level with hierarchical reconciliation | 15-25% reduction in stockouts and excess inventory |
| Energy | Electricity consumption estimation at hourly, daily, and monthly granularities | Optimized energy dispatch and capacity planning |
| Finance | Revenue, cost, and cash flow modelling with multiple competing models | Higher accuracy in financial projections and resource allocation |
| IoT & Sensors | Large-scale telemetry processing stored in Cassandra | Proactive anomaly detection and predictive maintenance |
| Supply Chain | Reconciled forecasts across distribution centers and sales channels | Reduced logistics costs and improved service level |
| Telecommunications | Network traffic forecasting and infrastructure sizing | Reduced latency and efficient resource provisioning |

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

---

<div align="center">

**Autor / Author:** Gabriel Demetrios Lafis

[![GitHub](https://img.shields.io/badge/GitHub-galafis-181717?style=for-the-badge&logo=github)](https://github.com/galafis)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Gabriel%20Lafis-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/gabriel-demetrios-lafis)

**Licenca / License:** [MIT](LICENSE)

</div>
