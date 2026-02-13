# 🥇 Quantitative Gold Analysis: A Scientific Approach to Market Sentiment & LSTMs

> **NLP Sentiment Analysis × LSTM Time-Series Modeling**  
> Talento-Tech Bootcamp 2025-2 · Universidad de Antioquia · Medellín, Colombia

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Post-Mortem](https://img.shields.io/badge/Status-Post--Mortem%20Audited-orange.svg)](#5-the-integrators-audit-known-technical-debt)

### 👉 [Versión en Español aquí](README.es.md)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Team: A Physics-Led Squad](#2-the-team-a-physics-led-squad)
3. [The "Federation" Architecture](#3-the-federation-architecture)
4. [Quick Start](#4-quick-start)
5. [The Integrator's Audit: Known Technical Debt](#5-the-integrators-audit-known-technical-debt)
6. [Lessons Learned](#6-lessons-learned)
7. [Future Roadmap: The Lakehouse Evolution](#7-future-roadmap-the-lakehouse-evolution)
8. [Project Structure](#8-project-structure)

---

## 1. Project Overview

Can the sentiment of financial news **predict** gold price movements?

This project builds an end-to-end pipeline that:
- **Scrapes** ~189,000 Wall Street Journal headlines (2016–2025).
- **Filters** ~18,700 gold-related articles using keyword heuristics.
- **Scores** each headline with **FinBERT** (ProsusAI/finbert), a transformer model fine-tuned for financial sentiment.
- **Detects anomalies** in gold price data using statistical methods.
- **Tests causality** between sentiment and price via Granger Causality.
- **Predicts** daily gold closing prices with LSTM networks — comparing a base model (price-only features) against a sentiment-enhanced model.

**Key finding:** Integrating FinBERT sentiment features into LSTM improved prediction accuracy, though the causal signal is nuanced. The full analysis, including statistical caveats, is documented across 8 notebooks.

---

## 2. The Team: A Physics-Led Squad

### Core Technical Governance

| Role | Member | Responsibility |
|------|--------|---------------|
| **Technical Lead & Integrator** | Pablo Sanchez *(Physics – UdeA)* | Unificación pipeline, 8-notebook architecture, Anomaly Detection, post-project audit |
| **Co-Lead & Statistical Analyst** | Jose Ortiz *(Physics – UdeA)* | Web scraping infrastructure, Granger Causality analysis, data validation |

Pablo and Jose formed the project's **governance pair**. Pablo guaranteed the pipeline ran end-to-end; Jose guaranteed the statistical claims held up under scrutiny. This dual-stewardship model — analogous to a Platform Engineer and a Quant Analyst — ensured that data integrity was protected across the full "front-to-back" engineering flow.

### The Original Squad

| Member | Background | Contribution |
|--------|------------|-------------|
| David Alava | Physics – UdeA | NLP Specialist (FinBERT implementation) |
| Sebastian Agudelo | Physics – UdeA | NLP Specialist (FinBERT implementation) |
| Dayana Henao | Physics – UdeA | ML Engineer (LSTM & Gold Price EDA) |
| Luis Vera | Forestry Engineering | EDA Specialist (News data analysis) |
| Michael Tarazona | Electrical Engineering – UdeA | Junior support |

> **A note on inclusive leadership:** Roles were assigned based on capability, not credential. Luis (Forestry Engineering) handled News EDA because his analytical rigor was excellent — not because of his degree title.

---

## 3. The "Federation" Architecture

We did **not** build a monolith. We built a **federation of scientific deliverables**.

Each notebook represents a domain boundary owned by a specific team member, with explicit CSV/JSON input-output contracts. This was a deliberate integration strategy to let 7 independent contributors work in parallel without blocking each other.

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE INTEGRATION PIPELINE                     │
│                                                                 │
│  📥 INGESTION        📊 EXPLORATION       🔬 ANALYSIS          │
│  ┌──────────┐        ┌──────────┐        ┌──────────────┐      │
│  │ 01       │───────▶│ 02       │───────▶│ 04           │      │
│  │ Data Load│        │ Price EDA│        │ Anomaly Det. │      │
│  └──────────┘        └──────────┘        └──────┬───────┘      │
│       │              ┌──────────┐               │              │
│       └─────────────▶│ 03       │               │              │
│                      │ News EDA │               │              │
│                      └────┬─────┘               │              │
│                           │                     │              │
│  🧠 NLP                  │         📈 MODELING  │              │
│  ┌──────────────┐        │        ┌─────────────┴──────┐      │
│  │ 05           │◀───────┘        │ 06                 │      │
│  │ FinBERT      │────────────────▶│ Correlation &      │      │
│  │ Sentiment    │                 │ Granger Causality  │      │
│  └──────────────┘                 └────────┬───────────┘      │
│                                            │                   │
│  🤖 PREDICTION           📋 SYNTHESIS      │                   │
│  ┌──────────────┐        ┌─────────────┐   │                   │
│  │ 07           │◀───────┤ Integrated  │◀──┘                   │
│  │ LSTM Models  │        │ Dataset     │                       │
│  └──────┬───────┘        └─────────────┘                       │
│         │                                                      │
│         ▼                                                      │
│  ┌──────────────┐                                              │
│  │ 08           │                                              │
│  │ Synthesis &  │                                              │
│  │ Results      │                                              │
│  └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

| # | Notebook | Owner(s) | Purpose |
|---|----------|----------|---------|
| 01 | Introducción y Carga de Datos | Pablo | Load hourly bars + news; resample to daily; validate alignment |
| 02 | EDA Precios del Oro | Dayana | Statistical analysis, stationarity tests, seasonal decomposition |
| 03 | EDA Noticias WSJ | Luis | News volume, temporal coverage, keyword filtering |
| 04 | Detección de Anomalías | Pablo | Outlier detection in gold prices via statistical methods |
| 05 | Análisis de Sentimientos (FinBERT) | David & Sebastian | FinBERT inference on ~18K headlines; daily aggregation |
| 06 | Correlación y Causalidad | Jose | Pearson/Spearman correlation, Granger Causality tests |
| 07 | Modelo LSTM Integrado | Dayana & Pablo | Base vs. sentiment-enhanced LSTM comparison |
| 08 | Síntesis y Resultados | Pablo | Final report generation, cross-notebook synthesis |

---

## 4. Quick Start

### Prerequisites
- Python 3.8+
- ~4 GB of disk space (FinBERT model cache)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/gold-prediction-pipeline.git
cd gold-prediction-pipeline

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env if your data is not in the project root
# Default: BASE_DIR=.
```

### Running the Notebooks

```bash
cd unificacion/notebooks
jupyter notebook
```

Execute notebooks **in numerical order** (01 → 08). Each notebook reads outputs from previous ones.

> **⚠️ Portability Note:** This project originally used hardcoded absolute paths. These have been replaced with environment-variable-driven configuration via `python-dotenv`. If you encounter path issues, verify your `.env` file.

### 📊 Dataset Availability

To keep this repository lightweight, the data files included here are samples (first 500 rows). This allows you to run the notebooks and verify the pipeline logic immediately.

**Full Dataset:** If you wish to reproduce the complete study with all ~189,000 headlines and full price history (~160MB), you can download the complete database here: [https://drive.google.com/drive/folders/1osPy3E6g6bIYcpd54menGyOlnp2SlJog?usp=sharing]

---

## 5. The Integrator's Audit: Known Technical Debt

> *"The mark of a senior engineer is not writing perfect code — it's knowing exactly where your code is imperfect and why."*

After the bootcamp concluded, I conducted a post-project architectural audit. Below are the findings I documented — not to hide our team's work, but to demonstrate that I understand the gap between a bootcamp prototype and production-grade engineering.

### 🔴 Look-ahead Bias (Data Leakage)

**What:** Sentiment moving averages in Notebook 05 use `rolling(window=7, center=True)`. The `center=True` parameter means the feature for day *t* incorporates sentiment from days *t+1, t+2, t+3* — future data that wouldn't exist at prediction time.

**Impact:** Model performance metrics (RMSE, MAE) may be artificially optimistic because the LSTM had indirect access to future sentiment signals.

**Fix:** Replace centered windows with strictly causal (trailing) windows: `rolling(window=7, center=False)`.

### 🟡 Time-Frequency Alignment

**What:** News timestamps are collapsed to calendar date (`dt.date`) before merging with daily price bars. This ignores:
- After-market-close news being assigned to the same day's closing price.
- Timezone mismatches between UTC price data and local-time news timestamps.

**Impact:** Potential contamination of same-day features with information that wasn't available during the trading session.

**Fix:** Normalize all timestamps to UTC; use as-of joins with market calendar awareness.

### 🟡 Silent Data Loss via Inner Joins

**What:** The integration step in Notebook 06 uses `df_precios.join(df_sentimientos, how='inner')`, which silently drops any date that doesn't appear in both DataFrames.

**Impact:** Trading days without news coverage are excluded from analysis and modeling, potentially biasing the dataset toward "eventful" days.

**Fix:** Use a left join on the price axis and explicitly handle missing sentiment (e.g., forward-fill or neutral imputation).

### 🟢 Portability (Resolved)

**What:** All file paths originally referenced `/home/els4nchez/Videos/TECH/...`.

**Status:** ✅ Fixed. Paths now use `os.getenv('BASE_DIR')` via `python-dotenv`.

---

## 6. Lessons Learned

- **Evolution from Scripting to Engineering:** We successfully built a functional LSTM pipeline using Jupyter Notebooks for rapid prototyping. I learned that this monolithic-per-notebook structure hinders modularity. Future iterations would refactor data processing logic into a standalone Python package to enable unit testing and CI/CD integration.

- **Infrastructure Agnosticism:** The project originally relied on local file paths. A key takeaway was the necessity of environment-variable-driven configuration (12-Factor App principles) to ensure the pipeline runs identically on a developer's laptop, a CI runner, or a cloud container.

- **Separation of Concerns in ML:** We tightly coupled feature engineering with model training. I recognized that separating these into distinct steps (e.g., using a tool like Apache Airflow or Prefect) would allow for better error handling, reproducibility, and incremental data processing without retraining the entire model.

---

## 7. Future Roadmap: The Lakehouse Evolution

If this project were to evolve toward a production-grade system (e.g., at Bancolombia), the architecture would follow a **Lakehouse / Bronze-Silver-Gold** pattern:

| Layer | Purpose | Current State | Production Target |
|-------|---------|---------------|-------------------|
| **🥉 Bronze** | Raw, immutable ingestion | CSV files in `data/raw/` | Parquet/Delta partitioned by date in cloud object storage |
| **🥈 Silver** | Cleaned, normalized, validated | `datos_procesados/*.csv` | Schema-validated, UTC-normalized, with data quality gates |
| **🥇 Gold** | Business-ready features | `datos_integrados_*.csv` | Feature store with point-in-time correctness and lineage tracking |

### Key Production Components

- **Feature Store** (e.g., Feast): Enforce that every feature for time *t* is computed using only data available at or before *t*. This eliminates look-ahead bias by design.
- **Pipeline Orchestration** (e.g., Airflow/Prefect): Replace manual notebook execution with versioned, testable DAGs.
- **Experiment Tracking** (e.g., MLflow): Replace `print()` statements with structured metric logging.
- **Containerization** (Docker + CI/CD): Ensure reproducibility across environments.

---

## 8. Project Structure

```
├── .env.example                          # Env template (BASE_DIR=. default)
├── .github/                              # Repo metadata (workflows, context)
├── README.md                             # English guide
├── README.es.md                          # Spanish guide
├── requirements.txt                      # Root dependencies
├── filtrado_noticias.py                  # WSJ headline filtering script
├── data/                                 # Local data placeholder (contents local/manual)
│   ├── raw/                              # Raw scraped data + samples (local)
│   └── processed/                        # Filtered articles + samples (local)
├── datos_horas/                          # Hourly gold price bars (local)
└── unificacion/
    ├── requirements.txt                  # Pipeline dependencies
    ├── notebooks/
    │   ├── 01_Introduccion_y_Carga_de_Datos.ipynb
    │   ├── 02_EDA_Precios_Oro.ipynb
    │   ├── 03_EDA_Noticias_WSJ.ipynb
    │   ├── 04_Deteccion_Anomalias.ipynb
    │   ├── 05_Analisis_Sentimientos_FinBERT.ipynb
    │   ├── 06_Correlacion_y_Causalidad.ipynb
    │   ├── 07_Modelo_LSTM_Integrado.ipynb
    │   └── 08_Sintesis_y_Resultados.ipynb
    ├── datos_procesados/                 # Processed outputs (local/generated)
    ├── modelos/                          # Trained models (.keras) (local)
    ├── figuras/                          # Plotly figures (local)
    └── informes/                         # Summary tables (local)
```

---

## Acknowledgments

This project was developed as part of the **Talento-Tech Bootcamp (2025-2)** in collaboration with the **Universidad de Antioquia**, Medellín. Special thanks to the bootcamp instructors for creating the environment that made this scientific collaboration possible.

The post-project architectural audit was conducted independently as preparation for the **Bancolombia Talento B** program, applying enterprise engineering standards to a bootcamp prototype.

---

<p align="center">
  <i>Built by physicists. Integrated by an engineer-in-training. Audited with honesty.</i>
</p>
