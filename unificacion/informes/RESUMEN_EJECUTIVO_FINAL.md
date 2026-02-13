
# RESUMEN EJECUTIVO
## Predicción de Precios del Oro con Análisis de Sentimientos de Noticias

**Fecha:** 2026-02-13
**Equipo:** Análisis de Datos - Proyecto Oro

---

## 1. DATOS PROCESADOS

- **Precios del oro:** 3,614 días (2016-01-03 a 2025-11-24)
- **Noticias analizadas:** 18,776 artículos del Wall Street Journal
- **Outliers detectados:** 132 eventos anómalos en precios
- **Dataset integrado:** 2,273 días con precios y sentimientos

## 2. METODOLOGÍA

### 2.1 Análisis de Sentimientos
- **Modelo:** FinBERT (ProsusAI/finbert)
- **Clasificación:** Positivo, Neutral, Negativo
- **Distribución:**
  - Positivo: 0.0%
  - Neutral: 0.0%
  - Negativo: 0.0%

### 2.2 Detección de Anomalías
- **Métodos:** IQR, Z-Score, Isolation Forest
- **Consenso:** Outliers detectados por ≥2 métodos

### 2.3 Modelos LSTM
- **Arquitectura:** 256-128-64 units, dropout 0.2
- **Secuencias:** 60 días lookback
- **División:** 60% train / 20% val / 20% test

## 3. RESULTADOS PRINCIPALES

### 3.1 Correlación y Causalidad
- **Lag óptimo:** -6 días
- **Correlación máxima:** -0.0696
- **Test de Granger:** NO SIGNIFICATIVO

### 3.2 Performance de Modelos

**LSTM Base (sin sentimiento):**
- RMSE: $524.01
- MAE: $453.63
- R²: -0.7065

**LSTM + Sentimiento:**
- RMSE: $555.55
- MAE: $486.24
- R²: -0.9181

**Mejora con sentimiento:**
- RMSE: -6.02%
- MAE: -7.19%
- R²: -29.95%

## 4. CONCLUSIONES

1. **Valor predictivo del sentimiento:** El sentimiento NO mejora significativamente las predicciones.

2. **Causalidad:** No se encontró causalidad Granger significativa.

3. **Anomalías:** Se detectaron 132 eventos anómalos en el precio del oro, algunos coincidentes con noticias de sentimiento extremo.

## 5. TRABAJO FUTURO

- Incorporar más fuentes de noticias (Reuters, Bloomberg)
- Analizar contenido completo de artículos
- Modelos más sofisticados (Transformers)
- Indicadores macroeconómicos adicionales
- Estrategias de trading basadas en sentimiento

---

**Archivos generados:**
- 8 tablas de resultados (CSV)
- 2 figuras principales (HTML + PNG)
- Modelos entrenados (Keras .keras)
- Datasets procesados (CSV)
