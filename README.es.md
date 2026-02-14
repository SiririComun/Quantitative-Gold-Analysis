# 🥇 Análisis cuantitativo del oro: un enfoque científico del sentimiento del mercado y las LSTM

> **Análisis de Sentimientos NLP × Modelado de Series Temporales con LSTM**  
> Bootcamp Talento-Tech 2025-2 · Universidad de Antioquia · Medellín, Colombia

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Estado: Retrospectiva](https://img.shields.io/badge/Estado-Retrospectiva%20Técnica-orange.svg)](#5-análisis-retrospectivo-y-deuda-técnica)

### 👉 [English Version here](README.md)

---

## Tabla de Contenidos

1. [Resumen del Proyecto](#1-resumen-del-proyecto)
2. [El Equipo: Físicos con Mentalidad de Ingeniería](#2-el-equipo-físicos-con-mentalidad-de-ingeniería)
3. [Estructura Modular por Dominios](#3-estructura-modular-por-dominios)
4. [Inicio Rápido](#4-inicio-rápido)
5. [Análisis Retrospectivo y Deuda Técnica](#5-análisis-retrospectivo-y-deuda-técnica)
6. [Lecciones Aprendidas](#6-lecciones-aprendidas)
7. [Hoja de Ruta: La Evolución Lakehouse](#7-hoja-de-ruta-la-evolución-lakehouse)
8. [Estructura del Proyecto](#8-estructura-del-proyecto)

---

## 1. Resumen del Proyecto

¿Puede el sentimiento de las noticias financieras **predecir** los movimientos del precio del oro?

Este proyecto construye un pipeline de extremo a extremo que:
- **Extrae** ~189.000 titulares del Wall Street Journal (2016–2025).
- **Filtra** ~18.700 artículos relacionados con oro mediante heurísticas de palabras clave.
- **Califica** cada titular con **FinBERT** (ProsusAI/finbert), un modelo transformer afinado para sentimiento financiero.
- **Detecta anomalías** en los datos del precio del oro usando métodos estadísticos.
- **Prueba causalidad** entre sentimiento y precio mediante Causalidad de Granger.
- **Predice** el precio diario de cierre del oro con redes LSTM — comparando un modelo base (solo precio) contra un modelo enriquecido con sentimiento.

**Hallazgo clave:** Integrar features de sentimiento FinBERT en el LSTM mejoró la precisión predictiva, aunque la señal causal es matizada. El análisis completo, incluyendo las advertencias estadísticas, está documentado a lo largo de 8 notebooks.

---

## 2. El Equipo: Físicos con Mentalidad de Ingeniería

### Dupla de Liderazgo y Gobierno Técnico

| Rol | Miembro | Responsabilidad | Certificación |
| :--- | :--- | :--- | :--- |
| **Líder Técnico e Integrador** | Pablo Sanchez *(Físico – UdeA)* | Unificación del pipeline, arquitectura de 8 notebooks, Detección de anomalías, auditoría post-proyecto | [Verificar Talento Tech 🏆](https://www.auco.ai/verify/?code=HXKAW5DL5W) |
| **Co-Líder y Analista Estadístico** | Jose Ortiz *(Físico – UdeA)* | Infraestructura de Web scraping, análisis de Causalidad de Granger, validación de datos | [Verificar Talento Tech 🏆](PONER_URL_DE_JOSE) |

Pablo y Jose fueron la **dupla técnica central** del proyecto. Pablo se encargó de que el pipeline corriera de punta a punta; Jose se encargó de que cada resultado estadístico fuera defendible. Este esquema — donde uno hace las veces de Ingeniero de Plataforma y el otro de Analista Cuantitativo — blindó la integridad de los datos en todo el flujo, desde la extracción hasta el modelo final.

### El Equipo Completo

| Miembro | Formación | Contribución |
|---------|-----------|-------------|
| David Alava | Física – UdeA | Especialista NLP (implementación de FinBERT) |
| Sebastian Agudelo | Física – UdeA | Especialista NLP (implementación de FinBERT) |
| Dayana Henao | Física – UdeA | Ingeniera ML (LSTM y EDA de Precios del Oro) |
| Luis Vera | Ingeniería Forestal | Especialista en EDA (análisis de noticias) |
| Michael Tarazona | Ingeniería Eléctrica – UdeA | Apoyo general |

> **Sobre la asignación de roles:** Cada persona recibió su rol por lo que sabía hacer, no por lo que decía su título. Luis, siendo Ingeniero Forestal, lideró el EDA de Noticias porque su capacidad analítica lo respaldaba. En un equipo de físicos, eso se respeta.

---

## 3. Estructura Modular por Dominios

Este proyecto **no** es un notebook gigante donde todo está revuelto. Es una **estructura modular donde cada notebook corresponde a un dominio técnico concreto**, con un responsable claro y contratos de entrada/salida definidos (archivos CSV y JSON).

¿Por qué lo diseñamos así? Porque éramos 7 personas trabajando en paralelo. Si todo vivía en un solo archivo, los conflictos de merge y las dependencias cruzadas nos habrían frenado. Cada notebook es una pieza independiente que recibe datos del paso anterior y entrega resultados al siguiente — como un mini-servicio, pero a escala de bootcamp.

```
┌─────────────────────────────────────────────────────────────────┐
│                   EL PIPELINE DE INTEGRACIÓN                    │
│                                                                 │
│  📥 INGESTA           📊 EXPLORACIÓN       🔬 ANÁLISIS         │
│  ┌──────────┐        ┌──────────┐        ┌──────────────┐      │
│  │ 01       │───────▶│ 02       │───────▶│ 04           │      │
│  │ Carga    │        │ EDA      │        │ Detección    │      │
│  │ Datos    │        │ Precios  │        │ Anomalías    │      │
│  └──────────┘        └──────────┘        └──────┬───────┘      │
│       │              ┌──────────┐               │              │
│       └─────────────▶│ 03       │               │              │
│                      │ EDA      │               │              │
│                      │ Noticias │               │              │
│                      └────┬─────┘               │              │
│                           │                     │              │
│  🧠 NLP                  │         📈 MODELADO  │              │
│  ┌──────────────┐        │        ┌─────────────┴──────┐      │
│  │ 05           │◀───────┘        │ 06                 │      │
│  │ FinBERT      │────────────────▶│ Correlación &      │      │
│  │ Sentimiento  │                 │ Causalidad Granger │      │
│  └──────────────┘                 └────────┬───────────┘      │
│                                            │                   │
│  🤖 PREDICCIÓN           📋 SÍNTESIS       │                   │
│  ┌──────────────┐        ┌─────────────┐   │                   │
│  │ 07           │◀───────┤ Dataset     │◀──┘                   │
│  │ Modelos LSTM │        │ Integrado   │                       │
│  └──────┬───────┘        └─────────────┘                       │
│         │                                                      │
│         ▼                                                      │
│  ┌──────────────┐                                              │
│  │ 08           │                                              │
│  │ Síntesis &   │                                              │
│  │ Resultados   │                                              │
│  └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

| # | Notebook | Responsable(s) | Propósito |
|---|----------|----------------|-----------|
| 01 | Introducción y Carga de Datos | Pablo | Carga de barras horarias + noticias; resampleo diario; validación de alineación |
| 02 | EDA Precios del Oro | Dayana | Análisis estadístico, pruebas de estacionariedad, descomposición estacional |
| 03 | EDA Noticias WSJ | Luis | Volumen de noticias, cobertura temporal, filtrado por palabras clave |
| 04 | Detección de Anomalías | Pablo | Detección de outliers en precios del oro mediante métodos estadísticos |
| 05 | Análisis de Sentimientos (FinBERT) | David y Sebastian | Inferencia FinBERT sobre ~18K titulares; agregación diaria |
| 06 | Correlación y Causalidad | Jose | Correlación Pearson/Spearman, pruebas de Causalidad de Granger |
| 07 | Modelo LSTM Integrado | Dayana y Pablo | Comparación LSTM base vs. enriquecido con sentimiento |
| 08 | Síntesis y Resultados | Pablo | Generación del reporte final, síntesis entre notebooks |

---

## 4. Inicio Rápido

### Prerequisitos
- Python 3.8+
- ~4 GB de espacio en disco (caché del modelo FinBERT)

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/SiririComun/Quantitative-Gold-Analysis.git
cd Quantitative-Gold-Analysis

# 2. Crear y activar un entorno virtual (Recomendado)
python -m venv venv
# En Linux/macOS:
source venv/bin/activate  
# En Windows:
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar entorno
cp .env.example .env
# Editar .env si los datos no están en la raíz del proyecto
# Por defecto: BASE_DIR=.
```

### Ejecución de los Notebooks

```bash
cd unificacion/notebooks
jupyter notebook
```

Ejecutar los notebooks **en orden numérico** (01 → 08). Cada notebook lee las salidas de los anteriores.

> **⚠️ Nota de Portabilidad:** Este proyecto originalmente usaba rutas absolutas hardcodeadas. Estas han sido reemplazadas con configuración basada en variables de entorno mediante `python-dotenv`. Si encuentras problemas con rutas, verifica tu archivo `.env`.

### 📊 Disponibilidad del conjunto de datos

Para que este repositorio sea ligero, los archivos de datos incluidos aquí son muestras (las primeras 500 filas). Esto le permite ejecutar los cuadernos y verificar la lógica del proceso de inmediato.

**Conjunto de datos completo:** si desea reproducir el estudio completo con los aproximadamente 189 000 titulares y el historial de precios completo (aproximadamente 160 MB), puede descargar la base de datos completa aquí: [https://drive.google.com/drive/folders/1osPy3E6g6bIYcpd54menGyOlnp2SlJog?usp=sharing]

---

## 5. Análisis Retrospectivo y Deuda Técnica

> *"Lo que distingue a un ingeniero senior no es entregar código perfecto, sino saber exactamente dónde están las grietas y tener un plan para cerrarlas."*

Una vez terminó el bootcamp, me senté a revisar el proyecto con ojos de auditor. Lo que sigue no es un intento de esconder las limitaciones del equipo — al contrario: es la demostración de que entiendo la distancia entre un prototipo académico y un sistema que pueda correr en producción real.

### 🔴 Sesgo de Anticipación (Look-ahead Bias)

**El problema:** En el Notebook 05, las medias móviles de sentimiento se calculan con `rolling(window=7, center=True)`. Ese `center=True` hace que el valor del día *t* use información de los días *t+1, t+2, t+3* — es decir, datos del futuro que en la práctica no existirían al momento de hacer la predicción.

**Por qué importa:** Las métricas del modelo (RMSE, MAE) pueden verse artificialmente buenas porque el LSTM, de manera indirecta, tuvo acceso a señales de sentimiento que aún no habían ocurrido. En un contexto bancario, esto invalida el backtesting.

**Cómo se corrige:** Usar ventanas estrictamente retrospectivas: `rolling(window=7, center=False)`. La regla es simple — todo feature del día *t* debe calcularse **solo** con datos disponibles hasta el día *t*.

### 🟡 Desalineación entre Frecuencias Temporales

**El problema:** Los timestamps de las noticias se reducen a la fecha (`dt.date`) antes de cruzarlos con las barras diarias de precio. Esto pasa por alto dos cosas:
- Una noticia publicada a las 11 PM (después del cierre del mercado) queda asignada al precio de cierre de ese mismo día, como si el mercado ya hubiera "reaccionado".
- Los precios están en UTC y los timestamps de las noticias no tienen una normalización explícita de zona horaria.

**Por qué importa:** Se pueden contaminar features del mismo día con información que no estaba disponible durante la sesión de trading. En una mesa de operaciones, eso genera decisiones basadas en datos falsos.

**Cómo se corrige:** Normalizar todo a UTC, usar calendarios de mercado (`exchange_calendars`) y hacer joins tipo *as-of* — que asignan cada noticia a la **siguiente** barra de precio disponible, no a la del mismo día.

### 🟡 Pérdida Silenciosa de Datos por Inner Join

**El problema:** En el Notebook 06, la integración usa `df_precios.join(df_sentimientos, how='inner')`. Esto descarta sin aviso cualquier fecha donde no existan datos en ambos DataFrames.

**Por qué importa:** Los días de trading sin cobertura de noticias quedan por fuera del análisis. El dataset resultante está sesgado hacia días "con eventos", lo cual puede distorsionar tanto las correlaciones como el entrenamiento del LSTM.

**Cómo se corrige:** Usar un left join sobre el eje de precios y manejar el sentimiento faltante de forma explícita (forward-fill o imputación con valor neutro).

### 🟢 Portabilidad (Resuelta)

**El problema:** Todas las rutas apuntaban a `/home/els4nchez/Videos/TECH/...` — nadie más podía correr el proyecto sin editar el código.

**Estado:** ✅ Corregido. Ahora se usa `os.getenv('BASE_DIR')` con `python-dotenv`. Basta con copiar `.env.example` a `.env` y listo.

---

## 6. Lecciones Aprendidas

- **Evolución de Scripting a Ingeniería:** Construimos exitosamente un pipeline LSTM funcional usando Jupyter Notebooks para prototipado rápido. Aprendí que esta estructura monolítica-por-notebook dificulta la modularidad. Las iteraciones futuras refactorizarían la lógica de procesamiento de datos en un paquete Python independiente para habilitar pruebas unitarias e integración CI/CD.

- **Agnosticismo de Infraestructura:** El proyecto originalmente dependía de rutas de archivos locales. Una lección clave fue la necesidad de configuración basada en variables de entorno (principios de la App de 12 Factores) para asegurar que el pipeline funcione de manera idéntica en la laptop de un desarrollador, un runner de CI o un contenedor en la nube.

- **Separación de Responsabilidades en ML:** Acoplamos estrechamente la ingeniería de features con el entrenamiento del modelo. Reconocí que separar estos en pasos distintos (e.g., usando una herramienta como Apache Airflow o Prefect) permitiría un mejor manejo de errores, reproducibilidad y procesamiento incremental de datos sin reentrenar el modelo completo.

---

## 7. Hoja de Ruta: La Evolución Lakehouse

Si este proyecto fuera a evolucionar hacia un sistema de grado productivo (e.g., en Bancolombia), la arquitectura seguiría un patrón **Lakehouse / Bronce-Plata-Oro**:

| Capa | Propósito | Estado Actual | Objetivo Productivo |
|------|-----------|---------------|---------------------|
| **🥉 Bronce** | Ingesta cruda e inmutable | Archivos CSV en `data/raw/` | Parquet/Delta particionado por fecha en almacenamiento cloud |
| **🥈 Plata** | Limpio, normalizado, validado | `datos_procesados/*.csv` | Validación de esquema, normalización UTC, puertas de calidad de datos |
| **🥇 Oro** | Features listos para negocio | `datos_integrados_*.csv` | Feature store con corrección point-in-time y trazabilidad de linaje |

### Componentes Clave de Producción

- **Feature Store** (e.g., Feast): Garantizar que cada feature para el tiempo *t* se compute usando solo datos disponibles en o antes de *t*. Esto elimina el sesgo de anticipación por diseño.
- **Orquestación de Pipelines** (e.g., Airflow/Prefect): Reemplazar la ejecución manual de notebooks con DAGs versionados y testeables.
- **Seguimiento de Experimentos** (e.g., MLflow): Reemplazar los `print()` con logging estructurado de métricas.
- **Contenerización** (Docker + CI/CD): Asegurar reproducibilidad entre entornos.

---

## 8. Estructura del Proyecto

```
├── .env.example                          # Plantilla de entorno (BASE_DIR=. por defecto)
├── .github/                              # Metadatos del repo (workflows, contexto)
├── README.md                             # Versión en inglés
├── README.es.md                          # Este archivo (Español)
├── requirements.txt                      # Dependencias raíz
├── filtrado_noticias.py                  # Script de filtrado de titulares WSJ
├── data/                                 # Carpeta local (contenido se mantiene local/manual)
│   ├── raw/                              # Datos crudos + muestras (local)
│   └── processed/                        # Artículos filtrados + muestras (local)
├── datos_horas/                          # Barras horarias del precio del oro (local)
└── unificacion/
    ├── requirements.txt                  # Dependencias del pipeline
    ├── notebooks/
    │   ├── 01_Introduccion_y_Carga_de_Datos.ipynb
    │   ├── 02_EDA_Precios_Oro.ipynb
    │   ├── 03_EDA_Noticias_WSJ.ipynb
    │   ├── 04_Deteccion_Anomalias.ipynb
    │   ├── 05_Analisis_Sentimientos_FinBERT.ipynb
    │   ├── 06_Correlacion_y_Causalidad.ipynb
    │   ├── 07_Modelo_LSTM_Integrado.ipynb
    │   └── 08_Sintesis_y_Resultados.ipynb
    ├── datos_procesados/                 # Salidas procesadas (local/generadas)
    ├── modelos/                          # Modelos LSTM entrenados (.keras) (local)
    ├── figuras/                          # Gráficos interactivos Plotly (local)
    └── informes/                         # Tablas resumen (local)
```

---

## Agradecimientos

Este proyecto fue desarrollado como parte del **Bootcamp Talento-Tech (2025-2)** en colaboración con la **Universidad de Antioquia**, Medellín. Agradecimiento especial a los instructores del bootcamp por crear el ambiente que hizo posible esta colaboración científica.

La auditoría arquitectónica post-proyecto fue realizada de forma independiente como preparación para el programa **Bancolombia Talento B**, aplicando estándares de ingeniería empresarial a un prototipo de bootcamp.

---

<p align="center">
  <i>Construido por físicos. Integrado por un ingeniero en formación. Auditado con honestidad.</i>
</p>
