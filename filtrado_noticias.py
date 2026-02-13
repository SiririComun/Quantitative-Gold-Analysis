import os
import pandas as pd
import re
from urllib.parse import urlparse
from pathlib import Path

from dotenv import load_dotenv

# --- CONFIGURACIÓN ---
load_dotenv()
# Usar rutas absolutas del workspace
BASE_DIR = Path(os.getenv('BASE_DIR', '.')).resolve()
DATA_RAW_DIR = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
archivo_entrada = DATA_RAW_DIR / 'hipervinculos_wsj.csv'
archivo_entrada_sample = DATA_RAW_DIR / 'hipervinculos_wsj_sample.csv'
archivo_salida = DATA_PROCESSED_DIR / 'articulos_filtrados_ordenados.csv'
nombre_columna_fecha = 'fecha'
# ---------------------

print("=" * 80)
print("FILTRADO DE HIPERVÍNCULOS WSJ → ARTÍCULOS REALES")
print("=" * 80)


def seleccionar_archivo(archivos: list[Path]) -> Path:
    for path in archivos:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No se encontraron archivos de entrada. Esperados: "
        f"{', '.join(str(p) for p in archivos)}"
    )

# 1. Cargar el archivo (usa muestra si el completo no está disponible)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
archivo_seleccionado = seleccionar_archivo([archivo_entrada, archivo_entrada_sample])
print(f"Archivo usado: {archivo_seleccionado}")
df = pd.read_csv(archivo_seleccionado)
print(f"Total de registros cargados: {len(df):,}")
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

# 2. Definir criterios de inclusión y exclusión
# Inclusión: URLs que representan páginas de artículos en nuevas secciones de WSJ
# - /articles/…-NUM
# - /business/…-NUM, /politics/…-NUM, /world/…-NUM, /markets/…-NUM, /economy/…-NUM
# Excluir navegación y no-artículos: /news/archive, /video, /livecoverage, /podcast, /photos, /client/login

secciones_articulo = [
	# Núcleo histórico y nuevas clasificaciones WSJ
	'articles',
	'business',
	'politics',
	'world',
	'finance',
	'economy',
	'markets',
	'us-news',
	'tech',
	'health',
	'sports',
	'lifestyle',
	'arts-culture',
	'real-estate',
	'style',
	'personal-finance',
	'science',
	'opinion',
]

# Subrutas frecuentes dentro de secciones (para profundidad adicional)
subrutas_relevantes = [
	'markets', 'economy', 'macroeconomics', 'policy', 'elections', 'china', 'middle-east', 'europe',
	'autos', 'retail', 'media', 'deals', 'tech', 'personal-tech', 'cybersecurity', 'biotech',
	'investing', 'banking', 'housing', 'healthcare', 'law', 'national-security', 'courts',
	'luxury-homes', 'commercial', 'fashion', 'design', 'travel', 'relationships'
]

# Exclusiones de navegación / contenido no artículo
pat_excluir = re.compile(
	r"/news/archive|/video/|/livecoverage|/podcast|/photos|/client/login|/subscribe|/newsletter|/audio|/market-data|/topics|/pro|/professional",
	re.IGNORECASE,
)

# Heurística más amplia: considerar artículo si
# 1) URL termina con -dígitos (ID de artículo), o
# 2) pertenece a sección conocida y tiene profundidad >= 3 y el último segmento es un slug con guiones

def es_articulo(url: str) -> bool:
	if not isinstance(url, str):
		return False
	if pat_excluir.search(url):
		return False
	# Regla 1: ID de artículo clásico
	if re.search(r"-\d+$", url):
		return True
	# Parseo robusto de ruta
	try:
		p = urlparse(url)
		parts = [x for x in p.path.split('/') if x]
	except Exception:
		parts = []
	if not parts:
		return False
	seg = parts[0].lower()
	last = parts[-1]
	depth = len(parts)
	# Solo considerar secciones conocidas
	if seg not in secciones_articulo:
		return False
	# Heurísticas S4: slug largo o parámetros mod/st/share
	if (last.count('-') >= 2 and len(last) >= 15) or re.search(r"[?&](mod|st|share)=", url):
		return True
	# Subrutas relevantes + slug con guion
	if any(sr in parts[:-1] for sr in subrutas_relevantes) and '-' in last:
		return True
	# Regla de un solo guion para recientes: profundidad >=3, longitud suficiente y no endpoint
	if '-' in last and depth >= 3 and len(last) >= 12 and not re.search(r"^(index|video|photos|gallery)$", last):
		return True
	# Opinión suele tener slugs descriptivos
	if seg == 'opinion' and '-' in last:
		return True
	return False

df_filtrado = df[df['url'].apply(es_articulo)].copy()
print(f"Registros después de incluir secciones y excluir navegación: {len(df_filtrado):,}")

# 3. Eliminar duplicados basados en URL
df_sin_duplicados = df_filtrado.drop_duplicates(subset=['url'], keep='first')
print(f"Registros después de eliminar duplicados de URL: {len(df_sin_duplicados):,}")

# 4. Convertir columna de fecha
df_sin_duplicados[nombre_columna_fecha] = pd.to_datetime(
	df_sin_duplicados[nombre_columna_fecha], errors='coerce'
)

# 5. Eliminar filas con fechas inválidas
df_con_fechas_validas = df_sin_duplicados.dropna(subset=[nombre_columna_fecha])
print(f"Registros después de eliminar fechas inválidas: {len(df_con_fechas_validas):,}")

# 6. Ordenar por fecha (asc)
df_ordenado = df_con_fechas_validas.sort_values(by=nombre_columna_fecha, ascending=True)

# 7. Guardar
df_ordenado.to_csv(archivo_salida, index=False)

print("\nProceso finalizado.")
print(f"Se encontraron {len(df_ordenado):,} artículos válidos.")
print(f"Archivo guardado en: {archivo_salida}")
print("\nPrimeras 5 filas:")
print(df_ordenado.head())