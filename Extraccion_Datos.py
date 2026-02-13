import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import os
import signal
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================
# CONFIGURACIÓN INTERACTIVA
# ============================================================
print("=" * 60)
print("CONFIGURACIÓN DE DESCARGA DE DATOS DE DUKASCOPY")
print("=" * 60)

# Símbolo
print("\nSímbolos disponibles:")
print("  1. XAU-USD (Oro)")
print("  2. XTI-USD (WTI - Petróleo)")
print("  3. UKOIL (Brent)")
symbol_choice = input("Selecciona símbolo (1-3) [default: 1]: ").strip() or "1"
symbols = {"1": "XAU-USD", "2": "XTI-USD", "3": "UKOIL"}
symbol = symbols.get(symbol_choice, "XAU-USD")

# Años
years_input = input("\nAño(s) a descargar (ej: 2025, o rango: 2023-2025) [default: 2025]: ").strip() or "2025"
if "-" in years_input:
    start_year, end_year = map(int, years_input.split("-"))
    years = np.arange(start_year, end_year + 1, dtype=int)
else:
    years = np.array([int(years_input)])

# Meses
months_input = input("Mes(es) a descargar (ej: 10, o rango: 1-12, o específicos: 1,6,12) [default: 10]: ").strip() or "10"
if "-" in months_input:
    start_month, end_month = map(int, months_input.split("-"))
    months = np.arange(start_month, end_month + 1, dtype=int)
elif "," in months_input:
    months = np.array([int(m.strip()) for m in months_input.split(",")])
else:
    months = np.array([int(months_input)])

# Días
days_input = input("Días a descargar (ej: 1-31, o días específicos: 1,15,20) [default: 1-31]: ").strip() or "1-31"
if "-" in days_input:
    start_day, end_day = map(int, days_input.split("-"))
    days = np.arange(start_day, end_day + 1, dtype=int)
elif "," in days_input:
    days = np.array([int(d.strip()) for d in days_input.split(",")])
else:
    days = np.array([int(days_input)])

# Timeframe
print("\nTimeframes disponibles:")
print("  1. Segundos (1s)")
print("  2. Minutos (1min)")
print("  3. Horas (1h)")
print("  4. Días (1D)")
timeframe_choice = input("Selecciona timeframe (1-4) [default: 3]: ").strip() or "3"
timeframes = {
    "1": ("1s", "segundos"),
    "2": ("1min", "minutos"),
    "3": ("1h", "horas"),
    "4": ("1D", "dias")
}
timeframe, folder_suffix = timeframes.get(timeframe_choice, ("1h", "horas"))

# Formato de salida
print("\nFormatos de salida:")
print("  1. Excel (.xlsx)")
print("  2. CSV (.csv)")
format_choice = input("Selecciona formato (1-2) [default: 1]: ").strip() or "1"
output_format = "xlsx" if format_choice == "1" else "csv"

# Agrupación de archivos
print("\n¿Cómo deseas guardar los datos?")
print("  1. Un archivo por todo el rango (todos los años/meses/días en un solo archivo)")
print("  2. Un archivo por año")
print("  3. Un archivo por mes")
print("  4. Un archivo por día")
grouping_choice = input("Selecciona agrupación (1-4) [default: 4]: ").strip() or "4"
grouping_options = {
    "1": "range",
    "2": "year",
    "3": "month",
    "4": "day"
}
grouping = grouping_options.get(grouping_choice, "day")

output_folder = f"./datos_{folder_suffix}"

print("\n" + "=" * 60)
print("CONFIGURACIÓN SELECCIONADA:")
print("=" * 60)
print(f"Símbolo: {symbol}")
print(f"Años: {years}")
print(f"Meses: {months}")
print(f"Días: {days}")
print(f"Timeframe: {timeframe}")
print(f"Formato: {output_format.upper()}")
print(f"Agrupación: {grouping}")
print(f"Carpeta de salida: {output_folder}")
print("=" * 60)

confirm = input("\n¿Continuar con esta configuración? (s/n) [default: s]: ").strip().lower() or "s"
if confirm != "s":
    print("Proceso cancelado.")
    sys.exit(0)

print("\nIniciando descarga...\n")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Control para detener el proceso limpiamente con Ctrl+C
stop_requested = False

def _signal_handler(signum, frame):
    global stop_requested
    print('\nRecibida señal de parada, terminando después de la iteración actual...')
    stop_requested = True


# Registrar handlers para SIGINT/SIGTERM
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# Crear sesión global con connection pooling y retry strategy
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session = create_session()

def fetch_and_decode_ticks(year, month0, day, hour, timeout=5):
    url = f"https://jetta.dukascopy.com/v1/ticks/{symbol}/{year}/{month0:02d}/{day:02d}/{hour:02d}"
    
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
    except KeyboardInterrupt:
        raise
    except Exception:
        return None

    if stop_requested or data is None:
        return None

    # Decodificación optimizada con operaciones vectorizadas
    initial_ts = data['timestamp']
    multiplier = data['multiplier']
    times = np.array(data['times'])
    asks_deltas = np.array(data['asks'])
    bids_deltas = np.array(data['bids'])
    ask_vols = np.array(data['askVolumes']) / 1000000
    bid_vols = np.array(data['bidVolumes']) / 1000000
    
    # Calcular timestamps acumulativos
    timestamps = initial_ts + np.cumsum(times)
    
    # Calcular precios acumulativos
    asks = data['ask'] + np.cumsum(asks_deltas * multiplier)
    bids = data['bid'] + np.cumsum(bids_deltas * multiplier)
    
    # Crear DataFrame directamente desde arrays
    return pd.DataFrame({
        'timestamp_ms': timestamps,
        'ask': asks,
        'bid': bids,
        'ask_vol': ask_vols,
        'bid_vol': bid_vols
    })

def aggregate_bars(df_ticks, timeframe, add_utc_column=True):
    """Agrega ticks al timeframe especificado"""
    df_ticks['timestamp'] = pd.to_datetime(df_ticks['timestamp_ms'], unit='ms', utc=True)
    df_ticks.set_index('timestamp', inplace=True)

    # Usar solo BID con agregación optimizada
    df_bars = df_ticks['bid'].resample(timeframe).agg(['first', 'max', 'min', 'last']).rename(
        columns={'first': 'Open', 'max': 'High', 'min': 'Low', 'last': 'Close'}
    )
    df_bars['Volume'] = df_ticks['bid_vol'].resample(timeframe).sum()
    df_bars.dropna(subset=['Open'], inplace=True)
    df_bars.reset_index(inplace=True)

    # Solo agregar columna UTC al final para ahorrar tiempo durante acumulación
    if add_utc_column:
        if timeframe in ['1s', '1min', '1h']:
            df_bars['UTC'] = df_bars['timestamp'].dt.strftime('%d.%m.%Y %H:%M:%S') + ' UTC'
        else:
            df_bars['UTC'] = df_bars['timestamp'].dt.strftime('%d.%m.%Y') + ' UTC'
        df_bars = df_bars[['UTC', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df_bars

def save_file(df, folder, filename, format_type, timeframe_val):
    """Guarda el DataFrame en el formato especificado"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Agregar columna UTC solo al momento de guardar si no existe
    if 'UTC' not in df.columns and 'timestamp' in df.columns:
        if timeframe_val in ['1s', '1min', '1h']:
            df['UTC'] = df['timestamp'].dt.strftime('%d.%m.%Y %H:%M:%S') + ' UTC'
        else:
            df['UTC'] = df['timestamp'].dt.strftime('%d.%m.%Y') + ' UTC'
        df = df[['UTC', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    if format_type == "xlsx":
        file_path = f"{folder}/{filename}.xlsx"
        df.to_excel(file_path, index=False, engine='openpyxl')
    else:
        file_path = f"{folder}/{filename}.csv"
        df.to_csv(file_path, index=False)
    
    return file_path


# ============================================================
# DESCARGA DE DATOS CON PROCESAMIENTO PARALELO
# ============================================================
# La API de Dukascopy usa meses base-0 (0=enero, 9=octubre, 10=noviembre)
# pero tú configuras el mes normal (1-12), así que restamos 1 aquí

# Diccionario para almacenar todos los datos según la agrupación
all_data = {}

def download_hour(year, month_api, day, hour):
    """Descarga y procesa una hora específica"""
    df_hour = fetch_and_decode_ticks(year, month_api, day, hour)
    if df_hour is not None and not df_hour.empty:
        # No agregar UTC todavía para ahorrar tiempo
        daily_bars = aggregate_bars(df_hour, timeframe, add_utc_column=False)
        return hour, daily_bars, len(df_hour)
    return hour, None, 0

for year in years:
    for month in months:
        for day in days:

            month_api = month - 1

            print(f"Descargando {symbol} para {day:02d}/{month:02d}/{year}...")

            daily_df = pd.DataFrame()
            
            # Descargar todas las horas en paralelo (máximo 12 workers para no saturar)
            with ThreadPoolExecutor(max_workers=12) as executor:
                futures = {executor.submit(download_hour, year, month_api, day, hour): hour 
                          for hour in range(24)}
                
                for future in as_completed(futures):
                    if stop_requested:
                        print('\nDetenido por solicitud del usuario.')
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    hour, df_hour, tick_count = future.result()
                    if df_hour is not None:
                        daily_df = pd.concat([daily_df, df_hour], ignore_index=True)
                        print(f"  Hora {hour:02d} ✓ ({tick_count} ticks → {len(df_hour)} barras)", flush=True)
            
            if stop_requested:
                break

            # Guardar según la agrupación seleccionada
            if not daily_df.empty:
                if grouping == "day":
                    # Guardar inmediatamente por día
                    filename = f"{symbol.replace('-', '_')}_{year}-{month:02d}-{day:02d}_{timeframe}_bars"
                    save_file(daily_df, output_folder, filename, output_format, timeframe)
                    print(f"\n✓ ¡Guardado! Total: {len(daily_df)} barras")
                elif grouping == "month":
                    # Acumular por mes
                    key = f"{year}-{month:02d}"
                    if key not in all_data:
                        all_data[key] = pd.DataFrame()
                    all_data[key] = pd.concat([all_data[key], daily_df], ignore_index=True)
                    print(f"\n✓ Acumulado para mes {month:02d}/{year}")
                elif grouping == "year":
                    # Acumular por año
                    key = f"{year}"
                    if key not in all_data:
                        all_data[key] = pd.DataFrame()
                    all_data[key] = pd.concat([all_data[key], daily_df], ignore_index=True)
                    print(f"\n✓ Acumulado para año {year}")
                elif grouping == "range":
                    # Acumular todo
                    if "all" not in all_data:
                        all_data["all"] = pd.DataFrame()
                    all_data["all"] = pd.concat([all_data["all"], daily_df], ignore_index=True)
                    print(f"\n✓ Acumulado en rango total")
            else:
                print(f"\n✗ No se encontraron datos para {year}-{month:02d}-{day:02d}")

            if stop_requested:
                break
        
        # Si agrupamos por mes, guardar al terminar cada mes
        if grouping == "month" and not stop_requested:
            key = f"{year}-{month:02d}"
            if key in all_data and not all_data[key].empty:
                filename = f"{symbol.replace('-', '_')}_{year}-{month:02d}_{timeframe}_bars"
                save_file(all_data[key], output_folder, filename, output_format, timeframe)
                print(f"\n✓ ¡Guardado mes completo! {filename} - Total: {len(all_data[key])} barras")
        
        if stop_requested:
            break
    
    # Si agrupamos por año, guardar al terminar cada año
    if grouping == "year" and not stop_requested:
        key = f"{year}"
        if key in all_data and not all_data[key].empty:
            filename = f"{symbol.replace('-', '_')}_{year}_{timeframe}_bars"
            save_file(all_data[key], output_folder, filename, output_format, timeframe)
            print(f"\n✓ ¡Guardado año completo! {filename} - Total: {len(all_data[key])} barras")
    
    if stop_requested:
        break

# Si agrupamos por rango, guardar al final
if grouping == "range" and "all" in all_data and not all_data["all"].empty:
    year_range = f"{years[0]}-{years[-1]}" if len(years) > 1 else f"{years[0]}"
    month_range = f"{months[0]:02d}-{months[-1]:02d}" if len(months) > 1 else f"{months[0]:02d}"
    filename = f"{symbol.replace('-', '_')}_{year_range}_{month_range}_{timeframe}_bars"
    save_file(all_data["all"], output_folder, filename, output_format, timeframe)
    print(f"\n✓ ¡Guardado rango completo! {filename} - Total: {len(all_data['all'])} barras")

print("\n¡Proceso completado!")