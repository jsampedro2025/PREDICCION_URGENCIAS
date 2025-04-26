import os
import glob
import pandas as pd
import numpy as np
import pickle
import requests
import warnings
from datetime import datetime

# Ignorar FutureWarnings para limpiar la salida
warnings.simplefilter(action='ignore', category=FutureWarning)

# —————————————————————————
# 0) Rutas fijas
# —————————————————————————

BASE_DIR = r"D:\PREDICCION URGENCIAS\MODELO PREDICCION DE URGENCIAS"
MODEL_PATH = glob.glob(os.path.join(BASE_DIR, "*.pkl"))[0]
HIST_PATH  = glob.glob(os.path.join(BASE_DIR, "*.xls*"))[0]
NEW_DATA_PATH = os.path.join(BASE_DIR, "Nuevas_Predicciones.xlsx")

# —————————————————————————
# 1) Carga de modelo y datasets
# —————————————————————————

def cargar_modelo():
    """Carga el modelo desde MODEL_PATH."""
    modelo = pickle.load(open(MODEL_PATH, "rb"))
    print(f"✅ Modelo cargado desde: {MODEL_PATH}")
    return modelo

def cargar_historico():
    """
    Carga el histórico principal de urgencias desde HIST_PATH.
    Si la columna 'Predicción' no existe se añade.
    """
    df = pd.read_excel(HIST_PATH, parse_dates=[0], index_col=0)
    if "Predicción" not in df.columns:
        df["Predicción"] = np.nan
    print(f"✅ Histórico cargado desde: {HIST_PATH} ({len(df)} registros)")
    return df

def cargar_nuevas_predicciones(modelo):
    """
    Carga (o crea) el archivo en el que se almacenarán las nuevas predicciones.
    Las columnas serán:
      "Fecha" + las variables de entrada que espera el modelo + 
      ["Predicción", "Limite_Inferior", "Limite_Superior", "Valor_Real"]
    """
    if os.path.exists(NEW_DATA_PATH):
        df = pd.read_excel(NEW_DATA_PATH, parse_dates=[0], index_col=0)
    else:
        columnas = ["Fecha"] + list(modelo.get_booster().feature_names) + ["Predicción", "Limite_Inferior", "Limite_Superior", "Valor_Real"]
        df = pd.DataFrame(columns=columnas)
    return df

# —————————————————————————
# 2) Fetch de datos meteorológicos AEMET
# —————————————————————————

API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJkci5zYW1wZWRyb0BnbWFpbC5jb20iLCJqdGkiOiJkMjdkOTM0My1mMWUzLTRjZmUtOTQ4Ni1jNjk2ODZkM2ExZjkiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTc0NTUwNjQ2NywidXNlcklkIjoiZDI3ZDkzNDMtZjFlMy00Y2ZlLTk0ODYtYzY5Njg2ZDNhMWY5Iiwicm9sZSI6IiJ9.6TJxvu7SKU8Mgpu-qQedTRCjv-VzIIPRADlEuHGE9BA"  # Reemplaza con tu API Key
LOCALITY_CODE = "20071"      # Código para Tolosa

def fetch_forecast():
    """Obtiene los datos meteorológicos de AEMET."""
    base = "https://opendata.aemet.es/opendata/api"
    endpoint = f"{base}/prediccion/especifica/municipio/diaria/{LOCALITY_CODE}"
    r1 = requests.get(endpoint, params={"api_key": API_KEY})
    r1.raise_for_status()
    data_url = r1.json()["datos"]
    r2 = requests.get(data_url, params={"api_key": API_KEY})
    r2.raise_for_status()
    raw = r2.json()
    if isinstance(raw, list):
        raw = raw[0]
    return raw["prediccion"]["dia"]

def compute_meteo_averages(day):
    """Calcula promedios de variables meteorológicas del día."""
    def safe_avg(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "Temp_Media":        (day["temperatura"]["maxima"] + day["temperatura"]["minima"]) / 2,
        "Temp_Max":          day["temperatura"]["maxima"],
        "Temp_Min":          day["temperatura"]["minima"],
        "Precipitacion":     safe_avg([p["value"] for p in day.get("probPrecipitacion", []) if isinstance(p.get("value"), (int, float))]),
        "Vel_Media_Viento":  safe_avg([v["velocidad"] for v in day.get("viento", []) if isinstance(v.get("velocidad"), (int, float))]),
        "Racha_Maxima":      safe_avg([r.get("value") for r in day.get("rachaMax", []) if isinstance(r.get("value"), (int, float))]),
        "Hum_Rel_Med":       safe_avg([h["value"] for h in day.get("humedadRelativa", {}).get("dato", [])
                                       if isinstance(h.get("value"), (int, float))])
    }

# —————————————————————————
# 3) Funciones para nuevas predicciones y actualización
# —————————————————————————

def realizar_prediccion(modelo, nuevas_predicciones, meteo_days):
    """
    Realiza una nueva predicción.
    Se solicitan la fecha y variables de entrada, se obtiene la predicción y se guardan
    junto con "Predicción", "Limite_Inferior", "Limite_Superior" y "Valor_Real" (inicialmente NaN)
    en el DataFrame de nuevas predicciones.
    """
    s = input("Introduce la fecha para la predicción (DD/MM/YYYY) o 'salir' para volver al menú: ")
    if s.lower() == "salir":
        return nuevas_predicciones
    try:
        fecha = datetime.strptime(s, "%d/%m/%Y").date()
    except ValueError:
        print("❌ Formato inválido.")
        return nuevas_predicciones

    # Buscar los datos meteorológicos para la fecha indicada
    day_m = next((d for d in meteo_days if datetime.fromisoformat(d["fecha"]).date() == fecha), None)
    if not day_m:
        print("❌ Fuera de rango meteorológico.")
        return nuevas_predicciones
    meteo = compute_meteo_averages(day_m)

    # Solicitar valores binarios (0=No, 1=Sí)
    carn   = int(input("Carnavales? (0=No, 1=Sí): "))
    ss     = int(input("Semana Santa? (0=No, 1=Sí): "))
    jugado = int(input("Partido en Donosti? (0=No, 1=Sí): "))
    sel    = int(input("Selección Española? (0=No, 1=Sí): "))

    # Ensamblar las variables usadas
    data = {
        "Fecha": fecha,
        **meteo,
        "Carnavales": carn,
        "Semana Santa": ss,
        "Jugado en Donosti": jugado,
        "Seleccion Española": sel
    }

    # Crear el DataFrame para la predicción usando las columnas que espera el modelo
    df_pred = pd.Series(data).reindex(modelo.get_booster().feature_names).fillna(0).infer_objects(copy=False).to_frame().T
    pred = modelo.predict(df_pred)[0]
    
    # Redondear la predicción a entero y calcular el margen (por ejemplo, 10% del valor)
    pred_int = int(round(pred))
    margen = int(round(pred_int * 0.1))
    limite_inferior = pred_int - margen
    limite_superior = pred_int + margen

    print(f"\n🎯 Predicción para {fecha.strftime('%d/%m/%Y')}: {pred_int} ± {margen} (Rango: {limite_inferior} - {limite_superior})")

    # Guardar en el registro la predicción redondeada y el rango calculado
    data["Predicción"] = pred_int
    data["Limite_Inferior"] = limite_inferior
    data["Limite_Superior"] = limite_superior
    data["Valor_Real"] = np.nan

    nuevas_predicciones = pd.concat([nuevas_predicciones, pd.DataFrame([data])], ignore_index=True)
    nuevas_predicciones.to_excel(NEW_DATA_PATH)
    print("✔️ Predicción y variables guardadas en Nuevas_Predicciones.xlsx.")
    return nuevas_predicciones

def ingresar_valor_real(nuevas_predicciones):
    """
    Muestra las predicciones registradas sin valor real y permite ingresar el valor observado.
    Se actualiza el DataFrame de nuevas predicciones y se guarda en el archivo Excel.
    """
    pendientes = nuevas_predicciones[nuevas_predicciones["Valor_Real"].isna()]
    if pendientes.empty:
        print("ℹ️ No hay predicciones pendientes de actualización de valor real.")
        return nuevas_predicciones

    print("\n📊 Predicciones pendientes (Fecha y Predicción):")
    resumen = pendientes[["Fecha", "Predicción", "Limite_Inferior", "Limite_Superior"]]
    print(resumen.to_string(index=False))
    
    s = input("\nIntroduce la fecha (DD/MM/YYYY) para la que quieres ingresar el valor real o 'salir': ")
    if s.lower() == "salir":
        return nuevas_predicciones
    try:
        fecha_sel = datetime.strptime(s, "%d/%m/%Y").date()
    except ValueError:
        print("❌ Formato inválido.")
        return nuevas_predicciones

    idx = nuevas_predicciones[(nuevas_predicciones["Fecha"] == pd.to_datetime(fecha_sel)) &
                              (nuevas_predicciones["Valor_Real"].isna())].index
    if idx.empty:
        print("❌ No se encontró un registro pendiente para esa fecha.")
        return nuevas_predicciones

    try:
        valor = float(input(f"Ingrese el valor real observado para {fecha_sel.strftime('%d/%m/%Y')}: "))
    except ValueError:
        print("❌ Valor inválido.")
        return nuevas_predicciones

    nuevas_predicciones.loc[idx, "Valor_Real"] = valor
    nuevas_predicciones.to_excel(NEW_DATA_PATH)
    print("✔️ Valor real actualizado en Nuevas_Predicciones.xlsx.")
    return nuevas_predicciones

def chequear_actualizacion(nuevas_predicciones, historial):
    """
    Si se han completado 50 o más registros (con Valor_Real no NaN),
    se avisa al usuario y se permite actualizar el dataset principal.
    """
    completados = nuevas_predicciones["Valor_Real"].notna().sum()
    if completados >= 50:
        autorizacion = input(f"\n⚠️ Se han completado {completados} registros. ¿Actualizar el dataset principal y reajustar el modelo? (S/N): ").strip().lower()
        if autorizacion == "s":
            historial = pd.concat([historial, nuevas_predicciones], ignore_index=True)
            historial.to_excel(HIST_PATH)
            print("✅ Dataset principal actualizado. Se deberá reentrenar el modelo con los nuevos datos.")
            columnas = nuevas_predicciones.columns
            nuevas_predicciones = pd.DataFrame(columns=columnas)
            nuevas_predicciones.to_excel(NEW_DATA_PATH)
    return historial, nuevas_predicciones

# —————————————————————————
# 4) Menú principal
# —————————————————————————

def main():
    modelo = cargar_modelo()
    historial = cargar_historico()
    nuevas_predicciones = cargar_nuevas_predicciones(modelo)
    meteo_days = fetch_forecast()

    while True:
        print("\n🔵 Menú Principal:")
        print("1️⃣ Realizar nueva predicción")
        print("2️⃣ Ingresar valor real para predicciones pendientes")
        print("3️⃣ Salir")

        opcion = input("Selecciona una opción (1/2/3): ").strip()
        if opcion == "1":
            nuevas_predicciones = realizar_prediccion(modelo, nuevas_predicciones, meteo_days)
            historial, nuevas_predicciones = chequear_actualizacion(nuevas_predicciones, historial)
        elif opcion == "2":
            nuevas_predicciones = ingresar_valor_real(nuevas_predicciones)
            historial, nuevas_predicciones = chequear_actualizacion(nuevas_predicciones, historial)
        elif opcion == "3":
            print("✅ Fin.")
            break
        else:
            print("❌ Opción inválida. Intente de nuevo.")

if __name__ == "__main__":
    main()


