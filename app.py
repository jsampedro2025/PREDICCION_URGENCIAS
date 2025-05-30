import os
import glob
import pandas as pd
import numpy as np
import pickle
import requests
import streamlit as st
from datetime import datetime
import warnings

# Suprimir FutureWarnings para mantener la salida limpia
warnings.simplefilter("ignore", category=FutureWarning)

# ---------- CONFIGURACIÓN DE RUTAS ----------
BASE_DIR = r"D:\PREDICCION URGENCIAS\MODELO PREDICCION DE URGENCIAS"
MODEL_PATH = glob.glob(os.path.join(BASE_DIR, "*.pkl"))[0]
HIST_PATH = glob.glob(os.path.join(BASE_DIR, "*.xls*"))[0]
NEW_DATA_PATH = os.path.join(BASE_DIR, "Nuevas_Predicciones.xlsx")

# ---------- FUNCIONES CON CACHEO ----------
@st.cache_resource
def cargar_modelo():
    modelo = pickle.load(open(MODEL_PATH, "rb"))
    return modelo

@st.cache_data
def cargar_historico():
    df = pd.read_excel(HIST_PATH, parse_dates=[0], index_col=0)
    if "Predicción" not in df.columns:
        df["Predicción"] = np.nan
    return df

def cargar_nuevas_predicciones(modelo):
    if os.path.exists(NEW_DATA_PATH):
        df = pd.read_excel(NEW_DATA_PATH, parse_dates=[0], index_col=0)
    else:
        columnas = ["Fecha"] + list(modelo.get_booster().feature_names) + [
            "Predicción", "Limite_Inferior", "Limite_Superior", "Valor_Real"]
        df = pd.DataFrame(columns=columnas)
    return df

@st.cache_data(ttl=3600)
def fetch_forecast():
    # Reemplaza "TU_API_KEY_AQUI" con tu API key válida de AEMET.
    API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJkci5zYW1wZWRyb0BnbWFpbC5jb20iLCJqdGkiOiJkMjdkOTM0My1mMWUzLTRjZmUtOTQ4Ni1jNjk2ODZkM2ExZjkiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTc0NTUwNjQ2NywidXNlcklkIjoiZDI3ZDkzNDMtZjFlMy00Y2ZlLTk0ODYtYzY5Njg2ZDNhMWY5Iiwicm9sZSI6IiJ9.6TJxvu7SKU8Mgpu-qQedTRCjv-VzIIPRADlEuHGE9BA"
    LOCALITY_CODE = "20071"  # Tolosa
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
    def safe_avg(values):
        return sum(values) / len(values) if values else 0.0
    return {
        "Temp_Media": (day["temperatura"]["maxima"] + day["temperatura"]["minima"]) / 2,
        "Temp_Max": day["temperatura"]["maxima"],
        "Temp_Min": day["temperatura"]["minima"],
        "Precipitacion": safe_avg([p["value"] for p in day.get("probPrecipitacion", [])
                                  if isinstance(p.get("value"), (int, float))]),
        "Vel_Media_Viento": safe_avg([v["velocidad"] for v in day.get("viento", [])
                                      if isinstance(v.get("velocidad"), (int, float))]),
        "Racha_Maxima": safe_avg([r.get("value") for r in day.get("rachaMax", [])
                                  if isinstance(r.get("value"), (int, float))]),
        "Hum_Rel_Med": safe_avg([h["value"] for h in day.get("humedadRelativa", {})
                                 .get("dato", []) if isinstance(h.get("value"), (int, float))])
    }

# ---------- FUNCIONES DE INTERFAZ ----------
def nueva_prediccion(modelo, nuevas_predicciones, meteo_days):
    st.subheader("Nueva Predicción")
    
    # Extraer las fechas disponibles desde los datos de AEMET
    forecast_dates = [datetime.fromisoformat(d["fecha"]).date() for d in meteo_days]
    if forecast_dates:
        min_date = min(forecast_dates)
        max_date = max(forecast_dates)
        st.info(f"El rango de fechas disponible es: {min_date.strftime('%d/%m/%Y')} a {max_date.strftime('%d/%m/%Y')}")
    else:
        st.error("No se encontraron datos meteorológicos disponibles.")
        return nuevas_predicciones

    # Selección de fecha restringida al rango disponible
    fecha = st.date_input("Selecciona la fecha de la predicción", value=min_date, min_value=min_date, max_value=max_date)
    if fecha not in forecast_dates:
        st.error("La fecha seleccionada no tiene datos meteorológicos disponibles.")
        return nuevas_predicciones

    # Variables binarias básicas
    carn = st.radio("Carnavales? (0=No, 1=Sí)", options=[0, 1], index=0, format_func=lambda x: "No" if x==0 else "Sí")
    ss = st.radio("Semana Santa? (0=No, 1=Sí)", options=[0, 1], index=0, format_func=lambda x: "No" if x==0 else "Sí")
    jugado = st.radio("Partido en Donosti? (0=No, 1=Sí)", options=[0, 1], index=0, format_func=lambda x: "No" if x==0 else "Sí")
    sel = st.radio("Selección Española? (0=No, 1=Sí)", options=[0, 1], index=0, format_func=lambda x: "No" if x==0 else "Sí")
    
    # Variables adicionales de epidemias
    gripe = st.radio("¿Hay casos de Gripe? (0=No, 1=Sí)", options=[0, 1], index=0, format_func=lambda x: "No" if x==0 else "Sí")
    covid = st.radio("¿Hay casos de Covid? (0=No, 1=Sí)", options=[0, 1], index=0, format_func=lambda x: "No" if x==0 else "Sí")
    brote = st.radio("¿Existe un brote epidémico? (0=No, 1=Sí)", options=[0, 1], index=0, format_func=lambda x: "No" if x==0 else "Sí")
    
    # Armar el diccionario con todas las variables de entrada
    # Las variables meteorológicas se computan a partir de los datos de AEMET.
    day_m = next((d for d in meteo_days if datetime.fromisoformat(d["fecha"]).date() == fecha), None)
    meteo = compute_meteo_averages(day_m)
    data = {
        "Fecha": fecha,
        **meteo,
        "Carnavales": carn,
        "Semana Santa": ss,
        "Jugado en Donosti": jugado,
        "Seleccion Española": sel,
        "Gripe": gripe,
        "Covid": covid,
        "Brote_Epidemico": brote
    }
    
    # Mostrar al usuario los valores que se usarán para la predicción
    st.markdown("**Variables de entrada para la predicción:**")
    st.table(pd.DataFrame(data, index=[0]))
    
    if st.button("Realizar Predicción"):
        # Se reindexa según las columnas que el modelo espera.
        df_pred = pd.Series(data).reindex(modelo.get_booster().feature_names).fillna(0).to_frame().T
        pred = modelo.predict(df_pred)[0]
        pred_int = int(round(pred))      # Predicción sin decimales
        margen = int(round(pred_int * 0.1))  # Margen del 10%
        lim_inf = pred_int - margen
        lim_sup = pred_int + margen

        st.success(f"Predicción: {pred_int} ± {margen} (Rango: {lim_inf} - {lim_sup})")
        data["Predicción"] = pred_int
        data["Limite_Inferior"] = lim_inf
        data["Limite_Superior"] = lim_sup
        data["Valor_Real"] = np.nan

        nuevas_predicciones = pd.concat([nuevas_predicciones, pd.DataFrame([data])], ignore_index=True)
        nuevas_predicciones.to_excel(NEW_DATA_PATH)
        st.info("Registro guardado en Nuevas_Predicciones.xlsx.")
    return nuevas_predicciones

def ingresar_valor_real(nuevas_predicciones):
    st.subheader("Ingresar/Ver Valor Real")
    pendientes = nuevas_predicciones[nuevas_predicciones["Valor_Real"].isna()]
    if pendientes.empty:
        st.info("No hay predicciones pendientes de actualizar.")
        return nuevas_predicciones
    st.dataframe(pendientes[["Fecha", "Predicción", "Limite_Inferior", "Limite_Superior"]])
    fecha_str = st.text_input("Ingresa la fecha (DD/MM/YYYY) para actualizar el valor real")
    if fecha_str:
        try:
            fecha_sel = datetime.strptime(fecha_str, "%d/%m/%Y").date()
        except ValueError:
            st.error("Formato de fecha inválido.")
            return nuevas_predicciones
        valor = st.number_input(f"Ingresa el valor observado para {fecha_sel.strftime('%d/%m/%Y')}", value=0.0)
        if st.button("Actualizar Valor Real"):
            idx = nuevas_predicciones[
                (nuevas_predicciones["Fecha"] == pd.to_datetime(fecha_sel)) &
                (nuevas_predicciones["Valor_Real"].isna())
            ].index
            if idx.empty:
                st.error("No se encontró registro pendiente para esa fecha.")
            else:
                nuevas_predicciones.loc[idx, "Valor_Real"] = valor
                nuevas_predicciones.to_excel(NEW_DATA_PATH)
                st.success("Valor real actualizado en Nuevas_Predicciones.xlsx.")
    return nuevas_predicciones

def actualizar_dataset(nuevas_predicciones, historial):
    st.subheader("Actualizar Dataset Principal")
    completados = nuevas_predicciones["Valor_Real"].notna().sum()
    st.write(f"Registros completados: {completados}")
    if completados < 50:
        st.warning("Se requieren al menos 50 registros completos para actualizar el dataset.")
        return historial, nuevas_predicciones
    if st.button("Actualizar Dataset"):
        historial = pd.concat([historial, nuevas_predicciones], ignore_index=True)
        historial.to_excel(HIST_PATH)
        st.success("Dataset principal actualizado. Reentrena el modelo con los nuevos datos.")
        columnas = nuevas_predicciones.columns
        nuevas_predicciones = pd.DataFrame(columns=columnas)
        nuevas_predicciones.to_excel(NEW_DATA_PATH)
    return historial, nuevas_predicciones

# ---------- INTERFAZ PRINCIPAL ----------
def main():
    st.title("Sistema de Predicción de Urgencias")
    st.sidebar.title("Menú")
    opcion = st.sidebar.selectbox("Seleccione acción:", 
                                  ["Nueva Predicción", "Ingresar/Ver Valor Real", "Actualizar Dataset"])
    
    modelo = cargar_modelo()
    historial = cargar_historico()
    nuevas_predicciones = cargar_nuevas_predicciones(modelo)
    meteo_days = fetch_forecast()
    
    if opcion == "Nueva Predicción":
        nuevas_predicciones = nueva_prediccion(modelo, nuevas_predicciones, meteo_days)
    elif opcion == "Ingresar/Ver Valor Real":
        nuevas_predicciones = ingresar_valor_real(nuevas_predicciones)
        st.subheader("Predicciones Pendientes")
        st.dataframe(
            nuevas_predicciones[nuevas_predicciones["Valor_Real"].isna()][
                ["Fecha", "Predicción", "Limite_Inferior", "Limite_Superior"]
            ]
        )
    elif opcion == "Actualizar Dataset":
        historial, nuevas_predicciones = actualizar_dataset(nuevas_predicciones, historial)
    
    st.sidebar.info("La aplicación se ejecuta en la nube y es accesible desde cualquier dispositivo.")

if __name__ == "__main__":
    main()

