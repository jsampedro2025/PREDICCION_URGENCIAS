import os
import pandas as pd
import numpy as np
import pickle
import requests
import streamlit as st
from datetime import datetime
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

# ---------- CONFIGURACIÓN INICIAL DE LA PÁGINA ----------
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# ---------- RUTAS ----------
# BASE_DIR:  ¡El directorio actual! (La raíz del repositorio)
BASE_DIR = "."

# MODEL_PATH:  Ruta al archivo .pkl (nombre exacto)
MODEL_PATH = os.path.join(BASE_DIR, "modelo_prediccion_urgencias.pkl")

# HIST_PATH: Ruta al archivo de datos históricos (nombre exacto o patrón)
HIST_PATH = os.path.join(BASE_DIR, "DATASET_MEJORADO.xlsx")

# NEW_DATA_PATH: Ruta al archivo de nuevas predicciones (nombre exacto)
NEW_DATA_PATH = os.path.join(BASE_DIR, "Nuevas_Predicciones.xlsx"
# ---------- FUNCIONES CON CACHEO ----------
@st.cache_resource
def cargar_modelo():
    try:
        modelo = pickle.load(open(MODEL_PATH, "rb"))
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

@st.cache_data
def cargar_historico():
    try:
        df = pd.read_excel(HIST_PATH, parse_dates=[0], index_col=0)
        if "Predicción" not in df.columns:
            df["Predicción"] = np.nan
        return df
    except Exception as e:
        st.error(f"Error al cargar datos históricos: {e}")
        return pd.DataFrame()

def cargar_nuevas_predicciones(modelo):
    try:
        if os.path.exists(NEW_DATA_PATH):
            df = pd.read_excel(NEW_DATA_PATH, parse_dates=[0], index_col=0)
        else:
            columnas = ["Fecha"] + (list(modelo.get_booster().feature_names) if modelo else []) + [
                "Predicción", "Limite_Inferior", "Limite_Superior", "Valor_Real"]
            df = pd.DataFrame(columns=columnas)
        return df
    except Exception as e:
        st.error(f"Error cargando nuevas predicciones: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_forecast():
    API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJkci5zYW1wZWRyb0BnbWFpbC5jb20iLCJqdGkiOiI3M2U3MmQxOC1jNTMyLTRkM2UtYmYwOC1hMGJhNWM4YWRmOGMiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTczODc4NDc1NCwidXNlcklkIjoiNzNlNzJkMTgtYzUzMi00ZDNlLWJmMDgtYTBiYTVjOGFkZjhjIiwicm9sZSI6IiJ9.aIbLKEsLbb5DmtaYyxStWLQipyqn_v6YynXbUIjQI_c"
    LOCALITY_CODE = "20071"  # Código INE de Tolosa
    base = "https://opendata.aemet.es/opendata/api"
    endpoint = f"{base}/prediccion/especifica/municipio/diaria/{LOCALITY_CODE}"
    try:
        r1 = requests.get(endpoint, params={"api_key": API_KEY})
        r1.raise_for_status()
        data_url = r1.json().get("datos")
        r2 = requests.get(data_url)
        r2.raise_for_status()
        raw = r2.json()
        if isinstance(raw, list):
            raw = raw[0]
        return raw.get("prediccion", {}).get("dia", [])
    except Exception as e:
        st.error(f"Error obteniendo pronóstico: {e}")
        return []

def compute_meteo_averages(day):
    def safe_avg(values):
        return sum(values) / len(values) if values else 0.0
    return {
        "Temp_Media": (day.get("temperatura", {}).get("maxima", 0) + day.get("temperatura", {}).get("minima", 0)) / 2,
        "Temp_Max": day.get("temperatura", {}).get("maxima", 0),
        "Temp_Min": day.get("temperatura", {}).get("minima", 0),
        "Precipitacion": safe_avg([p.get("value") for p in day.get("probPrecipitacion", []) if isinstance(p.get("value"), (int, float))]),
        "Vel_Media_Viento": safe_avg([v.get("velocidad") for v in day.get("viento", []) if isinstance(v.get("velocidad"), (int, float))]),
        "Racha_Maxima": safe_avg([r.get("value") for r in day.get("rachaMax", []) if isinstance(r.get("value"), (int, float))]),
        "Hum_Rel_Med": safe_avg([h.get("value") for h in day.get("humedadRelativa", {}).get("dato", []) if isinstance(h.get("value"), (int, float))])
    }

# ---------- FUNCIONES DE INTERFAZ ----------
def nueva_prediccion(modelo, nuevas_predicciones, meteo_days):
    st.subheader("🌤️ Nueva Predicción")

    if not meteo_days:
        st.error("Datos meteorológicos no disponibles.")
        return nuevas_predicciones

    forecast_dates = [datetime.fromisoformat(d["fecha"]).date() for d in meteo_days if "fecha" in d]
    fecha = st.date_input("Selecciona la fecha", value=min(forecast_dates), min_value=min(forecast_dates), max_value=max(forecast_dates))

    if fecha not in forecast_dates:
        st.error("Fecha no disponible.")
        return nuevas_predicciones

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        carn = st.radio("Carnavales?", [0, 1], index=0)
        ss = st.radio("Semana Santa?", [0, 1], index=0)
    with col2:
        jugado = st.radio("Partido en Donosti?", [0, 1], index=0)
        sel = st.radio("Selección Española?", [0, 1], index=0)
    with col3:
        gripe = st.radio("Casos de Gripe?", [0, 1], index=0)
        covid = st.radio("Casos de Covid?", [0, 1], index=0)
    with col4:
        brote = st.radio("Brote epidémico?", [0, 1], index=0)

    day_m = next((d for d in meteo_days if "fecha" in d and datetime.fromisoformat(d["fecha"]).date() == fecha), None)
    if not day_m:
        st.error("No se encontraron datos meteorológicos.")
        return nuevas_predicciones

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

    st.markdown("### Variables utilizadas")
    st.table(pd.DataFrame(data, index=[0]))

    if st.button("Realizar Predicción") and modelo is not None:
        try:
            df_pred = pd.Series(data).reindex(modelo.get_booster().feature_names).fillna(0).to_frame().T
            pred = modelo.predict(df_pred)[0]
            pred_int = int(round(pred))
            margen = int(round(pred_int * 0.1))
            lim_inf = int(pred_int - margen)
            lim_sup = int(pred_int + margen)

            st.success(f"Predicción: {pred_int} (±{margen}) → Rango: [{lim_inf} - {lim_sup}]")
            data.update({"Predicción": pred_int, "Limite_Inferior": lim_inf, "Limite_Superior": lim_sup, "Valor_Real": np.nan})

            nuevas_predicciones = pd.concat([nuevas_predicciones, pd.DataFrame([data])], ignore_index=True)
            nuevas_predicciones.to_excel(NEW_DATA_PATH, index=False)
            st.info("Predicción guardada.")
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
    return nuevas_predicciones

def ingresar_valor_real(nuevas_predicciones):
    st.subheader("✏️ Ingresar Valor Real")
    pendientes = nuevas_predicciones[nuevas_predicciones["Valor_Real"].isna()].copy()

    if pendientes.empty:
        st.info("Todas las predicciones tienen valor real.")
        return nuevas_predicciones

    edited = st.data_editor(
        pendientes[["Fecha", "Predicción", "Limite_Inferior", "Limite_Superior"]],
        num_rows="dynamic"
    )

    if st.button("Guardar Valores Reales"):
        try:
            nuevas_predicciones.update(edited)
            nuevas_predicciones.to_excel(NEW_DATA_PATH, index=False)
            st.success("Valores reales actualizados.")
        except Exception as e:
            st.error(f"Error actualizando valores: {e}")
    return nuevas_predicciones

# ---------- MAIN ----------
def main():
    st.title("📈 Predicción de Urgencias - Tolosa")

    modelo = cargar_modelo()
    historico = cargar_historico()
    nuevas_predicciones = cargar_nuevas_predicciones(modelo)
    meteo_days = fetch_forecast()

    tabs = st.tabs(["Realizar Predicción", "Ingresar Valor Real", "Histórico de Datos"])
    with tabs[0]:
        nuevas_predicciones = nueva_prediccion(modelo, nuevas_predicciones, meteo_days)
    with tabs[1]:
        nuevas_predicciones = ingresar_valor_real(nuevas_predicciones)
    with tabs[2]:
        st.dataframe(historico)

if __name__ == "__main__":
    main()

