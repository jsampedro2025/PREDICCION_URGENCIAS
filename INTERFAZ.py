import os
import glob
import pandas as pd
import numpy as np
import pickle
import requests
import streamlit as st
from datetime import datetime
import warnings

# No usaremos components.iframe para la página si se bloquea; en su lugar mostraremos un enlace.
# Suprimir FutureWarnings para mantener la salida limpia
warnings.simplefilter("ignore", category=FutureWarning)

# ---------- CONFIGURACIÓN INICIAL DE LA PÁGINA ----------
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# ---------- CONFIGURACIÓN DE RUTAS ----------
# BASE_DIR:  ¡El directorio actual! (La raíz del repositorio)
BASE_DIR = "."

# MODEL_PATH:  Ruta al archivo .pkl (nombre exacto)
MODEL_PATH = os.path.join(BASE_DIR, "modelo_prediccion_urgencias.pkl")

# HIST_PATH: Ruta al archivo de datos históricos (nombre exacto o patrón)
HIST_PATH = os.path.join(BASE_DIR, "DATASET_MEJORADO.xlsx")

# NEW_DATA_PATH: Ruta al archivo de nuevas predicciones (nombre exacto)
NEW_DATA_PATH = os.path.join(BASE_DIR, "Nuevas_Predicciones.xlsx")

# ---------- FUNCIONES CON CACHEO ----------
@st.cache_resource
def cargar_modelo():
    try:
        modelo = pickle.load(open(MODEL_PATH, "rb"))
        print(f"Modelo cargado exitosamente desde: {MODEL_PATH}")
        return modelo
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en: {MODEL_PATH}")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

@st.cache_data
def cargar_historico():
    try:
        df = pd.read_excel(HIST_PATH, parse_dates=[0], index_col=0)
        if "Predicción" not in df.columns:
            df["Predicción"] = np.nan
        print(f"Datos históricos cargados exitosamente desde: {HIST_PATH}")
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de datos históricos en: {HIST_PATH}")
        return pd.DataFrame() # Retornar un DataFrame vacío para evitar errores posteriores
    except Exception as e:
        st.error(f"Error al cargar los datos históricos: {e}")
        return pd.DataFrame()

def cargar_nuevas_predicciones(modelo):
    try:
        if os.path.exists(NEW_DATA_PATH):
            df = pd.read_excel(NEW_DATA_PATH, parse_dates=[0], index_col=0)
            print(f"Nuevas predicciones cargadas desde: {NEW_DATA_PATH}")
        else:
            columnas = ["Fecha"] + (list(modelo.get_booster().feature_names) if modelo else []) + [
                "Predicción", "Limite_Inferior", "Limite_Superior", "Valor_Real"]
            df = pd.DataFrame(columns=columnas)
            print(f"Archivo de nuevas predicciones no encontrado. Se creó un nuevo DataFrame.")
        return df
    except Exception as e:
        st.error(f"Error al cargar o crear el archivo de nuevas predicciones: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_forecast():
    API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJkci5zYW1wZWRyb0BnbWFpbC5jb20iLCJqdGkiOiJkMjdkOTM0My1mMWUzLTRjZmUtOTQ4Ni1jNjk2ODZkM2ExZjkiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTc0NTUwNjQ2NywidXNlcklkIjoiZDI3ZDkzNDMtZjFlMy00Y2ZlLTk0ODYtYzY5Njg2ZDNhMWY5Iiwicm9sZSI6IiJ9.6TJxvu7SKU8Mgpu-qQedTRCjv-VzIIPRADlEuHGE9BA"
    LOCALITY_CODE = "20071"  # Tolosa
    base = "https://opendata.aemet.es/opendata/api"
    endpoint = f"{base}/prediccion/especifica/municipio/diaria/{LOCALITY_CODE}"
    try:
        r1 = requests.get(endpoint, params={"api_key": API_KEY})
        r1.raise_for_status()
        data_url = r1.json().get("datos")
        if not data_url:
            st.error("Error: La API de AEMET no devolvió la URL de los datos.")
            return []
        r2 = requests.get(data_url, params={"api_key": API_KEY})
        r2.raise_for_status()
        raw = r2.json()
        if isinstance(raw, list):
            raw = raw[0]
        print("Datos meteorológicos obtenidos de AEMET.")
        return raw.get("prediccion", {}).get("dia", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con la API de AEMET: {e}")
        return []
    except KeyError:
        st.error("Error: Formato inesperado en la respuesta de la API de AEMET.")
        return []
    except Exception as e:
        st.error(f"Error inesperado al obtener el pronóstico: {e}")
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
    st.subheader("Nueva Predicción")

    if not meteo_days:
        st.error("No se pudieron obtener los datos meteorológicos. La predicción no está disponible.")
        return nuevas_predicciones

    # Extraer fechas disponibles de AEMET
    forecast_dates = [datetime.fromisoformat(d["fecha"]).date() for d in meteo_days if "fecha" in d]
    if not forecast_dates:
        st.error("No se encontraron fechas válidas en los datos meteorológicos.")
        return nuevas_predicciones

    min_date = min(forecast_dates)
    max_date = max(forecast_dates)
    st.info(f"El rango de fechas disponible es: {min_date.strftime('%d/%m/%Y')} a {max_date.strftime('%d/%m/%Y')}")

    # Selección de fecha
    fecha = st.date_input("Selecciona la fecha de la predicción",
                            value=min_date, min_value=min_date, max_value=max_date)
    if fecha not in forecast_dates:
        st.error("La fecha seleccionada no tiene datos meteorológicos disponibles.")
        return nuevas_predicciones

    # Crear columnas para las variables
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        carn = st.radio("Carnavales? (0=No, 1=Sí)", options=[0, 1],
                        index=0, format_func=lambda x: "No" if x == 0 else "Sí")
        ss = st.radio("Semana Santa? (0=No, 1=Sí)", options=[0, 1],
                      index=0, format_func=lambda x: "No" if x == 0 else "Sí")

    with col2:
        jugado = st.radio("Partido en Donosti? (0=No, 1=Sí)", options=[0, 1],
                          index=0, format_func=lambda x: "No" if x == 0 else "Sí")
        sel = st.radio("Selección Española? (0=No, 1=Sí)", options=[0, 1],
                       index=0, format_func=lambda x: "No" if x == 0 else "Sí")

    with col3:
        gripe = st.radio("¿Hay casos de Gripe? (0=No, 1=Sí)", options=[0, 1],
                         index=0, format_func=lambda x: "No" if x == 0 else "Sí")
        covid = st.radio("¿Hay casos de Covid? (0=No, 1=Sí)", options=[0, 1],
                         index=0, format_func=lambda x: "No" if x == 0 else "Sí")

    with col4:
        brote = st.radio("¿Existe un brote epidémico? (0=No, 1=Sí)", options=[0, 1],
                         index=0, format_func=lambda x: "No" if x == 0 else "Sí")

    # Obtener variables meteorológicas para la fecha seleccionada
    day_m = next((d for d in meteo_days if "fecha" in d and datetime.fromisoformat(d["fecha"]).date() == fecha), None)
    if day_m:
        meteo = compute_meteo_averages(day_m)
    else:
        st.error("No se encontraron datos meteorológicos para la fecha seleccionada.")
        return nuevas_predicciones

    # Construir el diccionario con todas las variables de entrada
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

    # Mostrar de forma clara al usuario todas las variables que se usarán para la predicción
    st.markdown("**Variables usadas para la predicción:**")
    st.table(pd.DataFrame(data, index=[0]))

    if st.button("Realizar Predicción") and modelo is not None:
        try:
            df_pred = pd.Series(data).reindex(modelo.get_booster().feature_names).fillna(0).to_frame().T
            pred = modelo.predict(df_pred)[0]
            pred_int = int(round(pred))  # Predicción sin decimales
            margen = int(round(pred_int * 0.1))  # Margen del 10%
            lim_inf = int(round(pred_int - margen))
            lim_sup = int(round(pred_int + margen))

            st.success(f"Predicción: {pred_int} ± {margen} (Rango: {lim_inf} - {lim_sup})")
            data["Predicción"] = pred_int
            data["Limite_Inferior"] = lim_inf
            data["Limite_Superior"] = lim_sup
            data["Valor_Real"] = np.nan

            print(f"Guardando nueva predicción en: {NEW_DATA_PATH}")
            print("Contenido de nuevas_predicciones justo antes de guardar:")
            print(nuevas_predicciones)

            nuevas_predicciones = pd.concat([nuevas_predicciones, pd.DataFrame([data])], ignore_index=True)

            print("Contenido de nuevas_predicciones DESPUÉS de concatenar:")
            print(nuevas_predicciones)

            nuevas_predicciones.to_excel(NEW_DATA_PATH, index=False) # Añadido index=False
            st.info("Registro guardado en Nuevas_Predicciones.xlsx.")
            return nuevas_predicciones # Asegurarse de retornar el DataFrame actualizado
        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")
            return nuevas_predicciones
    elif modelo is None:
        st.warning("El modelo no se ha cargado correctamente. La predicción no está disponible.")
        return nuevas_predicciones
    return nuevas_predicciones

def ingresar_valor_real(nuevas_predicciones):
    st.subheader("Ingresar Valor Real")
    pendientes = nuevas_predicciones[nuevas_predicciones["Valor_Real"].isna()].copy()

    if pendientes.empty:
        st.info("¡Enhorabuena! Todas las predicciones tienen su valor observado registrado.")
        return nuevas_predicciones

    pendientes["Valor Real"] = np.nan  # Añadimos una columna vacía para el valor real en la tabla

    def actualizar_valor(**kwargs):
        edited_rows = kwargs.get("edited_rows")
        if edited_rows:
            for index, row_data in edited_rows.items():
                fecha_editada = pendientes.iloc[index]["Fecha"]
                valor_real_ingresado = row_data.get("Valor Real")
                if valor_real_ingresado is not None:
                    try:
                        index_original = nuevas_predicciones[nuevas_predicciones["Fecha"] == fecha_editada].index[0]
                        nuevas_predicciones.loc[index_original, "Valor_Real"] = int(round(valor_real_ingresado))

                        print(f"Guardando nuevas predicciones (después de ingresar valor real) en: {NEW_DATA_PATH}")
                        print("Contenido de nuevas_predicciones justo antes de guardar:")
                        print(nuevas_predicciones)

                        nuevas_predicciones.to_excel(NEW_DATA_PATH, index=False) # Añadido index=False
                        st.success(f"Valor real guardado para la fecha: {fecha_editada.strftime('%d/%m/%Y')}")
                        st.rerun() # Volvemos a ejecutar la app para actualizar la tabla
                    except IndexError:
                        st.error(f"Error: No se encontró la fecha {fecha_editada} en el DataFrame de nuevas predicciones.")
                    except Exception as e:
                        st.error(f"Error al actualizar el valor real: {e}")

    edited_df = st.data_editor(
        pendientes[["Fecha", "Predicción", "Limite_Inferior", "Limite_Superior", "Valor Real"]],
        column_config={
            "Fecha": st.column_config.DateColumn("Fecha de Predicción"),
            "Predicción": st.column_config.NumberColumn("Predicción", format="%d"), # Formato para entero
            "Limite_Inferior": st.column_config.NumberColumn("Límite Inferior", format="%d"), # Formato para entero
            "Limite_Superior": st.column_config.NumberColumn("Límite Superior", format="%d"), # Formato para entero
            "Valor Real": st.column_config.NumberColumn(
                "Valor Observado",
                help="Ingrese el valor real observado para esta fecha.",
                required=True,
            ),
        },
        hide_index=True,
        num_rows=len(pendientes),
        on_change=actualizar_valor,
        key="data_editor_ingresar_valor"
    )

    return nuevas_predicciones

def actualizar_dataset(nuevas_predicciones, historial):
    st.subheader("Actualizar Dataset Principal")
    completados = nuevas_predicciones["Valor_Real"].notna().sum()
    st.write(f"Registros completados: {completados}")
    if completados < 50:
        st.warning("Se requieren al menos 50 registros completos para actualizar el dataset.")
        return historial, nuevas_predicciones
    if st.button("Actualizar Dataset"):
        try:
            historial = pd.concat([historial, nuevas_predicciones], ignore_index=True)
            historial.to_excel(HIST_PATH, index=False) # Añadido index=False
            st.success("Dataset principal actualizado. Reentrena el modelo con los nuevos datos.")
            columnas = nuevas_predicciones.columns
            nuevas_predicciones = pd.DataFrame(columns=columnas)

            print(f"Guardando nuevas predicciones (después de actualizar dataset) en: {NEW_DATA_PATH}")
            print("Contenido de nuevas_predicciones justo antes de guardar:")
            print(nuevas_predicciones)

            nuevas_predicciones.to_excel(NEW_DATA_PATH, index=False) # Añadido index=False
            return historial, nuevas_predicciones
        except Exception as e:
            st.error(f"Error al actualizar el dataset principal o al limpiar Nuevas_Predicciones.xlsx: {e}")
            return historial, nuevas_predicciones
    return historial, nuevas_predicciones

def ver_calendario_seleccion():
    st.subheader("Calendario de Partidos")

    st.markdown("**Real Sociedad:**")
    real_sociedad_url = "https://www.sport.es/resultados/futbol/primera-division/calendario-liga/real-sociedad.html"
    st.markdown(f"[Ver calendario de la Real Sociedad]({real_sociedad_url})")
    st.info("Haz clic en el enlace para ver el calendario de partidos de la Real Sociedad en Sport.es.")

    st.markdown("**Selección Española:**")
    seleccion_url = "https://www.sefutbol.com/calendario-resultados"
    st.markdown(f"[Ver calendario de la Selección Española]({seleccion_url})")
    st.info("Haz clic en el enlace para ver el calendario de partidos de la Selección Española.")

    st.sidebar.info("La información de los partidos se muestra a través de enlaces externos.")

# ---------- INTERFAZ PRINCIPAL ----------
def main():
    st.title("Sistema de Predicción de Urgencias")
    st.sidebar.title("Menú")
    opcion = st.sidebar.radio("Seleccione acción:",
                                     ["Nueva Predicción", "Ingresar/Ver Valor Real", "Actualizar Dataset", "Ver Calendario y Selección"])

    modelo = cargar_modelo()
    historial = cargar_historico()
    nuevas_predicciones = cargar_nuevas_predicciones(modelo)
    meteo_days = fetch_forecast()

    if opcion == "Nueva Predicción":
        nuevas_predicciones = nueva_prediccion(modelo, nuevas_predicciones, meteo_days)
    elif opcion == "Ingresar/Ver Valor Real":
        nuevas_predicciones = ingresar_valor_real(nuevas_predicciones)
    elif opcion == "Actualizar Dataset":
        historial, nuevas_predicciones = actualizar_dataset(nuevas_predicciones, historial)
    elif opcion == "Ver Calendario y Selección":
        ver_calendario_seleccion()

    st.sidebar.info("Esta aplicación se ejecuta en la nube y es accesible desde cualquier dispositivo.")

if __name__ == "__main__":
    main()
