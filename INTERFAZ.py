import os
import glob
import pandas as pd
import numpy as np
import pickle
import requests
import streamlit as st
from datetime import datetime, timedelta
import warnings
from PIL import Image

# No usaremos components.iframe para la página si se bloquea; en su lugar mostraremos un enlace.
# Suprimir FutureWarnings para mantener la salida limpia
warnings.simplefilter("ignore", category=FutureWarning)

# ---------- CONFIGURACIÓN INICIAL DE LA PÁGINA ----------
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# ---------- CONFIGURACIÓN DE RUTAS ----------
# BASE_DIR = r"D:\PREDICCION URGENCIAS\MODELO PREDICCION DE URGENCIAS" # Cambiar a la ruta correcta
BASE_DIR = "https://raw.githubusercontent.com/jsampedro2025/PREDICCION_URGENCIAS/main"  # Ruta base del repositorio en GitHub
MODEL_PATH = f"{BASE_DIR}/modelo_prediccion_urgencias.pkl"  # Ruta del modelo
HIST_PATH = f"{BASE_DIR}/DATASET_MEJORADO.xlsx"  # Ruta del histórico
NEW_DATA_PATH = os.path.join(".", "Nuevas_Predicciones.xlsx") # Ruta local para el nuevo archivo de predicciones
PRED_OBS_PATH = os.path.join(".", "Predicciones_Con_Observado.xlsx") # Nuevo path para el archivo con valores observados


# Comprobación de existencia de archivos - No se puede hacer directamente con URLs
# if not os.path.exists(MODEL_PATH):
#     st.error(f"Error: No se encontró el archivo del modelo en la ruta: {MODEL_PATH}.  Asegúrate de que el archivo exista y la ruta sea correcta.")
# if not os.path.exists(HIST_PATH):
#     st.error(f"Error: No se encontró el archivo histórico en la ruta: {HIST_PATH}. Asegúrate de que el archivo exista y la ruta sea correcta.")


# ---------- FUNCIONES CON CACHEO ----------
@st.cache_resource
def cargar_modelo():
    try:
        # Descargar el modelo desde la URL
        response = requests.get(MODEL_PATH)
        response.raise_for_status()  # Lanza una excepción para errores 4xx o 5xx
        modelo_data = response.content
        modelo = pickle.loads(modelo_data)
        print(f"Modelo cargado exitosamente desde: {MODEL_PATH}")
        return modelo
    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar el modelo desde la URL: {e}")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

@st.cache_data
def cargar_historico():
    try:
        # Descargar el archivo histórico desde la URL
        response = requests.get(HIST_PATH)
        response.raise_for_status()
        historico_data = response.content
        df = pd.read_excel(historico_data) # Leer desde los bytes
        df.columns = df.columns.astype(str)
        if "Predicción" not in df.columns:
            df["Predicción"] = np.nan
        print(f"Datos históricos cargados exitosamente desde: {HIST_PATH}")
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error al descargar el archivo histórico desde la URL: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar los datos históricos: {e}")
        return pd.DataFrame()

def cargar_nuevas_predicciones():
    try:
        if os.path.exists(NEW_DATA_PATH):
            df = pd.read_excel(NEW_DATA_PATH, parse_dates=[0]) # No se usa index_col al cargar
            print(f"Nuevas predicciones cargadas desde: {NEW_DATA_PATH}")
        else:
            df = pd.DataFrame()
            print(f"Archivo de nuevas predicciones no encontrado. Se creó un nuevo DataFrame vacío.")
        return df
    except Exception as e:
        st.error(f"Error al cargar o crear el archivo de nuevas predicciones: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_forecast():
    # Reemplaza "TU_API_KEY_AQUI" con tu API key real de AEMET.
    API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJkci5zYW1wZWRyb0BnbWFpbC5jb20iLCJqdGkiOiI3M2U3MmQxOC1jNTMyLTRkM2UtYmYwOC1hMGJhNWM4YWRmOGMiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTczODc4NDc1NCwidXNlcklkIjoiNzNlNzJkMTgtYzUzMi00ZDNlLWJmMDgtYTBiYTVjOGFkZjhjIiwicm9sZSI6IiJ9.aIbLKEsLbb5DmtaYyxStWLQipyqn_v6YynXbUIjQI_c"
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
        return None  # Retornar None para indicar fallo

    # Extraer fechas disponibles de AEMET
    forecast_dates = [datetime.fromisoformat(d["fecha"]).date() for d in meteo_days if "fecha" in d]
    if not forecast_dates:
        st.error("No se encontraron fechas válidas en los datos meteorológicos.")
        return None  # Retornar None para indicar fallo

    min_date = min(forecast_dates)
    max_date = max(forecast_dates)
    st.info(f"El rango de fechas disponible es: {min_date.strftime('%d/%m/%Y')} a {max_date.strftime('%d/%m/%Y')}")

    # Selección de fecha
    fecha = st.date_input("Selecciona la fecha de la predicción",
                            value=min_date, min_value=min_date, max_value=max_date)
    if fecha not in forecast_dates:
        st.error("La fecha seleccionada no tiene datos meteorológicos disponibles.")
        return None  # Retornar None para indicar fallo

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
        return None  # Retornar None para indicar fallo

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

            # Guardar la predicción en un nuevo DataFrame y guardarlo en Excel
            nuevas_predicciones_df = pd.DataFrame([data])
            if not os.path.exists(NEW_DATA_PATH):
                nuevas_predicciones_df.to_excel(NEW_DATA_PATH, index=False)
            else:
                # Read the existing excel file
                existing_df = pd.read_excel(NEW_DATA_PATH)
                # Append the new data
                updated_df = pd.concat([existing_df, nuevas_predicciones_df], ignore_index=True)
                # Write the combined data to the excel file
                updated_df.to_excel(NEW_DATA_PATH, index=False)

            st.info(f"Predicción guardada en: {NEW_DATA_PATH}") # Informar al usuario
            return data # Retornar el diccionario con la predicción
        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")
            return None
    elif modelo is None:
        st.warning("El modelo no se ha cargado correctamente. La predicción no está disponible.")
        return None
    return None

def ingresar_valor_real(nuevas_predicciones, nuevas_predicciones_dict):
    st.subheader("Ingresar Valor Real")

    # Leer el archivo de Nuevas_Predicciones.xlsx
    nuevas_predicciones_df = cargar_nuevas_predicciones()

    if nuevas_predicciones_df.empty:
        st.info("No hay predicciones pendientes de asignar valor real.")
        return pd.DataFrame()

    # Filtrar las predicciones que no tienen valor real
    pendientes_df = nuevas_predicciones_df[nuevas_predicciones_df["Valor_Real"].isna()].copy()
    if pendientes_df.empty:
        st.info("Todas las predicciones tienen valor real asignado.")
        return pd.DataFrame()

    pendientes_df["Fecha"] = pd.to_datetime(pendientes_df["Fecha"])

    # Mostrar solo las columnas necesarias
    columnas_a_mostrar = ["Fecha", "Predicción", "Limite_Inferior", "Limite_Superior", "Valor_Real"]
    pendientes_df = pendientes_df[columnas_a_mostrar]

    # Crear un data editor para ingresar el valor real
    edited_df = st.data_editor(
        pendientes_df,
        column_config={
            "Fecha": st.column_config.DateColumn("Fecha de Predicción"),
            "Predicción": st.column_config.NumberColumn("Predicción", format="%d"),
            "Limite_Inferior": st.column_config.NumberColumn("Límite Inferior", format="%d"),
            "Limite_Superior": st.column_config.NumberColumn("Límite Superior", format="%d"),
            "Valor_Real": st.column_config.NumberColumn(
                "Valor Observado",
                help="Ingrese el valor real observado para esta fecha.",
                required=True,
            ),
        },
        hide_index=True,
        num_rows=len(pendientes_df),
        key="data_editor_ingresar_valor"
    )

    # Botón para guardar los datos
    if st.button("Guardar Valores Reales"):
        # Verificar si se ha ingresado algún valor real
        if edited_df["Valor_Real"].notna().any():
            try:
                # Actualizar los valores reales en el DataFrame original
                for index, row in edited_df.iterrows():
                    fecha_prediccion = row["Fecha"]
                    valor_real = row["Valor_Real"]
                    # Actualizar el valor real en el DataFrame principal usando la fecha como clave
                    nuevas_predicciones_df.loc[nuevas_predicciones_df["Fecha"] == fecha_prediccion, "Valor_Real"] = valor_real

                # Guardar el DataFrame actualizado en el archivo Excel NUEVAS_PREDICCIONES_PATH
                nuevas_predicciones_df.to_excel(NEW_DATA_PATH, index=False)
                st.success(f"Valores reales guardados exitosamente en: {NEW_DATA_PATH}")
            except Exception as e:
                st.error(f"Error al guardar los valores reales: {e}")
        else:
            st.warning("No se ha ingresado ningún valor real.")

    return nuevas_predicciones_df # Retornar el DataFrame actualizado

def actualizar_dataset(nuevas_predicciones, historial):
    st.subheader("Actualizar Dataset Principal")
    if nuevas_predicciones.empty:
        st.warning("No hay registros para actualizar el dataset.")
        return historial, nuevas_predicciones
    completados = nuevas_predicciones["Valor_Real"].notna().sum()
    st.write(f"Registros completados: {completados}")
    if completados < 50:
        st.warning("Se requieren al menos 50 registros completos para actualizar el dataset.")
        return historial, nuevas_predicciones
    if st.button("Actualizar Dataset"):
        try:
            # Leer el archivo de Predicciones_Con_Observado.xlsx
            if os.path.exists(PRED_OBS_PATH):
                pred_con_obs_df = pd.read_excel(PRED_OBS_PATH)
                # Concatenar con el historial
                historial = pd.concat([historial, pred_con_obs_df], ignore_index=True)
                historial.to_excel(HIST_PATH, index=False)
                os.remove(PRED_OBS_PATH) # Eliminar el archivo despues de actualizar el principal
            else:
                st.warning("No se encontró el archivo 'Predicciones_Con_Observado.xlsx'. No se actualizó el dataset.")

            nuevas_predicciones = pd.DataFrame() # Reinicia el DataFrame
            if os.path.exists(NEW_DATA_PATH):
                os.remove(NEW_DATA_PATH)
            st.success("Dataset principal actualizado y nuevas predicciones reiniciadas.")
            return historial, nuevas_predicciones
        except Exception as e:
            st.error(f"Error al actualizar el dataset: {e}")
            return historial, nuevas_predicciones
    return historial, nuevas_predicciones

def ver_calendario_seleccion(nuevas_predicciones):
    st.subheader("Calendario de Predicciones y Valores Reales")
    if nuevas_predicciones.empty:
        st.warning("No hay predicciones registradas.")
        return

    nuevas_predicciones["Fecha"] = pd.to_datetime(nuevas_predicciones["Fecha"])  # Asegurarse de que 'Fecha' sea datetime
    nuevas_predicciones = nuevas_predicciones.sort_values("Fecha")

    # Crear el calendario
    cal_df = nuevas_predicciones[["Fecha", "Predicción", "Valor_Real"]].copy()

    # Agrupar por semana y calcular los promedios
    cal_df = cal_df.set_index('Fecha').resample('W-MON').agg(
        Predicción=('Predicción', 'mean'),
        Valor_Real=('Valor_Real', 'mean')
    ).reset_index()

    cal_df['Fecha_Fin'] = cal_df['Fecha'] + timedelta(days=6)
    cal_df['Semana_Inicio_Fin'] = cal_df.apply(lambda row: f"{row['Fecha'].strftime('%d/%m/%Y')} - {row['Fecha_Fin'].strftime('%d/%m/%Y')}", axis=1)


    # Mostrar el calendario agrupado por semana
    st.dataframe(cal_df[['Semana_Inicio_Fin', 'Predicción', 'Valor_Real']], hide_index=True)

    # Opcional: Mostrar los datos diarios también
    if st.checkbox("Mostrar datos diarios"):
        st.dataframe(nuevas_predicciones[["Fecha", "Predicción", "Valor_Real"]], hide_index=True)

def main():
    st.title("Sistema de Predicción de Urgencias")
    st.sidebar.title("Menú")
    opcion = st.sidebar.radio("Seleccione acción:",
                                 ["Nueva Predicción", "Ingresar/Ver Valor Real", "Actualizar Dataset", "Ver Calendario y Selección"])

    modelo = cargar_modelo()
    historial = cargar_historico()
    nuevas_predicciones = cargar_nuevas_predicciones()
    meteo_days = fetch_forecast()
    nuevas_predicciones_dict = {} # Inicializamos el diccionario aquí

    if opcion == "Nueva Predicción":
        nuevas_predicciones_dict = nueva_prediccion(modelo, nuevas_predicciones, meteo_days)
    elif opcion == "Ingresar/Ver Valor Real":
        nuevas_predicciones = ingresar_valor_real(nuevas_predicciones, nuevas_predicciones_dict)
    elif opcion == "Actualizar Dataset":
        historial, nuevas_predicciones = actualizar_dataset(nuevas_predicciones, historial)
    elif opcion == "Ver Calendario y Selección":
        ver_calendario_seleccion(nuevas_predicciones)

    st.sidebar.info("Esta aplicación se ejecuta en la nube y es accesible desde cualquier dispositivo.")

if __name__ == "__main__":
    main()

