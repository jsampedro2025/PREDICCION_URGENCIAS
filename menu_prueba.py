import streamlit as st

st.set_page_config(initial_sidebar_state="expanded")

st.sidebar.title("Menú")
opcion = st.sidebar.radio("Seleccione opción:", ["Opción 1", "Opción 2"])

st.title(f"Has seleccionado: {opcion}")
