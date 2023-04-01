import os
import time

from procesador import Procesador
from utils import *
import streamlit as st

st.set_page_config(page_icon="logo_ungs.png", page_title="Detección de Fraudes de tarjetas de crédito", layout="wide")
add_bg_from_local('f1.jpg')
col1, col2, col3 = st.columns([1.5, 8, 1.5])

with col2:
    colA, colB, colC = st.columns([1, 0.5, 1])
    colB.image("logo_ungs.png", use_column_width=True)
    st.title("DETECCIÓN DE FRAUDES CON ML")
    st.header("CONSULTAR CONJUNTO DE DATOS:")
    box_warning = st.expander("Columnas admitidas")
    with box_warning:
        st.write("Columnas admitidas: [category], [amt], [state], [city_pop], [trans_hora], [trans_mes], "
                 "[trans_dia], [delay_entre_trans], [edad_usuario], [dif_lat_comprador_merch], "
                 "[dif_long_comprador_merch], [dif_lat_prev_merch], [dif_long_prev_merch], [gender_f]")
    archivo_subido = st.file_uploader("archivo", label_visibility="hidden", type=['pkl', 'csv'], key="1")
    box_info = st.info('Puede agregar un dataset completo o una fila')

    modelo = cargar_modelo('modelo_entrenado.pkl')
    if archivo_subido is not None:
        box_info.empty()
        box_warning.empty()

        nombre, extension = os.path.splitext(archivo_subido.name)

        if extension == '.pkl':
            categoria = predecir_pkl(archivo_subido, modelo)
            info_resultado = st.info(f"La transacción analizada es: {categoria}")
            st.stop()

        elif extension == '.csv':
            df_prediccion = predecir_csv(archivo_subido, modelo)
            info_box_wait = st.info('Cargando...')
            st.dataframe(df_prediccion)
            info_box_wait.empty()
            colA, colB, colC = st.columns([1, 1, 1])
            with colB:
                boton_crear_descargable = st.button('CREAR ARCHIVO', use_container_width=True)
            if boton_crear_descargable:
                crear_descargable(df_prediccion, nombre)

    else:
        procesador = Procesador('data.pkl')
        st.header("CONSULTAR DATOS MANUALMENTE:")
        dato_sin_procesar = agregar_datos_manualmente()
        colA, colB, colC = st.columns([1, 1, 1])
        with colB:
            boton_predecir = st.button("PREDECIR", use_container_width=True)
        if boton_predecir:
            spinner = st.spinner('Procesando dato... esto puede llevar un tiempo.')
            with spinner:
                dato_procesado = procesar_dato(dato_sin_procesar, procesador, procesador.get_df())
                time.sleep(5)
            # st.write(dato_procesado) #ver dataframe creado

            categoria = predecir_dato(dato_procesado, modelo)
            info_resultado = st.info(f"La transacción analizada es:{categoria}")
