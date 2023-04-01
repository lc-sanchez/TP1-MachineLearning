import base64
import pickle
import time
from io import BytesIO
import pandas as pd
import streamlit as st


def cargar_modelo(ruta):
    # Cargar el modelo entrenado desde el archivo pkl
    with open(ruta, 'rb') as pkl:
        archivo = pickle.load(pkl)
    return archivo


def predecir(modelo, datos):
    # Realizar la predicción con el modelo cargado
    prediccion = modelo.predict(datos)
    return prediccion


def obtener_clasificacion(prediccion):
    # Aclaración: la prediccion es un valor de 1 o 0
    # No se requiere el uso de "for" pues tendremos sólo 1 dato
    if prediccion == 1:
        clasificacion = ' Fraude'
    else:
        clasificacion = ' No fraude'
    return clasificacion


def agregar_predicciones(dataframe, prediccion):
    df_pred = pd.DataFrame({'prediccion': prediccion})
    # concatenar el dataframe de predicciones con el dataframe original
    df_resultado = pd.concat([dataframe, df_pred], axis=1)
    df_resultado['clasificacion'] = df_resultado['prediccion'].apply(obtener_clasificacion)
    df_resultado = df_resultado.drop('prediccion', axis=1)
    return df_resultado


def convertir_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )


def procesar_categorias(category):
    switcher = {
        'Entretenimiento': 'entertainment',
        'Comida y cena': 'food_dining',
        'Transporte de gasolina': 'gas_transport',
        'Compras en línea': 'grocery_net',
        'Almacen': 'grocery_pos',
        'Salud y fitness': 'health_fitness',
        'Hogar': 'home',
        'Niños y mascotas': 'kids_pets',
        'Otros online': 'misc_net',
        'Otros tienda': 'misc_pos',
        'Cuidado personal': 'personal_care',
        'Shopping online': 'shopping_net',
        'Shopping tienda': 'shopping_pos',
        'Viajes': 'travel',
        'Estado': 'state'
    }
    return switcher.get(category, 'Opción inválida')


def procesar_generos(gender_f):
    if gender_f == "Femenino":
        genero = int(1)
        return genero
    else:
        genero = int(0)
        return genero


def procesar_dias(dia):
    switcher = {
        'Lunes': 'Monday',
        'Martes': 'Tuesday',
        'Miércoles': 'Wednesday',
        'Jueves': 'Thursday',
        'Viernes': 'Friday',
        'Sábado': 'Saturday',
        'Domingo': 'Sunday'
    }
    return switcher.get(dia, 'Día inválido')


def agregar_datos_manualmente():
    categorias = ['Entretenimiento',
                  'Comida y cena',
                  'Transporte de gasolina',
                  'Compras en línea',
                  'Almacen',
                  'Salud y fitness',
                  'Hogar',
                  'Niños y mascotas',
                  'Otros tienda',
                  'Otros online',
                  'Cuidado personal',
                  'Shopping online',
                  'Shopping tienda',
                  'Viajes',
                  'Estado']
    dias = ['Lunes',
            'Martes',
            'Miércoles',
            'Jueves',
            'Viernes',
            'Sábado',
            'Domingo']
    generos = ['Femenino',
               'Masculino']

    with st.expander("AGREGAR DATOS MANUALMENTE"):
        col1, col2 = st.columns([1, 1])

        with col1:
            amt = float(st.number_input("INGRESA EL MONTO (MONTO)", value=0.0, min_value=0.0))
            category = st.selectbox("SELECCIONA LA CATEGORÍA (CATEGORÍA)", categorias)
            edad_usuario = float(st.number_input("INGRESA LA EDAD DEL USUARIO (EDAD)", min_value=18, max_value=100))
            gender_f = st.radio("SELECCIONA EL GÉNERO DEL USUARIO (GENERO)", generos)

        with col2:
            trans_hora = int(st.slider("INGRESA LA HORA DE LA TRANSACCIÓN (HORA)", 00, 23))
            trans_dia = (st.selectbox("INGRESA EL DÍA DE LA TRANSACCIÓN (DÍA)", dias))
            trans_mes = int(st.selectbox("SELECCIONA EL MES DE LA TRANSACCIÓN (MES)", range(1, 13)))

            categoria = procesar_categorias(category)
            genero = int(procesar_generos(gender_f))
            dia = procesar_dias(trans_dia)

            # Crear un diccionario con los valores obtenidos del formulario
            data = {"category": categoria,
                    "amt": amt,
                    "state": '0',
                    "city_pop": 0,
                    "trans_hora": trans_hora,
                    "trans_mes": trans_mes,
                    "trans_dia": dia,
                    "delay_entre_trans": 0,
                    "edad_usuario": edad_usuario,
                    "dif_lat_comprador_merch": 0,
                    "dif_long_comprador_merch": 0,
                    "dif_lat_prev_merch": 0,
                    "dif_long_prev_merch": 0,
                    "gender_f": genero}

            # Convertir el diccionario en un DataFrame
            dato = pd.DataFrame([data], index=[0])
    return dato


def predecir_dato(dato_procesado, modelo):
    info_box_wait_clasificacion = st.info('Realizando la clasificación...')  # Cargamos el dato en una variable
    prediccion = predecir(modelo, dato_procesado)  # Obtenemos la clasificación
    categoria = obtener_clasificacion(prediccion)
    info_box_wait_clasificacion.empty()
    return categoria


def predecir_pkl(archivo, modelo):
    info_box_wait_clasificacion = st.info('Realizando la clasificación...')  # Cargamos el dato en una variable
    dato = pickle.loads(archivo.getvalue())  # Acá viene la predicción con el modelo
    prediccion = predecir(modelo, dato)  # Obtenemos la clasificación
    categoria = obtener_clasificacion(prediccion)
    info_box_wait_clasificacion.empty()
    return categoria


def predecir_csv(archivo, modelo):
    info_box_wait_clasificacion = st.info('Realizando la clasificación...')
    dataset_subido = pd.read_csv(archivo)
    prediccion = predecir(modelo, dataset_subido)
    df_prediccion = agregar_predicciones(dataset_subido, prediccion)
    info_box_wait_clasificacion.empty()

    return df_prediccion


def crear_descargable(dataframe, nombre):
    colA, colB, colC = st.columns([1, 1, 1])
    with colB:
        spinner = st.spinner('Creando descargable... esto puede llevar un tiempo.')
        with spinner:
            archivo_excel = convertir_excel(dataframe)
            time.sleep(5)
        st.download_button(label='DESCARGAR RESULTADOS', data=archivo_excel,
                           file_name=nombre + '_resultados.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           use_container_width=True)


def procesar_dato(dato, procesador, df_procesador):
    dato_procesado = procesador.procesarConsulta(df_procesador, dato)
    return dato_procesado
