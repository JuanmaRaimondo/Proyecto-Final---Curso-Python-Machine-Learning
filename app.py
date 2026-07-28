import cv2
import streamlit as st
import numpy as np
from PIL import Image

## Primera Parte, Capturar imagen y definir la interfaz grafica.

st.set_page_config(page_title="Trabajo Final Curso Machine Learning", layout="centered")
st.title("Trabajo Final- Monitor Biometrico Web") 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

col1, col2, col3 = st.columns([1,2,1])
with col2:
    foto_camara = st.camera_input(label="Capturar imagen para el monitor biometrico")

if foto_camara is not None:
    img_original = Image.open(foto_camara)
    img_array = np.array(img_original)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    rostros = face_cascade.detectMultiScale(
        gris,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(100, 60)
    )
    for (x, y, w, h) in rostros:
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_bgr, "Humano Detectado", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    img_para_mostrar = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img_para_mostrar, caption="Resultado del analisis Biometrico")

    st.subheader("Datos Métricos")
    st.metric(label="¨Personas detectadas", value=len(rostros))
    if len(rostros) > 0:
         st.success("Persona detectada")
    else: st.warning("No se detecto ninguna persona")
         



    st.button(label="Capturar imagen")