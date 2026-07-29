from pathlib import Path
import urllib.request

import cv2
import numpy as np
import streamlit as st
from PIL import Image


def get_face_cascade_path():
    candidates = []

    data_module = getattr(cv2, "data", None)
    if data_module is not None:
        cascade_dir = getattr(data_module, "haarcascades", None)
        if cascade_dir:
            candidates.append(Path(cascade_dir) / "haarcascade_frontalface_default.xml")

    cv2_dir = Path(cv2.__file__).resolve().parent
    candidates.extend(
        [
            cv2_dir / "data" / "haarcascades" / "haarcascade_frontalface_default.xml",
            cv2_dir / "data" / "haarcascade_frontalface_default.xml",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def ensure_face_cascade_file():
    cascade_path = get_face_cascade_path()
    if cascade_path:
        return cascade_path

    local_path = Path(__file__).resolve().parent / "haarcascade_frontalface_default.xml"
    if local_path.exists():
        return str(local_path)

    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            local_path.write_bytes(response.read())
        return str(local_path)
    except Exception:
        return None


def load_face_cascade():
    cascade_path = ensure_face_cascade_file()
    if not cascade_path:
        return None

    try:
        cascade = cv2.CascadeClassifier(cascade_path)
    except Exception:
        return None

    if cascade is None:
        return None

    try:
        is_empty = cascade.empty()
    except Exception:
        is_empty = True

    return cascade if not is_empty else None


def main():
    st.set_page_config(page_title="Trabajo Final Curso Machine Learning", layout="centered")
    st.title("Trabajo Final- Monitor Biometrico Web")

    face_cascade = load_face_cascade()

    if face_cascade is None:
        st.error(
            "No se pudo cargar el clasificador de rostros. Este entorno no dispone de los modelos Haar necesarios para la detección."
        )
        st.stop()

    col1, col2, col3 = st.columns([1, 2, 1])
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
            minSize=(100, 60),
        )
        for (x, y, w, h) in rostros:
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_bgr, "Humano Detectado", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        img_para_mostrar = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img_para_mostrar, caption="Resultado del analisis Biometrico")

        st.subheader("Datos Métricos")
        st.metric(label="Personas detectadas", value=len(rostros))
        if len(rostros) > 0:
            st.success("Persona detectada")
        else:
            st.warning("No se detecto ninguna persona")

        st.button(label="Capturar imagen")


if __name__ == "__main__":
    main()