import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

CLASSES = ['Lung Normal (lung_n)', 'Lung Adenocarcinoma (lung_aca)', 'Lung Squamous Cell Carcinoma (lung_scc)']
IMG_SIZE = 128

@st.cache_resource
def load_cancer_model():
    return tf.keras.models.load_model('MobileNetV2.h5')

st.title("Lung Cancer Image Classifier 🫁")
st.write("อัปโหลดรูปภาพเนื้อเยื่อปอดเพื่อทำการจำแนกประเภท")

model = load_cancer_model()

uploaded_file = st.file_uploader("เลือกไฟล์ภาพ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='ภาพที่อัปโหลด', use_container_width=True)
    st.write("กำลังประมวลผล...")

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100

    st.success(f"ผลการทำนาย: {CLASSES[predicted_class_idx]}")
    st.info(f"ความมั่นใจ: {confidence:.2f}%")
