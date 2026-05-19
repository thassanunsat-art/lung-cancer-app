import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. ตั้งค่าหน้าเพจ
st.set_page_config(page_title="AI Image Classifier", page_icon="🤖")
st.title("แอปพลิเคชันจำแนกรูปภาพด้วย MobileNetV2 🤖")

# 2. ฟังก์ชันโหลดโมเดล (ใช้ @st.cache_resource เพื่อไม่ให้โหลดใหม่ทุกครั้งที่รีเฟรชหน้า)
@st.cache_resource
def load_model():
    # โหลดโมเดล Keras ที่คุณอัปโหลดไว้
    model = tf.keras.models.load_model('MobileNetV2.keras')
    return model

model = load_model()
st.success("โหลดโมเดลสำเร็จแล้ว!")

# 3. สร้างส่วนอัปโหลดไฟล์รูปภาพ
uploaded_file = st.file_uploader("อัปโหลดรูปภาพที่นี่ (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงรูปภาพที่อัปโหลด
    image = Image.open(uploaded_file)
    st.image(image, caption="รูปภาพที่อัปโหลด", use_column_width=True)
    
    st.write("กำลังประมวลผล...")
    
    # 4. เตรียมรูปภาพให้พร้อมสำหรับโมเดล (Pre-processing)
    # จากข้อมูลโมเดลของคุณ ใช้ขนาด 128x128
    image_resized = image.resize((128, 128)) 
    image_array = np.array(image_resized)
    
    # ตรวจสอบว่าภาพมี 3 channels (RGB)
    if image_array.shape[-1] == 4: # ถ้าเป็น RGBA ให้แปลงเป็น RGB
        image_array = image_array[..., :3]
        
    image_array = image_array / 255.0 # Normalize ค่าสีให้อยู่ในช่วง 0-1
    image_array = np.expand_dims(image_array, axis=0) # เพิ่มมิติ Batch size ให้เป็น (1, 128, 128, 3)
    
    # 5. ทำนายผล (Prediction)
    if st.button("ทำนายผลภาพนี้"):
        predictions = model.predict(image_array)
        
        # แสดงผลลัพธ์ (สมมติว่าเป็นโมเดลที่มีหลายคลาส)
        predicted_class = np.argmax(predictions, axis=1)
        st.subheader(f"ผลการทำนาย: คลาสที่ {predicted_class[0]}")
        st.write("ความน่าจะเป็น:", predictions)
