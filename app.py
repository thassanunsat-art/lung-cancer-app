import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ==========================================
# 1. ตั้งค่าหน้าต่างโปรแกรม & ดีไซน์ CSS Custom
# ==========================================
st.set_page_config(
    page_title="Lung Cancer Image Classifier",
    page_icon="🫁",
    layout="centered"
)

# แทรก CSS เพื่อเปลี่ยนหน้าตา Streamlit ให้เป็นการ์ด UI เหมือนรูปตัวอย่าง
st.markdown("""
    <style>
        /* สีพื้นหลังด้านนอก */
        [data-testid="stAppViewContainer"] {
            background-color: #0B132B; 
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        /* ซ่อนแถมเครื่องมือของ Streamlit */
        [data-testid="stHeader"], footer {
            visibility: hidden;
        }

        /* เปลี่ยนกล่อง Content ให้กลายเป็นการ์ดมนทรงโค้งตามภาพตัวอย่าง */
        .main .block-container {
            background-color: #F8FAFC; 
            border: 4px solid #3B82F6; 
            border-radius: 40px;
            padding: 40px 30px !important;
            max-width: 460px !important;
            margin: 30px auto !important;
            box-shadow: 0 15px 35px rgba(0,0,0,0.6);
            text-align: center;
        }

        /* ส่วนหัวข้อหลัก */
        .app-header {
            margin-bottom: 25px;
        }
        .app-header h1 {
            color: #1E293B !important;
            font-size: 26px !important;
            font-weight: 800 !important;
            line-height: 1.2 !important;
            margin-top: 10px !important;
        }
        .app-header p {
            color: #64748B !important;
            font-size: 13.5px !important;
            font-weight: 500 !important;
            line-height: 1.4 !important;
            margin-top: 10px !important;
        }

        /* ตกแต่งกล่องอัปโหลดไฟล์ */
        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #3B82F6 !important;
            background-color: #F1F5F9 !important;
            border-radius: 20px !important;
        }
        
        /* ตกแต่งปุ่มทำนายผล (Predict Button) */
        div.stButton > button:first-child {
            background: linear-gradient(to right, #3B82F6, #1D4ED8) !important;
            color: white !important;
            border: none !important;
            border-radius: 30px !important;
            padding: 14px 20px !important;
            font-size: 16px !important;
            font-weight: bold !important;
            width: 100% !important;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
            transition: transform 0.1s, box-shadow 0.2s;
            margin-top: 15px;
        }
        div.stButton > button:first-child:hover {
            transform: scale(1.01);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6) !important;
        }

        /* ส่วนหัวข้อผลลัพธ์ */
        .result-title {
            color: #1E293B;
            font-size: 20px;
            font-weight: 700;
            margin: 20px 0 10px 0;
        }

        /* กล่องแสดงผลลัพธ์สีเข้มด้านล่าง */
        .result-box {
            background-color: #1E293B; 
            border-radius: 20px;
            padding: 25px 20px;
            color: #94A3B8;
            font-weight: 600;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }

        /* ตกแต่งปุ่มล้างข้อมูล (Clear Button) */
        .clear-container div.stButton > button {
            background-color: white !important;
            color: #475569 !important;
            border: 2px solid #CBD5E1 !important;
            border-radius: 30px !important;
            padding: 10px 20px !important;
            width: 100% !important;
            font-weight: bold !important;
        }

        /* ข้อความท้ายแอป */
        .footer-text {
            color: #94A3B8;
            font-size: 11px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ส่วนโหลดโมเดล AI (TensorFlow)
# ==========================================
CLASSES = ['Lung Normal (lung_n)', 'Lung Adenocarcinoma (lung_aca)', 'Lung Squamous Cell Carcinoma (lung_scc)']
IMG_SIZE = 128

@st.cache_resource
def load_cancer_model():
    return tf.keras.models.load_model('best_model.keras')

model = load_cancer_model()

# ==========================================
# 3. โครงสร้างเนื้อหาและการประมวลผล UI
# ==========================================

# ส่วนหัวข้อดีไซน์การแพทย์
st.markdown("""
    <div class="app-header">
        <svg viewBox="0 0 64 64" width="80" height="80" fill="none" stroke="#3B82F6" stroke-width="2.5">
            <circle cx="32" cy="32" r="22" stroke-dasharray="4 2"/>
            <circle cx="32" cy="32" r="12" fill="#3B82F6" fill-opacity="0.15"/>
            <path d="M32 2v6M32 56v6M2 32h6M56 32h6" stroke-linecap="round"/>
            <path d="M18 18l4 4M42 42l4 4M18 42l4-4M42 18l-4 4" stroke-linecap="round"/>
        </svg>
        <h1>Lung Cancer<br>Image Classifier 🫁</h1>
        <p>Innovative Deep Learning Architecture for the Classification of Lung Cancer From Histopathology Images</p>
    </div>
""", unsafe_allow_html=True)

# ตัวอัปโหลดรูปภาพ
uploaded_file = st.file_uploader("เลือกไฟล์ภาพ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# แสดงรูปภาพพรีวิวเมื่อมีการอัปโหลด
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='ภาพเนื้อเยื่อที่อัปโหลด', use_container_width=True)

# ปุ่มสั่ง Predict
predict_btn = st.button("Predict Cancer Type")

st.markdown('<div class="result-title">Prediction Result</div>', unsafe_allow_html=True)

# ตรรกะการคำนวณผลและแสดงผลลัพธ์ในกล่องดีไซน์แบบ Custom
if predict_btn:
    if uploaded_file is not None:
        with st.spinner('กำลังประมวลผลภาพ...'):
            # ทำ Preprocessing รูปภาพตามโมเดลของคุณ
            img = image.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # ให้โมเดลทายผล
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100
            
            result_class = CLASSES[predicted_class_idx]
            
            # ปรับสีกล่องตามผลลัพธ์ (ถ้า Normal เป็นสีเขียว, ถ้าเป็นมะเร็งเป็นสีแดง)
            status_color = "#10B981" if "Normal" in result_class else "#EF4444"

            # แสดงผลลัพธ์แบบสวยงามในการ์ดสีเข้ม
            st.markdown(f"""
                <div class="result-box" style="border: 2px solid {status_color};">
                    <svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="{status_color}" stroke-width="2">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M22 4L12 14.01l-3-3" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    <div style="color: {status_color}; font-size: 16px; font-weight: 800;">{result_class}</div>
                    <div style="color: #94A3B8; font-size: 14px;">ความมั่นใจ: {confidence:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="result-box" style="border: 1px solid #EF4444;">
                <div style="color: #EF4444; font-size: 14px;">กรุณาอัปโหลดรูปภาพก่อนทำการทำนายผล</div>
            </div>
        """, unsafe_allow_html=True)
else:
    # สถานะเริ่มต้นตอนยังไม่ได้กดปุ่ม Predict
    st.markdown("""
        <div class="result-box">
            <svg viewBox="0 0 24 24" width="35" height="35" fill="none" stroke="#475569" stroke-width="2">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <div style="color: #64748B; font-size: 14px;">ยังไม่มีการประมวลผลผลลัพธ์</div>
        </div>
    """, unsafe_allow_html=True)

# ปุ่มเคลียร์ข้อมูลทั้งหมด (Clear)
st.markdown('<div class="clear-container">', unsafe_allow_html=True)
if st.button("Clear Data"):
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ข้อความเครดิตส่วนท้าย
st.markdown("""
    <div class="footer-text">
        Deep learning model for lung cancer image classification
    </div>
""", unsafe_allow_html=True)
