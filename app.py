import streamlit as st
import pandas as pd
import joblib
import numpy as np  
from typing import Callable, Optional

# --- Konstanta ---
MODEL_PATH = 'model.joblib'

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Risiko Bedah Toraks", 
    page_icon="🩺", 
    layout="wide"
)

# --- Fungsi Pemuatan Model (di-cache) ---
@st.cache_resource
def load_model(path: str) -> Optional[Callable]:
    """Memuat model dari path yang diberikan, mengembalikan None jika tidak ditemukan."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"File model tidak ditemukan di '{path}'. Pastikan file ada atau jalankan skrip training.")
        return None

# --- Komponen UI ---
def build_sidebar() -> pd.DataFrame:
    """Membangun sidebar dan mengumpulkan input dari pengguna."""
    st.sidebar.header("Data Pra-Operasi Pasien")

    # DGN
    dgn_options = [f'DGN{i}' for i in [1, 2, 3, 4, 5, 6, 8]]
    dgn = st.sidebar.selectbox('Kode Diagnosis (DGN)', dgn_options, index=2)

    # Performance Status
    pre6_options = ['PRZ0', 'PRZ1', 'PRZ2']
    pre6 = st.sidebar.selectbox('Status Performa (PRE6)', pre6_options, index=1)

    # Ukuran Tumor
    pre14_options = ['T1', 'T2', 'T3', 'T4']
    pre14 = st.sidebar.selectbox('Ukuran Tumor (PRE14)', pre14_options, index=1)

    st.sidebar.subheader("Pengukuran Kapasitas Paru")
    pre4 = st.sidebar.slider('Kapasitas Vital Paksa (PRE4 - FVC)', 1.4, 6.3, 3.2, 0.1)
    pre5 = st.sidebar.slider('Volume Ekspirasi Paksa dlm 1d (PRE5 - FEV1)', 0.9, 5.0, 2.5, 0.1)
    
    age = st.sidebar.slider('Usia', 21, 87, 62)

    st.sidebar.subheader("Kondisi Lain (Benar/Salah)")
    pre7 = st.sidebar.checkbox('Nyeri sebelum operasi (PRE7)', value=False)
    pre8 = st.sidebar.checkbox('Hemoptisis sebelum operasi (PRE8)', value=False)
    pre9 = st.sidebar.checkbox('Dispnea sebelum operasi (PRE9)', value=False)
    pre10 = st.sidebar.checkbox('Batuk sebelum operasi (PRE10)', value=True)
    pre11 = st.sidebar.checkbox('Kelemahan sebelum operasi (PRE11)', value=False)
    pre17 = st.sidebar.checkbox('Diabetes Melitus (PRE17)', value=False)
    pre19 = st.sidebar.checkbox('Infark Miokard dalam 6 bulan (PRE19)', value=False)
    pre25 = st.sidebar.checkbox('Penyakit Arteri Perifer (PRE25)', value=False)
    pre30 = st.sidebar.checkbox('Merokok (PRE30)', value=True)
    pre32 = st.sidebar.checkbox('Asma (PRE32)', value=False)

    data = {
        'DGN': dgn, 'PRE4': pre4, 'PRE5': pre5, 'PRE6': pre6,
        'PRE7': pre7, 'PRE8': pre8, 'PRE9': pre9, 'PRE10': pre10,
        'PRE11': pre11, 'PRE14': pre14, 'PRE17': pre17, 'PRE19': pre19,
        'PRE25': pre25, 'PRE30': pre30, 'PRE32': pre32, 'AGE': age
    }
    return pd.DataFrame(data, index=[0])

def display_prediction_results(model: Callable, input_df: pd.DataFrame):
    """Menjalankan prediksi dan menampilkan hasilnya."""
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Hasil Prediksi")
        if prediction:
            st.error("Prediksi: RISIKO TINGGI (Pasien diprediksi tidak akan bertahan 1 tahun)")
        else:
            st.success("Prediksi: RISIKO RENDAH (Pasien diprediksi akan bertahan 1 tahun)")

        st.subheader("Tingkat Kepercayaan Prediksi")
        proba_df = pd.DataFrame(
            prediction_proba,
            columns=['Kepercayaan untuk RISIKO RENDAH', 'Kepercayaan untuk RISIKO TINGGI'],
            index=["Probabilitas"]
        )
        st.dataframe(proba_df.style.format('{:.2%}'))
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

# --- Aplikasi Utama ---
def main():
    """Fungsi utama untuk menjalankan aplikasi Streamlit."""
    st.title("🩺 Prediksi Risiko Pasca Operasi Bedah Toraks")
    st.write("Aplikasi ini memprediksi risiko kelangsungan hidup 1 tahun pasien setelah operasi kanker paru-paru. Silakan masukkan data pra-operasi pasien menggunakan panel di sebelah kiri.")

    model = load_model(MODEL_PATH)
    
    if model:
        input_df = build_sidebar()
        
        st.subheader("Ringkasan Data Pasien:")
        st.dataframe(input_df)
        
        if st.button("Prediksi Risiko"):
            display_prediction_results(model, input_df)
            
    st.markdown("---")
    st.write("Aplikasi ini menggunakan model Gaussian Naive Bayes yang dilatih pada dataset Bedah Toraks dari UCI Repository.")

if __name__ == "__main__":
    main() 