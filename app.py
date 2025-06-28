import streamlit as st
import numpy as np
import json
from typing import Dict, Any, Optional, Tuple

# --- Konstanta ---
PARAMS_PATH = 'manual_model_params.json'

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Risiko Bedah Toraks", 
    page_icon="🩺", 
    layout="wide"
)

# --- Fungsi Pemuatan & Prediksi Manual ---
@st.cache_resource
def load_manual_params(path: str) -> Optional[Dict[str, Any]]:
    """Memuat parameter model manual dari file JSON."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File parameter '{path}' tidak ditemukan. Jalankan skrip training terlebih dahulu.")
        return None

def predict_manual(params: Dict[str, Any], input_dict: Dict[str, Any]) -> Tuple[bool, np.ndarray]:
    """Melakukan prediksi menggunakan perhitungan manual dari parameter yang diekstrak."""
    num_params = params['numerical']
    cat_params = params['categorical']
    bool_features = params['boolean_features']
    clf_params = params['classifier']

    # 1. Scaling Fitur Numerik
    input_num_list = [input_dict[f] for f in num_params['features']]
    scaled_num = (np.array(input_num_list) - np.array(num_params['mean'])) / np.array(num_params['scale'])

    # 2. One-Hot Encoding Fitur Kategorikal
    encoded_cat_parts = []
    for i, feature in enumerate(cat_params['features']):
        categories = np.array(cat_params['categories'][i])
        user_value = input_dict[feature]
        encoded = (categories == user_value).astype(int)
        encoded_cat_parts.append(encoded)
    encoded_cat = np.concatenate(encoded_cat_parts)

    # 3. Fitur Boolean
    input_bool = np.array([float(input_dict[f]) for f in bool_features])

    # 4. Gabungkan semua fitur menjadi satu vektor
    final_features = np.concatenate([scaled_num, encoded_cat, input_bool])

    # 5. Perhitungan Manual Gaussian Naive Bayes
    classes = np.array(clf_params['classes'])
    class_priors = np.array(clf_params['class_prior'])
    theta = np.array(clf_params['theta'])
    var = np.array(clf_params['var'])
    
    epsilon = 1e-9
    var += epsilon

    joint_log_likelihood = []
    for i in range(len(classes)):
        log_prior = np.log(class_priors[i])
        likelihood = -0.5 * np.sum(np.log(2. * np.pi * var[i, :]) + ((final_features - theta[i, :]) ** 2) / var[i, :])
        joint_log_likelihood.append(log_prior + likelihood)

    log_prob = np.array(joint_log_likelihood)
    exp_log_prob = np.exp(log_prob - np.max(log_prob))
    probabilities = exp_log_prob / np.sum(exp_log_prob)
    
    prediction = classes[np.argmax(probabilities)]
    return prediction, probabilities.reshape(1, -1)

# --- Komponen UI ---
def build_sidebar() -> Dict[str, Any]:
    """Membangun sidebar dan mengumpulkan input dari pengguna ke dalam dictionary."""
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

    return {'DGN': dgn, 'PRE4': pre4, 'PRE5': pre5, 'PRE6': pre6,
            'PRE7': pre7, 'PRE8': pre8, 'PRE9': pre9, 'PRE10': pre10,
            'PRE11': pre11, 'PRE14': pre14, 'PRE17': pre17, 'PRE19': pre19,
            'PRE25': pre25, 'PRE30': pre30, 'PRE32': pre32, 'AGE': age}

def display_prediction_results(prediction: bool, probabilities: np.ndarray):
    """Menampilkan hasil dari prediksi manual."""
    st.subheader("Hasil Prediksi")
    if prediction:
        st.error("Prediksi: RISIKO TINGGI (Pasien diprediksi tidak akan bertahan 1 tahun)")
    else:
        st.success("Prediksi: RISIKO RENDAH (Pasien diprediksi akan bertahan 1 tahun)")

    st.subheader("Tingkat Kepercayaan Prediksi")
    proba_df_data = {
        'Kepercayaan untuk RISIKO RENDAH': [f"{probabilities[0, 0]:.2%}"],
        'Kepercayaan untuk RISIKO TINGGI': [f"{probabilities[0, 1]:.2%}"]
    }
    st.table(proba_df_data)

# --- Aplikasi Utama ---
def main():
    """Fungsi utama untuk menjalankan aplikasi Streamlit."""
    st.title("🩺 Prediksi Risiko Pasca Operasi Bedah Toraks")
    st.write("Aplikasi ini memprediksi risiko kelangsungan hidup 1 tahun pasien setelah operasi kanker paru-paru.")

    params = load_manual_params(PARAMS_PATH)
    if params:
        input_dict = build_sidebar()
        st.subheader("Ringkasan Data Pasien:")
        st.json(input_dict)
        
        if st.button("Prediksi Risiko"):
            prediction, probabilities = predict_manual(params, input_dict)
            display_prediction_results(prediction, probabilities)
            
    st.markdown("---")
    st.write("Aplikasi ini menggunakan model Gaussian Naive Bayes yang dilatih pada dataset Bedah Toraks.")

if __name__ == "__main__":
    main() 