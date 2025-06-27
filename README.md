# 🩺 Prediksi Risiko Pasca Operasi Bedah Toraks

Aplikasi Streamlit ini memprediksi risiko kelangsungan hidup satu tahun untuk pasien setelah operasi kanker paru-paru. Prediksi dibuat berdasarkan model Gaussian Naive Bayes yang dilatih pada dataset Bedah Toraks dari UCI Repository.

## ✨ Fitur

- Antarmuka yang ramah pengguna untuk memasukkan data pra-operasi pasien.
- Prediksi risiko (Rendah/Tinggi) secara *real-time*.
- Menampilkan tingkat kepercayaan dari prediksi.
- Aplikasi ringan dan cepat berkat *caching* model dan manajemen dependensi yang efisien.

## 🛠️ Struktur Proyek

```
pendat/
├── .gitignore          # File yang diabaikan oleh Git
├── app.py              # Kode utama aplikasi Streamlit
├── model.joblib        # File model machine learning yang sudah dilatih
├── requirements-train.txt # Dependensi untuk melatih model
├── requirements.txt    # Dependensi untuk menjalankan aplikasi
├── train.py            # Skrip untuk melatih model
└── README.md           # Anda sedang membacanya
```

## 🚀 Cara Menjalankan

### 1. Prasyarat

- Python 3.9+
- `pip` untuk manajemen paket

### 2. Setup Lingkungan

Clone repositori ini dan buatlah sebuah *virtual environment*:

```bash
git clone <URL_REPOSITORI_ANDA>
cd pendat
python -m venv venv
source venv/bin/activate  # Di Windows, gunakan: venv\Scripts\activate
```

### 3. Melatih Model (Opsional)

Model (`model.joblib`) sudah termasuk di dalam repositori. Namun, jika Anda ingin melatihnya kembali:

1.  Instal dependensi untuk pelatihan:
    ```bash
    pip install -r requirements-train.txt
    ```
2.  Jalankan skrip pelatihan:
    ```bash
    python train.py
    ```

### 4. Menjalankan Aplikasi Streamlit

1.  Instal dependensi untuk aplikasi:
    ```bash
    pip install -r requirements.txt
    ```

2.  Jalankan aplikasi Streamlit:
    ```bash
    streamlit run app.py
    ```

Buka browser Anda dan navigasi ke alamat lokal yang ditampilkan (biasanya `http://localhost:8501`). 