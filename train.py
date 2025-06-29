import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Tuple
import json

from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# --- Konfigurasi Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Konstanta ---
TARGET_COLUMN = 'Risk1Yr'
MODEL_FILENAME = 'model.joblib'
BOOLEAN_COLS = [
    'PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19', 'PRE25', 
    'PRE30', 'PRE32', TARGET_COLUMN
]

def load_and_prepare_data(uci_id: int = 277) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Mengunduh data dari UCI Repo, membersihkannya, dan memisahkannya menjadi fitur dan target.
    """
    logging.info(f"Mengunduh dan memuat dataset dari UCI (ID: {uci_id})...")
    try:
        dataset = fetch_ucirepo(id=uci_id)
        x_raw = dataset.data.features
        y_raw = dataset.data.targets
        df = pd.concat([x_raw, y_raw], axis=1)
        
        for col in BOOLEAN_COLS:
            if col in df.columns:
                df[col] = df[col].map({'T': True, 'F': False})
        
        logging.info("Dataset berhasil dimuat dan dibersihkan.")
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]
        return X, y
    except Exception as e:
        logging.error(f"Gagal memuat atau memproses data: {e}")
        raise

def build_training_pipeline(X: pd.DataFrame) -> ImbPipeline:
    """Membangun pipeline untuk training (dengan SMOTE)."""
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    logging.info(f"Kolom Numerik: {numerical_cols}")
    logging.info(f"Kolom Kategorikal: {categorical_cols}")

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )

    training_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', GaussianNB())
    ])
    
    logging.info("Pipeline training berhasil dibuat.")
    return training_pipeline

def save_manual_model_params(pipeline: Pipeline, X: pd.DataFrame, path: str = "manual_model_params.json"):
    """Mengekstrak dan menyimpan parameter model untuk prediksi manual."""
    params = {}
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    # Ekstrak info dari preprocessor
    num_transformer = preprocessor.named_transformers_['num']
    cat_transformer = preprocessor.named_transformers_['cat']
    
    # Memperbaiki cara mendapatkan nama fitur.
    # Kolom sudah tersedia langsung di atribut transformers_.
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]
    
    params['numerical'] = {
        'features': num_features,
        'mean': num_transformer.mean_.tolist(),
        'scale': num_transformer.scale_.tolist()
    }
    params['categorical'] = {
        'features': cat_features,
        'categories': [c.tolist() for c in cat_transformer.categories_]
    }
    
    # Dapatkan nama fitur boolean/passthrough
    bool_features = sorted(list(set(X.columns) - set(num_features) - set(cat_features)))
    params['boolean_features'] = bool_features

    # Ekstrak info dari classifier
    params['classifier'] = {
        'class_prior': classifier.class_prior_.tolist(),
        'classes': classifier.classes_.tolist(),
        'theta': classifier.theta_.tolist(),
        'var': classifier.var_.tolist()
    }
    
    with open(path, 'w') as f:
        json.dump(params, f, indent=4)
    logging.info(f"Parameter model manual berhasil disimpan ke '{path}'.")

def train_and_save_model(
    training_pipeline: ImbPipeline, 
    X: pd.DataFrame, 
    y: pd.Series, 
    model_path: str = MODEL_FILENAME
) -> None:
    """
    Melatih model dan menyimpan pipeline prediksi yang telah dilatih dan siap digunakan.
    """
    logging.info("Memulai pelatihan model dengan SMOTE...")
    training_pipeline.fit(X, y)
    logging.info("Model berhasil dilatih.")
    
    # Membuat pipeline prediksi final dari komponen yang sudah dilatih.
    # Ini adalah cara yang lebih kuat daripada memindahkan state secara manual.
    logging.info("Membuat pipeline prediksi final untuk deployment...")
    prediction_pipeline = Pipeline(steps=[
        ('preprocessor', training_pipeline.named_steps['preprocessor']),
        ('classifier', training_pipeline.named_steps['classifier'])
    ])

    try:
        # Simpan model joblib (opsional, untuk perbandingan)
        joblib.dump(prediction_pipeline, model_path)
        logging.info(f"Model prediksi joblib telah disimpan ke '{model_path}'.")
        
        # Simpan parameter untuk model manual
        save_manual_model_params(prediction_pipeline, X)

    except Exception as e:
        logging.error(f"Gagal menyimpan model: {e}")
        raise

def main():
    """Fungsi utama untuk menjalankan seluruh proses."""
    X, y = load_and_prepare_data()
    train_pipe = build_training_pipeline(X)
    train_and_save_model(train_pipe, X, y)

if __name__ == "__main__":
    main() 