import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Tuple

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

def build_pipelines(X: pd.DataFrame) -> Tuple[ImbPipeline, Pipeline]:
    """
    Membangun pipeline untuk training (dengan SMOTE) dan prediksi (tanpa SMOTE).
    """
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
    
    prediction_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])
    
    logging.info("Pipeline training dan prediksi berhasil dibuat.")
    return training_pipeline, prediction_pipeline

def train_and_save_model(
    training_pipeline: ImbPipeline, 
    prediction_pipeline: Pipeline, 
    X: pd.DataFrame, 
    y: pd.Series, 
    model_path: str = MODEL_FILENAME
) -> None:
    """
    Melatih model menggunakan training pipeline dan menyimpan prediction pipeline yang telah dilatih.
    """
    logging.info("Memulai pelatihan model dengan SMOTE...")
    training_pipeline.fit(X, y)
    logging.info("Model berhasil dilatih.")
    
    # Transfer state dari pipeline terlatih ke pipeline prediksi
    prediction_pipeline.named_steps['preprocessor'].transformers_ = training_pipeline.named_steps['preprocessor'].transformers_
    prediction_pipeline.named_steps['classifier'].theta_ = training_pipeline.named_steps['classifier'].theta_
    prediction_pipeline.named_steps['classifier'].var_ = training_pipeline.named_steps['classifier'].var_
    prediction_pipeline.named_steps['classifier'].class_prior_ = training_pipeline.named_steps['classifier'].class_prior_
    prediction_pipeline.named_steps['classifier'].classes_ = training_pipeline.named_steps['classifier'].classes_

    try:
        joblib.dump(prediction_pipeline, model_path)
        logging.info(f"Model prediksi yang ringan telah disimpan ke '{model_path}'.")
    except Exception as e:
        logging.error(f"Gagal menyimpan model: {e}")
        raise

def main():
    """Fungsi utama untuk menjalankan seluruh proses."""
    X, y = load_and_prepare_data()
    train_pipe, pred_pipe = build_pipelines(X)
    train_and_save_model(train_pipe, pred_pipe, X, y)

if __name__ == "__main__":
    main() 