import pandas as pd
import numpy as np
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Load Data
print("🔄 Memuat dataset Bedah Toraks...")
thoracic_surgery_data = fetch_ucirepo(id=277)
X_raw = thoracic_surgery_data.data.features
y_raw = thoracic_surgery_data.data.targets
df = pd.concat([X_raw, y_raw], axis=1)

# Konversi kolom boolean dari 'T'/'F' menjadi Tipe Data Boolean (True/False)
bool_cols = ['PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32', 'Risk1Yr']
for col in bool_cols:
    df[col] = df[col].map({'T': True, 'F': False})
print("✅ Dataset berhasil dimuat.")

# 2. Pisahkan Fitur dan Target
X = df.drop('Risk1Yr', axis=1)
y = df['Risk1Yr']

# 3. Identifikasi tipe kolom untuk preprocessing
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
# Kolom boolean akan dilewatkan tanpa perubahan oleh ColumnTransformer
# Kita hanya perlu memastikan mereka ada di X

print(f"Kolom Numerik: {numerical_cols}")
print(f"Kolom Kategorikal: {categorical_cols}")

# 4. Buat Pipeline Preprocessing
# Pipeline untuk data numerik: scaling
numeric_transformer = StandardScaler()

# Pipeline untuk data kategorikal: one-hot encoding
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Gabungkan transformer menggunakan ColumnTransformer
# 'passthrough' akan membiarkan kolom boolean tidak diubah
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough' # Biarkan kolom boolean (PRE7, dll) apa adanya
)

# 5. Buat Pipeline Lengkap dengan SMOTE dan Model
# Kita menggunakan pipeline dari imblearn untuk mengintegrasikan SMOTE
full_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', GaussianNB())
])

# 6. Latih model dengan seluruh data
print("🔄 Melatih model Gaussian Naive Bayes dengan SMOTE...")
full_pipeline.fit(X, y)
print("✅ Model berhasil dilatih.")

# 7. Simpan pipeline lengkap ke file
joblib.dump(full_pipeline, 'model.joblib')
print("💾 Model berhasil disimpan sebagai 'model.joblib'.") 