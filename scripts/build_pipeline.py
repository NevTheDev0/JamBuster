import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from transformers import FeatureDropperTransformer
import joblib
import os

# === CONFIG CONSTANTS ===
CSV_PATH = "traffic_processed.csv"
PREPROCESSOR_PATH = "models/preprocessor_pipeline.pkl"
TRAIN_DATA_PATH = "models/train_ready_data.pkl"
DROP_COLS = ["timestamp", "road", "confidence", "congestion_level"]  # added congestion_level

# === Load and Validate Data ===
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# === Feature Engineering ===
df['speed_ratio'] = df['current_speed'] / df['free_flow_speed']
df['future_speed'] = df['current_speed'].shift(-3)
df['will_congest_change'] = (df['future_speed'] < 0.85 * df['free_flow_speed']).astype(int)  # more dynamic!
df.dropna(subset=['future_speed'], inplace=True)


# === Feature Columns ===
categorical = ['weather'] 
numerical = ['current_speed', 'hour']
boolean = ['rush_hour', 'is_weekend', 'low_confidence']

# === Transformers ===
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = Pipeline(steps=[
    ("dropper", FeatureDropperTransformer(drop_cols=DROP_COLS, verbose=True)),
    ("column_transformer", ColumnTransformer(transformers=[
        ("cat", cat_transformer, categorical),
        ("num", "passthrough", numerical + boolean),
    ]))
])

# === Transform Data ===
X = df.drop(columns=["congestion_level"])  # Safe, also dropped in transformer
X_transformed = preprocessor.fit_transform(X)
y = df['will_congest_change']

# === Save ===
os.makedirs("models", exist_ok=True)
joblib.dump(preprocessor, PREPROCESSOR_PATH)
joblib.dump((X_transformed, y), TRAIN_DATA_PATH)

# === Debug Output ===
print("✅ Pipeline built and data saved!")
print(f"🔢 Final shape of features: {X_transformed.shape}")
print(f"🧠 Target variable distribution:\n{y.value_counts()}")

feature_names = preprocessor.named_steps['column_transformer'].get_feature_names_out()
print(feature_names)
