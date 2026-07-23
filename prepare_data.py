from src.utils.config_loader import load_config
from src.data_layer.loader import load_raw
from src.feature_layer.graph_features import add_graph_features
from src.feature_layer.preprocessing import clean_and_encode, temporal_split
import pickle
import os

print("Loading configuration...")
cfg = load_config()

print("Loading raw data...")
df = load_raw(cfg["data"]["transaction_path"], cfg["data"]["identity_path"])

print("Adding graph features...")
df = add_graph_features(df, cfg)

print("Cleaning and encoding...")
df, encoders = clean_and_encode(df, cfg)

print("Applying temporal split...")
X_train, X_test, y_train, y_test, feature_names, spw = temporal_split(df, cfg)

os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/train_featured.pkl'
print(f"Saving to {output_path}...")

payload = (X_train, X_test, y_train, y_test, feature_names)

with open(output_path, 'wb') as f:
    pickle.dump(payload, f)

print("Done! Data saved successfully.")
