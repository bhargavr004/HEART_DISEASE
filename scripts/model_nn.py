import os
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils import load_splits, summarize_metrics, save_report
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

def build_model(input_dim, lr=1e-3, dropout1=0.3, dropout2=0.2):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation="relu"))
    model.add(Dropout(dropout1))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(dropout2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["AUC"])
    return model

def main(epochs=200, batch_size=32):
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # scale inputs
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = build_model(input_dim=X_train_s.shape[1], lr=1e-3)
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train_s,
        y_train.values,
        validation_data=(X_val_s, y_val.values),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=2
    )

    # save scaler (joblib) and keras model
    scaler_path = MODELS / "keras_mlp_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    keras_path = MODELS / "keras_mlp.h5"
    model.save(str(keras_path))
    print("Saved Keras model to", keras_path)

    # Evaluate on validation
    y_val_pred = (model.predict(X_val_s).ravel() >= 0.5).astype(int)
    y_val_proba = model.predict(X_val_s).ravel()

    val_metrics = summarize_metrics(y_val, y_val_pred, y_val_proba)
    save_report({"model":"keras_mlp","val_metrics":val_metrics}, "mlp_val_keras.json")

    # store metadata for easy loading
    meta = {"model_path": str(keras_path), "scaler": str(scaler_path)}
    joblib.dump(meta, MODELS / "keras_mlp_meta.joblib")
    print("Saved metadata:", meta)
    print("VAL Metrics:", val_metrics)

if __name__ == "__main__":
    main()