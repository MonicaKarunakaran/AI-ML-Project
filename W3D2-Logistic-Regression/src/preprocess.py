import os

import joblib

from sklearn.preprocessing import StandardScaler

from src.config import *


def preprocess_data(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(scaler, SCALER_PATH)

    return X_train_scaled, X_test_scaled