# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor   # <-- XGBoost

# ----------------------------------------------------------
# 🔥 Feature columns used during both training & simulation
# ----------------------------------------------------------
FEATURE_COLS = ["avg_lap_ms", "lap_std", "pit_count", "pit_avg"]


def train_position_model(features: pd.DataFrame, results: pd.DataFrame):
    """
    Merge engineered features with race results and train a model to predict finishing positionOrder.
    """

    results_min = results[["raceId", "driverId", "positionOrder"]].dropna()

    data = features.merge(
        results_min,
        on=["raceId", "driverId"],
        how="inner"
    )

    data = data.dropna(subset=FEATURE_COLS + ["positionOrder"])

    X = data[FEATURE_COLS]
    y = data["positionOrder"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------
    # 🔥 XGBoost Regressor (new improved model)
    # ---------------------------------------
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"🔥 XGBoost Model MAE (lower is better): {mae:.3f}")

    return model

