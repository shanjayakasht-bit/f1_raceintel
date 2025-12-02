import numpy as np
import pandas as pd

# IMPORTANT: use the same features the model was trained on
FEATURE_COLS = ["avg_lap_ms", "lap_std", "pit_count", "pit_avg"]

def simulate_race(model, features, laps=50):
    race_predictions = []

    # Filter ONLY the columns the model knows
    base_features = features[FEATURE_COLS].copy()

    for lap in range(laps):

        # Add some lap-to-lap variation
        noisy_features = base_features.copy()
        noisy_features["lap_std"] += np.random.normal(0, 0.3, size=len(features))

        # Predict race positions
        preds = model.predict(noisy_features)

        race_predictions.append(preds)

    return np.array(race_predictions)

