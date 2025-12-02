# src/strategy_simulation.py

import pandas as pd

def simulate_strategy(model, driver_features, pit_stop_count=1):
    """
    Simulate finishing position based on different pit stop strategies.

    model: trained sklearn model
    driver_features: 1 row of engineered feature data
    pit_stop_count: number of pit stops to simulate
    """

    row = driver_features.copy()
    row["pit_stops"] = pit_stop_count  # override feature for simulation

    prediction = model.predict([row])[0]
    return prediction

from src.model_training import FEATURE_COLS

def simulate_strategy(model, driver_features, pit_stop_count=1):
    """
    Simulate finishing position for different pit-stop scenarios.
    """

    row = driver_features.copy()
    row["pit_stops"] = pit_stop_count

    # IMPORTANT: Only use the training feature columns
    row = row[FEATURE_COLS]

    prediction = model.predict([row])[0]
    return prediction

from src.model_training import FEATURE_COLS
import pandas as pd

def simulate_strategy(model, driver_features, pit_stop_count=1):
    """
    Simulate different pit stop strategies using the trained model.
    """

    row = driver_features.copy()

    # modify pit count for simulation
    row["pit_count"] = pit_stop_count

    # keep only training columns
    row = row[FEATURE_COLS]

    # convert to DataFrame to avoid sklearn warnings
    row_df = pd.DataFrame([row])

    prediction = model.predict(row_df)[0]
    return prediction

