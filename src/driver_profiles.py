import pandas as pd

def compute_driver_profiles(laptimes, results):
    profiles = (
        laptimes.groupby("driverId")["milliseconds"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={
            "mean": "avg_lap_ms",
            "std": "lap_std",
            "min": "best_lap_ms",
            "max": "worst_lap_ms"
        })
    )

    results_min = results[["driverId", "points"]].groupby("driverId").sum().reset_index()

    profiles = profiles.merge(results_min, on="driverId", how="left")

    return profiles
