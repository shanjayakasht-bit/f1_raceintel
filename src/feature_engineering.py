import pandas as pd

def engineer_features(laptimes: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    """
    Create driver-race level features:
    - avg_lap_ms: average lap time
    - lap_std: lap time consistency (std dev)
    - pit_count: number of pit stops
    - pit_avg: average pit-stop duration
    """

   
    avg_lap = (
        laptimes
        .groupby(["raceId", "driverId"])["milliseconds"]
        .mean()
        .reset_index(name="avg_lap_ms")
    )

   
    consistency = (
        laptimes
        .groupby(["raceId", "driverId"])["milliseconds"]
        .std()
        .reset_index(name="lap_std")
    )

    
    pit_info = (
        pit
        .groupby(["raceId", "driverId"])
        .agg(
            pit_count=("stop", "count"),
            pit_avg=("milliseconds", "mean")
        )
        .reset_index()
    )

   
    df = avg_lap.merge(consistency, on=["raceId", "driverId"], how="left")
    df = df.merge(pit_info, on=["raceId", "driverId"], how="left")

    
    df["pit_count"] = df["pit_count"].fillna(0)
    df["pit_avg"] = df["pit_avg"].fillna(0)

   
    df["lap_std"] = df["lap_std"].fillna(0)

    return df

