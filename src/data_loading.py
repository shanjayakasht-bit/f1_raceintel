import pandas as pd

def load_f1_data():
    drivers = pd.read_csv("data/drivers.csv")
    races = pd.read_csv("data/races.csv")
    results = pd.read_csv("data/results.csv")
    laptimes = pd.read_csv("data/lap_times.csv")
    pit = pd.read_csv("data/pit_stops.csv")

    return drivers, races, results, laptimes, pit
