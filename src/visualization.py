import matplotlib.pyplot as plt
import seaborn as sns

plt.ion()  # prevents plots from blocking execution


def plot_lap_times(laptimes, race_id, driver_id):
    df = laptimes[(laptimes["raceId"] == race_id) & (laptimes["driverId"] == driver_id)]

    plt.figure(figsize=(10, 5))
    plt.plot(df["lap"], df["milliseconds"], marker='o', linewidth=1)
    plt.title(f"Lap Time Trajectory – Race {race_id}, Driver {driver_id}")
    plt.xlabel("Lap Number")
    plt.ylabel("Lap Time (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pit_stop_impact(laptimes, pit, driver_id):
    driver_laps = laptimes[laptimes["driverId"] == driver_id]
    driver_pits = pit[pit["driverId"] == driver_id]

    plt.figure(figsize=(10, 5))
    plt.plot(driver_laps["lap"], driver_laps["milliseconds"], label="Lap Time")

    for _, row in driver_pits.iterrows():
        pit_lap = row["lap"]
        plt.axvline(x=pit_lap, color='red', linestyle='--', label='Pit Stop')

    plt.title(f"Pit Stop Impact – Driver {driver_id}")
    plt.xlabel("Lap Number")
    plt.ylabel("Lap Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
