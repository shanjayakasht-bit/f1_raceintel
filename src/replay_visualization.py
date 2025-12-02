import matplotlib.pyplot as plt
import numpy as np

# ============================================================
#  HEATMAP: Predicted positions per lap
# ============================================================
def plot_race_heatmap(race_matrix):
    """
    Heatmap showing predicted positions per lap.
    race_matrix shape: (laps, drivers)
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(race_matrix, aspect="auto", cmap="viridis", origin="lower")
    plt.colorbar(label="Predicted Position")
    plt.xlabel("Driver Index")
    plt.ylabel("Lap Number")
    plt.title("Race Replay Heatmap – Predicted Positions by Lap")
    plt.tight_layout()
    plt.show()


# ============================================================
#  POSITION TRENDS: Each driver's predicted race path
# ============================================================
def plot_driver_position_trends(race_matrix, driver_ids):
    """
    Line chart of each driver's predicted position throughout the race.
    """
    laps = race_matrix.shape[0]
    num_drivers = race_matrix.shape[1]

    # prevent index errors by trimming
    driver_ids = list(driver_ids)[:num_drivers]

    plt.figure(figsize=(10, 6))

    for idx in range(num_drivers):
        plt.plot(
            range(laps),
            race_matrix[:, idx],
            label=f"Driver {driver_ids[idx]}"
        )

    plt.gca().invert_yaxis()
    plt.xlabel("Lap")
    plt.ylabel("Predicted Position")
    plt.title("Driver Position Trends During AI-Simulated Race")
    plt.legend(loc="upper right")
    plt.subplots_adjust(bottom=0.15, top=0.90)

    plt.show()


# ============================================================
#  OPTIONAL: GIF / VIDEO FRAME GENERATION
# ============================================================
def generate_race_frames(race_matrix, driver_ids):
    frames = []
    laps = race_matrix.shape[0]
    num_drivers = race_matrix.shape[1]

    driver_ids = list(driver_ids)[:num_drivers]

    for lap in range(laps):
        fig, ax = plt.subplots(figsize=(8, 4))
        positions = race_matrix[lap]

        ax.bar(driver_ids, positions, color="orange")
        ax.set_ylim(max(positions) + 2, 1)
        ax.set_ylabel("Predicted Position")
        ax.set_title(f"Lap {lap + 1} – AI Race Simulation")

        frames.append(fig)

    return frames
