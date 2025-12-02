from src.data_loading import load_f1_data
from src.feature_engineering import engineer_features
from src.model_training import train_position_model
from src.visualization import plot_lap_times, plot_pit_stop_impact
from src.strategy_simulation import simulate_strategy
from src.driver_profiles import compute_driver_profiles
from src.race_replay import simulate_race
from src.replay_visualization import plot_race_heatmap, plot_driver_position_trends


def main():
    print("🔹 Loading data...")
    drivers, races, results, laptimes, pit = load_f1_data()

    print("🔧 Engineering features...")
    features = engineer_features(laptimes, pit)

    print("\nDEBUG — Feature columns:")
    print(features.columns.tolist())

    print("\n🤖 Training model...")
    model = train_position_model(features, results)

    print("\n📊 Plotting lap times...")
    plot_lap_times(laptimes, race_id=1, driver_id=1)

    print("\n🛠 Plotting pit stop impact...")
    plot_pit_stop_impact(laptimes, pit, driver_id=1)

    print("\n🔥 Running strategy simulation...")
    example_driver = features.iloc[0]

    pos0 = simulate_strategy(model, example_driver, 0)
    pos1 = simulate_strategy(model, example_driver, 1)
    pos2 = simulate_strategy(model, example_driver, 2)

    print(f"\nPredicted position (0-stop): {pos0:.2f}")
    print(f"Predicted position (1-stop): {pos1:.2f}")
    print(f"Predicted position (2-stop): {pos2:.2f}")

    print("\n🧠 Building driver profiles...")
    profiles = compute_driver_profiles(laptimes, results)
    print(profiles.head())

    print("\n🎮 Simulating AI race replay...")
    race_matrix = simulate_race(model, features)
    print("Replay matrix shape:", race_matrix.shape)

    print("\n🎨 Showing race heatmap...")
    plot_race_heatmap(race_matrix)

    print("\n📈 Showing race position trends...")
    num_cols = race_matrix.shape[1]
    driver_ids = list(range(1, num_cols + 1))  # safe fallback IDs
    plot_driver_position_trends(race_matrix, driver_ids)

    print("\n🏁 Race replay visualization completed.")


if __name__ == "__main__":
    main()
