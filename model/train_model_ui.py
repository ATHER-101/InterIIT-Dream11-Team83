import pandas as pd
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
import pickle
import csv

import sys
arguments = sys.argv[1:]


def load_mappings_names(file_path):
    mapping = {}
    with open(file_path, mode="r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            mapping[row["name"].strip()] = str(row["player_id"])
    return mapping


def load_mappings_players(file_path):
    mapping = {}
    with open(file_path, mode="r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            mapping[row["name"].strip()] = str(row["player_id"])
    return mapping


# Load mappings from names.csv and players.csv
names_file = "../../data/raw/additional_data/names.csv"
players_file = "../../data/raw/additional_data/people.csv"
names_name_to_id = load_mappings_names(names_file)
players_name_to_id = load_mappings_players(players_file)

# Reverse mappings for ID to Name
names_id_to_name = {v: k for k, v in names_name_to_id.items()}
players_id_to_name = {v: k for k, v in players_name_to_id.items()}


def generate_model(
    format,
    train_start_date="01-01-2000",
    train_end_date="30-06-2024",
    test_start_date="01-07-2024",
    test_end_date="10-11-2024",
):
    df = pd.read_csv(
        f"../../data/interim/{format}_final_dataset/{format}_final_dataset_after_drop.csv"
    )
    df["dates"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    to_remove = (
        [
            "player_id",
            "match_id",
            "date",
            "dates",
            "role",
            "batting_points",
            "bowling_points",
            "fielding_points",
            "total_fantasy_points",
            "fielding_points_ewma",
            "Team_batting_points_avg_5",
            "Team_bowling_points_avg_5",
            "Allrounder",
            "Batsman",
            "Bowler",
            "Other",
            "opponentplayer_batting_points_rank1",
            "opponentplayer_batting_points_rank2",
            "opponentplayer_batting_points_rank3",
            "opponentplayer_batting_points_rank4",
            "opponentplayer_batting_points_rank5",
            "opponentplayer_bowling_points_rank1",
            "opponentplayer_bowling_points_rank2",
            "opponentplayer_bowling_points_rank3",
            "opponentplayer_bowling_points_rank4",
            "opponentplayer_bowling_points_rank5",
        ]
        + [f"opponentplayer{i}" for i in range(1, 12)]
        + [f"player{i}" for i in range(1, 11)]
    )
    input_cols = [x for x in df.columns if x not in to_remove]
    print(input_cols)
    target_cols = ["total_fantasy_points"]

    from sklearn.preprocessing import StandardScaler

    train_start_dates = pd.to_datetime(train_start_date, format="%d-%m-%Y")
    train_end_dates = pd.to_datetime(train_end_date, format="%d-%m-%Y")
    test_start_dates = pd.to_datetime(test_start_date, format="%d-%m-%Y")
    test_end_dates = pd.to_datetime(test_end_date, format="%d-%m-%Y")

    # Generate train and test indices
    train_idx = df[
        (df["dates"] <= train_end_dates) & (df["dates"] > train_start_dates)
    ].index
    test_idx = df[
        (df["dates"] <= test_end_dates) & (df["dates"] > test_start_dates)
    ].index

    # Apply indices to the original dataframe
    X_train_df = df.loc[train_idx, input_cols]
    y_train_df = df.loc[train_idx, target_cols]
    df_combined = pd.concat(
        [X_train_df, y_train_df]
    )  # Save the combined DataFrame to a CSV file df_combined.to_csv('combined_data.csv', index=False)
    df_combined.to_csv(
        f"../../data/processed/training_data_{format}_{train_end_date}.csv", index=False
    )
    X_test_df = df.loc[test_idx, input_cols]
    y_test_df = df.loc[test_idx, target_cols]

    # Normalize inputs (convert to numpy array)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    # Convert targets to numpy arrays
    y_train = y_train_df.to_numpy()
    y_test = y_test_df.to_numpy()

    # from sklearn.impute import SimpleImputer
    # imp_mean = SimpleImputer( strategy='mean')
    # X_train=imp_mean.fit_transform(X_train)
    # X_test=imp_mean.fit_transform(X_test)

    model2 = XGBRegressor(n_estimators=10, max_depth=5, reg_lambda=10)
    model2.fit(X_train, y_train)
    with open(
        f"../../model_artifacts/model_{format}_{train_end_date}.pkl", "wb"
    ) as file:
        pickle.dump(model2, file)
        print(f"Model saved as model_{format}_{train_end_date}.pkl")

    test_df = df.loc[test_idx]
    predictions = model2.predict(X_test)
    test_df["Predicted Points"] = predictions

    results = []
    for match_date, match_data in test_df.groupby("date"):
        # Map IDs to names

        match_id = match_data["match_id"].iloc[0]

        match_data["Player Name"] = match_data["player_id"].map(
            lambda pid: names_id_to_name.get(
                pid, players_id_to_name.get(pid, "Unknown(" + str(pid) + ")")
            )
        )

        # Extract players for each team
        team1_players = (
            match_data.filter(regex="^player[1-9]$").iloc[0].dropna().tolist()
        )
        team2_players = (
            match_data.filter(regex="^opponentplayer[1-9]$").iloc[0].dropna().tolist()
        )

        # Assign team labels based on team ID
        match_data["Team"] = match_data["player_id"].apply(
            lambda pid: "Team 1" if pid in team1_players else "Team 2"
        )

        # Sort predicted points
        predicted_team = match_data.sort_values(by="Predicted Points", ascending=False)

        # Ensure at least one player from each team in the predicted team
        top_team1_player = predicted_team[predicted_team["Team"] == "Team 1"].head(1)
        top_team2_player = predicted_team[predicted_team["Team"] == "Team 2"].head(1)

        # Exclude already selected players and fill the rest of the top 11
        remaining_predicted_players = predicted_team.drop(
            index=top_team1_player.index.union(top_team2_player.index)
        )
        final_predicted_team = pd.concat(
            [top_team1_player, top_team2_player, remaining_predicted_players.head(9)]
        )

        # Calculate weighted fantasy score for predicted team
        final_predicted_team["Weighted Points"] = final_predicted_team[
            "Predicted Points"
        ].values
        final_predicted_team["Weighted Points"] = final_predicted_team[
            "Weighted Points"
        ].astype(float)

        final_predicted_team.iloc[
            0, final_predicted_team.columns.get_loc("Weighted Points")
        ] *= 2  # Top player 2x
        final_predicted_team.iloc[
            1, final_predicted_team.columns.get_loc("Weighted Points")
        ] *= 1.5  # Second player 1.5x
        predicted_team_score = final_predicted_team["Weighted Points"].sum()

        # Sort actual points
        actual_team = match_data.sort_values(by="total_fantasy_points", ascending=False)

        # Ensure at least one player from each team in the actual team
        top_team1_actual = actual_team[actual_team["Team"] == "Team 1"].head(1)
        top_team2_actual = actual_team[actual_team["Team"] == "Team 2"].head(1)

        # Exclude already selected players and fill the rest of the top 11
        remaining_actual_players = actual_team.drop(
            index=top_team1_actual.index.union(top_team2_actual.index)
        )
        final_actual_team = pd.concat(
            [top_team1_actual, top_team2_actual, remaining_actual_players.head(9)]
        )

        # Calculate weighted fantasy score for actual team
        final_actual_team["Weighted Points"] = final_actual_team[
            "total_fantasy_points"
        ].values
        final_actual_team["Weighted Points"] = final_actual_team[
            "Weighted Points"
        ].astype(float)

        final_actual_team.iloc[
            0, final_actual_team.columns.get_loc("Weighted Points")
        ] *= 2  # Top player 2x
        final_actual_team.iloc[
            1, final_actual_team.columns.get_loc("Weighted Points")
        ] *= 1.5  # Second player 1.5x
        actual_team_score = final_actual_team["Weighted Points"].sum()

        # Calculate MAE
        mae = abs(predicted_team_score - actual_team_score)
        if actual_team_score != 0:
            mape = (
                abs((actual_team_score - predicted_team_score) / actual_team_score)
                * 100
            )
        else:
            mape = None
        # Build alternating players and points
        predicted_players_and_points = [
            (
                final_predicted_team.iloc[i]["Player Name"],
                final_predicted_team.iloc[i]["Predicted Points"],
            )
            for i in range(11)
        ]
        actual_players_and_points = [
            (
                final_actual_team.iloc[i]["Player Name"],
                final_actual_team.iloc[i]["total_fantasy_points"],
            )
            for i in range(11)
        ]

        # Build result row with alternating columns
        result_row = {
            "match_id": match_id,
            "Match Date": match_date,
            **{f"player{i+1}": predicted_players_and_points[i][0] for i in range(11)},
            **{
                f"predicted_points{i+1}": predicted_players_and_points[i][1]
                for i in range(11)
            },
            **{
                f"actual_player{i+1}": actual_players_and_points[i][0]
                for i in range(11)
            },
            **{
                f"actual_points{i+1}": actual_players_and_points[i][1]
                for i in range(11)
            },
            "Predicted Total Points": predicted_team_score,
            "Actual Total Points": actual_team_score,
            "MAE": mae,
            "MAPE (%)": mape,  # Include MAPE
        }

        # Append result row
        results.append(result_row)

    # Save results to CSV
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        f"../../data/processed/test_results_{format}_{test_end_date}.csv", index=False
    )
    print("Test results saved.")


generate_model("Test", arguments[0], arguments[1], arguments[2], arguments[3])
generate_model("ODI", arguments[0], arguments[1], arguments[2], arguments[3])
generate_model("T20", arguments[0], arguments[1], arguments[2], arguments[3])

# List of CSV file paths
csv_files = [
    f"../../data/processed/test_results_Test_{arguments[3]}.csv",
    f"../../data/processed/test_results_ODI_{arguments[3]}.csv",
    f"../../data/processed/test_results_T20_{arguments[3]}.csv",
]

# Read and concatenate all CSV files
dataframes = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(dataframes, ignore_index=True)

# Convert "Match Date" to datetime format
combined_df["Match Date"] = pd.to_datetime(combined_df["Match Date"])

# Sort by "Match Date"
sorted_df = combined_df.sort_values(by="Match Date")

# Save to a new CSV file (optional)
sorted_df.to_csv(f"../../data/processed/test_results_{arguments[3]}.csv", index=False)

delete_files = [
    f"../../data/processed/test_results_Test_{arguments[3]}.csv",
    f"../../data/processed/test_results_ODI_{arguments[3]}.csv",
    f"../../data/processed/test_results_T20_{arguments[3]}.csv",
    f"../../data/processed/training_data_Test_{arguments[1]}.csv",
    f"../../data/processed/training_data_ODI_{arguments[1]}.csv",
    f"../../data/processed/training_data_T20_{arguments[1]}.csv",
]
for file in delete_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted {file}")
print("Files concatenated and sorted successfully!")
