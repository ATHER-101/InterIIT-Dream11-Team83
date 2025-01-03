import os
import pickle
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import csv
import json
import shap
from groq import Groq
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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


names_file = "../../data/raw/additional_data/names.csv"
players_file = "../../data/raw/additional_data/people.csv"
names_id_to_name = load_mappings_names(names_file)
players_id_to_name = load_mappings_players(players_file)


def load_or_train_model(
    model_file, dataset_file, train_start_date="01-01-2000", train_end_date="30-06-2024"
):

    # Check if model and scaler already exist
    if os.path.exists(model_file):
        # print("Loading existing model")
        with open(model_file, "rb") as file:
            model = pickle.load(file)
    else:
        # print("Model  not found. Training new model...")

        # Load dataset and filter training data
        df = pd.read_csv(dataset_file)
        # print("hi")

        df["dates"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

        train_start_dates = pd.to_datetime(train_start_date, format="%d-%m-%Y")
        train_end_dates = pd.to_datetime(train_end_date, format="%d-%m-%Y")

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
            ]
            + [f"opponentplayer{i}" for i in range(1, 12)]
            + [f"player{i}" for i in range(1, 11)]
        )
        input_cols = [x for x in df.columns if x not in to_remove]
        target_cols = ["total_fantasy_points"]

        train_idx = df[
            (df["dates"] <= train_end_dates) & (df["dates"] > train_start_dates)
        ].index
        X_train_df = df.loc[train_idx, input_cols]
        y_train_df = df.loc[train_idx, target_cols]

        # Normalize inputs
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df)
        y_train = y_train_df.to_numpy()

        # Train the model
        model = XGBRegressor(n_estimators=100, max_depth=5, reg_lambda=10)
        model.fit(X_train, y_train)
        # Save the model and scaler
        with open(model_file, "wb") as file:
            pickle.dump(model, file)

        # print(f"Model saved: {model_file}")

    return model


def prepare_columns(df):
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
        ]
        + [f"opponentplayer{i}" for i in range(1, 12)]
        + [f"player{i}" for i in range(1, 11)]
    )
    input_cols = [x for x in df.columns if x not in to_remove]
    target_cols = ["total_fantasy_points"]
    # print(input_cols)
    return input_cols, target_cols


# input cols:
# ['batting_points_avg_3', 'bowling_points_avg_3', 'batting_points_avg_5', 'bowling_points_avg_5', 'batting_points_avg_10', 'bowling_points_avg_10', 'batting_points_ewma',
# 'bowling_points_ewma', 'OpponentTeam_batting_points_ewma', 'OpponentTeam_bowling_points_ewma', 'OpponentTeam_batting_points_avg_5', 'OpponentTeam_bowling_points_avg_5']

import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer


def generate_predictions(format, team1_players, team2_players, test_date="01-07-2024"):
    # Load the dataset
    dataset_file = (
        f"../../data/interim/{format}_final_dataset/{format}_final_dataset_after_drop.csv"
    )
    df = pd.read_csv(dataset_file)

    # Convert date column to datetime for easier processing
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    test_date = datetime.strptime(test_date, "%d-%m-%Y")

    # List to store player data
    player_data = []

    # Combine both teams for processing
    teams = {"team1": team1_players, "team2": team2_players}

    for team_name, team_players in teams.items():
        # Opponent team players
        opponent_players = team1_players if team_name == "team2" else team2_players

        # Aggregate opponent team features
        opponent_rows = df[df["player_id"].isin(opponent_players)]
        opponent_aggregate = (
            opponent_rows[
                [
                    "batting_points_ewma",
                    "bowling_points_ewma",
                    "batting_points_avg_5",
                    "bowling_points_avg_5",
                ]
            ]
            .sum()
            .to_dict()
        )

        for player_id in team_players:
            # Filter rows for the player
            player_rows = df[df["player_id"] == player_id]

            # Find the row with the nearest date to the test_date
            nearest_row = player_rows.iloc[
                (player_rows["date"] - test_date).abs().argsort()[:1]
            ]

            if nearest_row.empty:
                player_features = {"player_id": player_id}
                player_features.update(
                    {
                        feature: None
                        for feature in [
                            "batting_points_avg_3",
                            "bowling_points_avg_3",
                            "batting_points_avg_5",
                            "bowling_points_avg_5",
                            "batting_points_avg_10",
                            "bowling_points_avg_10",
                            "batting_points_ewma",
                            "bowling_points_ewma",
                            "OpponentTeam_batting_points_ewma",
                            "OpponentTeam_bowling_points_ewma",
                            "OpponentTeam_batting_points_avg_5",
                            "OpponentTeam_bowling_points_avg_5",
                        ]
                    }
                )
            else:
                # Extract player-specific features
                player_features = nearest_row.iloc[0][
                    [
                        "batting_points_avg_3",
                        "bowling_points_avg_3",
                        "batting_points_avg_5",
                        "bowling_points_avg_5",
                        "batting_points_avg_10",
                        "bowling_points_avg_10",
                        "batting_points_ewma",
                        "bowling_points_ewma",
                    ]
                ].to_dict()

                # Add player ID
                player_features["player_id"] = player_id

                # Add opponent features
                player_features.update(
                    {
                        "OpponentTeam_batting_points_ewma": opponent_aggregate[
                            "batting_points_ewma"
                        ],
                        "OpponentTeam_bowling_points_ewma": opponent_aggregate[
                            "bowling_points_ewma"
                        ],
                        "OpponentTeam_batting_points_avg_5": opponent_aggregate[
                            "batting_points_avg_5"
                        ],
                        "OpponentTeam_bowling_points_avg_5": opponent_aggregate[
                            "bowling_points_avg_5"
                        ],
                    }
                )

            # Append to player data
            player_data.append(player_features)

    # Convert the player data to a DataFrame
    predictions_df = pd.DataFrame(player_data)
    # print(predictions_df.head())
    # print(predictions_df.columns)

    return predictions_df


def generate_predictions_with_model(
    team1_players,
    team2_players,
    format,
    test_date="01-07-2024",
    model_file_path="path/to/model.pkl",
):
    # Step 1: Generate the features DataFrame
    predictions_df = generate_predictions(
        format, team1_players, team2_players, test_date
    )

    # Step 2: Separate the player_id column
    player_ids = predictions_df["player_id"]
    feature_columns = predictions_df.drop(columns=["player_id"])

    # Step 3: Handle missing values
    imp_mean = SimpleImputer(strategy="mean")
    features_imputed = imp_mean.fit_transform(feature_columns)

    # Step 4: Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)

    # Step 5: Load the pre-trained model
    model = load_or_train_model(
        model_file_path,
        f"../../data/interim/{format}_final_dataset/{format}_final_dataset_after_drop.csv",
    )

    # Step 6: Predict points for each player
    predicted_points = model.predict(features_scaled)

    # Step 7: Map player_id to their predicted points
    player_points = dict(zip(player_ids, predicted_points))

    # Step 8: Separate players by team
    team1_points = {
        pid: player_points[pid] for pid in team1_players if pid in player_points
    }
    team2_points = {
        pid: player_points[pid] for pid in team2_players if pid in player_points
    }

    # Combine and sort all players by predicted points in descending order
    sorted_players = sorted(player_points.items(), key=lambda x: x[1], reverse=True)

    # Select top 11 players
    top_11 = sorted_players[:11]
    top_11_ids = {pid for pid, _ in top_11}

    # Check if both teams are represented
    team1_in_top = any(pid in team1_points for pid in top_11_ids)
    team2_in_top = any(pid in team2_points for pid in top_11_ids)

    # Ensure at least one player from each team is in the top 11
    if not team1_in_top:
        # Add the highest-ranked player from team1
        top_team1_player = max(team1_points.items(), key=lambda x: x[1])
        top_11.append(top_team1_player)
    elif not team2_in_top:
        # Add the highest-ranked player from team2
        top_team2_player = max(team2_points.items(), key=lambda x: x[1])
        top_11.append(top_team2_player)

    # Ensure we only return 11 players
    top_11 = sorted(top_11, key=lambda x: x[1], reverse=True)[:11]

    # Convert back to a dictionary for output
    top_11_dict = dict(top_11)
    top_11_dict = {
        k: float(v) for k, v in top_11_dict.items()
    }  # Convert to native float

    return top_11_dict


def generate_shap_values_for_players(
    top_11_players,
    names_id_to_name,
    players_id_to_name,
    format,
    test_date="01-07-2024",
    model_file_path="path/to/model.pkl",
):
    # Step 1: Generate the features DataFrame
    predictions_df = generate_predictions(
        format, team1_players, team2_players, test_date
    )

    # Step 2: Separate the player_id column
    # player_ids = predictions_df['player_id']
    player_ids = top_11_players
    feature_columns = predictions_df.drop(columns=["player_id"])

    # Step 3: Handle missing values
    imp_mean = SimpleImputer(strategy="mean")
    features_imputed = imp_mean.fit_transform(feature_columns)

    # Step 4: Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)

    # Step 5: Load the pre-trained model
    with open(model_file_path, "rb") as file:
        model = pickle.load(file)

    # Step 6: Predict points for each player
    predicted_points = model.predict(features_scaled)

    # Step 7: Map player_id to their predicted points
    player_points = dict(zip(player_ids, predicted_points))

    # Step 8: Generate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_scaled)

    # Step 9: Get feature names
    feature_names = feature_columns.columns

    # Step 10: Combine SHAP results with predictions and player IDs
    player_shap_results = []
    player_features_impact = {}  # Dictionary to store top 5 features for each player
    for i, player_id in enumerate(player_ids):
        # Create shap_details dictionary with SHAP values for each feature
        shap_details = {
            feature: round(shap_values[i, j], 4)
            for j, feature in enumerate(feature_names)
        }
        shap_details["player_id"] = player_id
        shap_details["predicted_points"] = round(predicted_points[i], 2)
        # print(shap_details)

        # Extract top 5 features, excluding 'player_id' and 'predicted_points'
        top_features = sorted(
            [
                (k, v)
                for k, v in shap_details.items()
                if k not in ["player_id", "predicted_points"]
            ],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]
        player_features_impact[player_id] = top_features

        player_shap_results.append(shap_details)

    # Step 11: Sort by predicted points and select the top 11 players
    top_players = sorted(
        player_shap_results, key=lambda x: x["predicted_points"], reverse=True
    )[:11]

    # Step 12: Convert to the desired output format
    results = {}
    for player in top_players:
        player_features = {
            k: v
            for k, v in player.items()
            if k not in ["player_id", "predicted_points"]
        }
        results[player["player_id"]] = {
            "Predicted Points": player["predicted_points"],
            "Features": player_features,
        }

    return player_features_impact


# def generate_explainability(player_id, features, api_key):
#     # Load player data
#     dataset_file = "../../data/raw/additional_data/players_data.csv"
#     if not os.path.exists(dataset_file):
#         raise FileNotFoundError(f"File not found: {dataset_file}")

#     df = pd.read_csv(dataset_file)

#     # Fetch player name and role
#     player_data = df.loc[df["player_id"] == player_id]
#     player_name = player_data["names"].values[0] if not player_data.empty else "Unknown"
#     player_role = player_data["role"].values[0] if not player_data.empty else "Unknown"

#     # Format features into a string
#     features_list = "\n".join(
#         [f"  {i+1}. {key}: {value:+.4f}" for i, (key, value) in enumerate(features)]
#     )

#     # Define the context and message
#     client = Groq(api_key=api_key)
#     prompt_message = (
#         f"Given the following impactful features and their corresponding explanations, "
#         f"generate a clear, user-friendly explanation about the predicted fantasy performance of the player. "
#         f"Use a natural and engaging tone, avoiding technical terms like SHAP values. Focus on making the explanation understandable and relatable "
#         f"to a general audience, and ensure it highlights the key factors influencing the prediction.\n\n"
#         f"### Input Format:\n"
#         f"- Player Name: {player_name}\n"
#         f"- Role: {player_role}\n"
#         f"- Most Impactful Features:\n"
#         f"{features_list}\n"
#         f"- General Context to understand the features: "
#         f"in the features if player1_batting_points_avg_10 denotes opponent player 1's batting points average over 10 matches, "
#         f"sixes_ewma denotes the exponent weighted average of sixes hit by the player of all the previous matches.\n\n"
#         f"### Output Example:\n"
#         f"'Since {player_name}'s recent performances have been exceptional, it is likely they will continue their form in this match. "
#         f"Additionally, the opponent's [specific weakness or factor] might give them an edge. The [additional context, e.g., conditions, team dynamics] "
#         f"further supports the prediction of a strong performance.'\n\n"
#         f"Generate a concise yet elaborate paragraph summarizing the input in a way that provides actionable insights to the user."
#         f"Avoid technical jargon and keep the explanation engaging and user-friendly. No additional comments or follow-ups. I just want the final short paragraph that should directly be shown to user"
#         "DO NOT GIVE ANY THING OTHER THAN THE EXPLANATION DIRRECTLY START THE EXPLANATION!!"
#     )
#     # Generate explanation
#     try:
#         chat_completion = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt_message,
#                 }
#             ],
#             model="llama3-70b-8192",
#         )
#         explanation = chat_completion.choices[0].message.content
#         # print(f"Generated explanation for {player_id}: {explanation}")  # Debugging
#         return (player_id, explanation)
#     except Exception as e:
#         print(f"Error generating explanation for {player_id}: {e}")
#         return (player_id, None)


# def parallel_generate_explanations(top_features):
#     explanations = []
#     with ThreadPoolExecutor(max_workers=len(API_KEYS)) as executor:
#         futures = {
#             executor.submit(
#                 generate_explainability,
#                 player_id,
#                 features,
#                 API_KEYS[i % len(API_KEYS)],
#             ): player_id
#             for i, (player_id, features) in enumerate(top_features.items())
#         }

#         for future in as_completed(futures):
#             try:
#                 result = future.result()
#                 if result:
#                     explanations.append(result)
#             except Exception as e:
#                 player_id = futures[future]
#                 print(f"Error processing player {player_id}: {e}")

#     return explanations

def generate_explainability(player_id, features, api_key):
    # Load player data
    dataset_file = "../../data/raw/additional_data/players_data.csv"
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"File not found: {dataset_file}")

    df = pd.read_csv(dataset_file)

    # Fetch player name and role
    player_data = df.loc[df["player_id"] == player_id]
    player_name = player_data["names"].values[0] if not player_data.empty else "Unknown"
    player_role = player_data["role"].values[0] if not player_data.empty else "Unknown"

    # Format features into a string
    features_list = "\n".join(
        [f"  {i+1}. {key}: {value:+.4f}" for i, (key, value) in enumerate(features)]
    )

    # Define the context and message
    client = Groq(api_key=api_key)
    prompt_message = (
        f"Given the following impactful features and their corresponding explanations, "
        f"generate a clear, user-friendly explanation about the predicted fantasy performance of the player. "
        f"Use a natural and engaging tone, avoiding technical terms like SHAP values. Focus on making the explanation understandable and relatable "
        f"to a general audience, and ensure it highlights the key factors influencing the prediction.\n\n"
        f"### Input Format:\n"
        f"- Player Name: {player_name}\n"
        f"- Role: {player_role}\n"
        f"- Most Impactful Features:\n"
        f"{features_list}\n"
        f"- General Context to understand the features: "
        f"in the features if player1_batting_points_avg_10 denotes opponent player 1's batting points average over 10 matches, "
        f"sixes_ewma denotes the exponent weighted average of sixes hit by the player of all the previous matches.\n\n"
        f"### Output Example:\n"
        f"'Since {player_name}'s recent performances have been exceptional, it is likely they will continue their form in this match. "
        f"Additionally, the opponent's [specific weakness or factor] might give them an edge. The [additional context, e.g., conditions, team dynamics] "
        f"further supports the prediction of a strong performance.'\n\n"
        f"Generate a concise yet elaborate paragraph summarizing the input in a way that provides actionable insights to the user."
        f"Avoid technical jargon and keep the explanation engaging and user-friendly. No additional comments or follow-ups. I just want the final short paragraph that should directly be shown to user"
        "DO NOT GIVE ANY THING OTHER THAN THE EXPLANATION DIRRECTLY START THE EXPLANATION!!"
    )
    # Generate explanation
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_message,
                }
            ],
            model="llama3-70b-8192",
        )
        explanation = chat_completion.choices[0].message.content
        return (player_id, explanation)
    except Exception as e:
        print(f"Error generating explanation for {player_id}: {e}")
        return (player_id, None)


def generate_explanations(top_features):
    explanations = []
    for i, (player_id, features) in enumerate(top_features.items()):
        try:
            api_key = API_KEYS[i % len(API_KEYS)]
            result = generate_explainability(player_id, features, api_key)
            print(result)
            if result:
                explanations.append(result)
        except Exception as e:
            print(f"Error processing player {player_id}: {e}")
    return explanations



# Example usage
# team1_players = [
#     "7ed9fd56",
#     "5bdcdb72",
#     "d18f9182",
#     "8dd02a98",
#     "896d78ad",
#     "c5928dec",
#     "9868bc75",
#     "0f12f9df",
#     "469ea22b",
#     "a12e1d51",
#     "ea0cdc12",
# ]
# team2_players = [
#     "8305580c",
#     "39f01cdb",
#     "a343262c",
#     "d2a6c0e6",
#     "cca50cd6",
#     "7ca5e05d",
#     "99b75528",
#     "e087956b",
#     "ffe699c0",
#     "49b6c09f",
#     "29b89ae8",
# ]

team1_players = arguments[1:12]
team2_players = arguments[12:23]
# model_file_path = f"../../model/model_{format}_30-06-2024.pkl"
format = arguments[0]
if(format=='T20'):
    format = 'ODI'
model_file_path = f"../../model_artifacts/model_{format}_30-06-2024.pkl"
top_11_players = generate_predictions_with_model(
    team1_players,
    team2_players,
    format,
    test_date="01-07-2024",
    model_file_path=model_file_path,
)
# print(top_11_players)
top_features = generate_shap_values_for_players(
    top_11_players,
    names_id_to_name,
    players_id_to_name,
    format,
    test_date="01-07-2024",
    model_file_path=model_file_path,
)

# print(json.dumps(top_11_players))
API_KEYS = [
    "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",
    "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",
]

explanations = generate_explanations(top_features)

# print(explanations)

# Convert to the desired JSON format
formatted_output = [{"id": item[0], "explanation": item[1]} for item in explanations]

# Dump the JSON
json_output = json.dumps(formatted_output, indent=4)

# Print the JSON
print(json_output)