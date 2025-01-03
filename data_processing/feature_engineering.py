import os
import json
from datetime import datetime
import shutil
from collections import defaultdict
import csv
import pandas as pd
import numpy as np

# # Code to split data into three folders
def split_data(x):
    # Path to the source folder
    source_folder = "../data/raw/cricsheet_data/all_json"
    location_folder="../data/raw/cricsheet_data/"
    # Target folders
    target_folders = {
        match_type: os.path.join(location_folder, match_type) for match_type in x
    }

    # Recreate target folders
    for folder in target_folders.values():
        if os.path.exists(folder):
            # shutil.rmtree(folder)  # Remove the folder if it exists
            continue
        os.makedirs(folder)  # Create the folder

    # Date limit for filtering
    date_limit = datetime.strptime("2024-11-10", "%Y-%m-%d")

    # Process files
    for filename in os.listdir(source_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(source_folder, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                
                # Extract metadata
                created_date_str = data.get("meta", {}).get("created", "")
                match_type = data.get("info", {}).get("match_type", "")
                
                if created_date_str and match_type:
                    created_date = datetime.strptime(created_date_str, "%Y-%m-%d")
                    
                    # Filter by date and match type
                    if created_date <= date_limit and match_type in target_folders:
                        target_path = os.path.join(target_folders[match_type], filename)
                        shutil.move(file_path, target_path)
                        print(f"Moved {filename} to {match_type} folder")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
#  #----------------------------------------------------------------------------------------------------------------------------------------------------
# # code to create player wise datasets
def extract_playerwise_datasets(x):
    from fantasy_calculator import (
        fantasy_calculator_T20,
        fantasy_calculator_ODI,
        fantasy_calculator_Test,
    ) 

    def process_match_data(json_file, output_folder):
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        player_info = {}
        with open('../data/raw/additionaldata/players_data.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                player_id = row["player_id"]
                if player_id:
                    player_info[player_id] = {
                        "role": row.get("role", ""),
                        "batting_style": row.get("batting_style", ""),
                        "allrounder_status": "Allrounder" if "allrounder" in row.get("role", "").lower() else "Not Allrounder",
                        "bowling_hand": row.get("bowling_style", "").split('-')[0].strip() if row.get("bowling_style") else "",
                        "bowling_type": row.get("bowling_style", "").split(' ')[1].strip() if ' ' in row.get("bowling_style", "") else ""
                    }

        # Extract match info
        match_date = data["info"]["dates"][0]
        venue = data["info"]["venue"]
        event = data["info"].get("event", {}).get("name", "Unknown Event")
        match_type = data["info"]["match_type"]
        gender=data["info"]["gender"]
        teams = data["info"]["teams"]
        innings_data = data["innings"]

        # Map player names to player IDs using the 'registry' section
        player_id_map = {name: player_id for name, player_id in data["info"]["registry"]["people"].items()}
        match_id = os.path.basename(json_file).replace('.json', '')

        # Map players to their respective teams
        player_team_map = {}
        for team in teams:
            for player in data["info"]["players"][team]:
                player_team_map[player] = team

        # Prepare player stats dictionary
        player_stats = defaultdict(lambda: {
            "match_id": match_id,
            "date": match_date,
            "venue": venue,
            "event": event,
            "match_type": match_type,
            "team": None,
            "gender":gender,
            "runs_scored": 0,
            "balls_faced": 0,
            "boundaries": 0,
            "sixes": 0,
            "balls_bowled": 0,
            "wickets": 0,
            "runs_given": 0,
            "bowled_lbw": 0,
            "maidens": 0,
            "catches": 0,
            "stumpings": 0,
            "run_outs": 0
        })

        # Initialize player stats with team information from the rosters
        for player, team in player_team_map.items():
            player_stats[player]["team"] = team

        # Process innings
        for inning in innings_data:
            team = inning["team"]
            overs = inning.get("overs", [])
            for over in overs:
                maidens_flag = True  # To track maiden overs
                for delivery in over["deliveries"]:
                    # Batting stats
                    batter = delivery["batter"]
                    bowler = delivery["bowler"]
                    runs = delivery["runs"]["batter"]
                    player_stats[batter]["runs_scored"] += runs
                    player_stats[batter]["balls_faced"] += 1
                    if runs == 4:
                        player_stats[batter]["boundaries"] += 1
                    if runs == 6:
                        player_stats[batter]["sixes"] += 1

                    # Wicket stats
                    if "wickets" in delivery:
                        for wicket in delivery["wickets"]:
                            kind = wicket["kind"]
                            player_out = wicket["player_out"]

                            # Increment the bowler's wickets
                            player_stats[bowler]["wickets"] += 1

                            if kind in {"bowled", "lbw"}:
                                player_stats[bowler]["bowled_lbw"] += 1
                            if kind == "caught":
                                for fielder in wicket.get("fielders", []):
                                    if(('substitute' in fielder.keys()) and len(fielder) != 2):
                                        continue
                                    
                                    player_stats[fielder["name"]]["catches"] += 1
                            if kind == "stumped":
                                
                                player_stats[bowler]["stumpings"] += 1
                            if kind == "run out":
                                for fielder in wicket.get("fielders", []):
                                    if(('substitute' in fielder.keys()) and len(fielder) != 2):
                                        continue
                                    player_stats[fielder["name"]]["run_outs"] += 1

                    # Bowling stats
                    if "extras" not in delivery or "wides" not in delivery["extras"] and "no_balls" not in delivery["extras"]:
                        player_stats[bowler]["balls_bowled"] += 1  # Count only legitimate balls

                    player_stats[bowler]["runs_given"] += delivery["runs"]["total"]
                    if delivery["runs"]["total"] > 0 or "extras" in delivery:
                        maidens_flag = False

                # Update maidens if all deliveries in the over were zero runs
                if maidens_flag:
                    player_stats[bowler]["maidens"] += 1

        # Add opponent and current playing XI
        for player in player_stats:
            team = player_stats[player]["team"]
            opponent_team = teams[1] if teams[0] == team else teams[0]
            current_team = team

            # Assign opponent XI
            opponent_playing_xi = data["info"]["players"].get(opponent_team, [])
            for i, opponent_player in enumerate(opponent_playing_xi[:11]):
                player_stats[player][f"opponentplayer{i+1}"] = player_id_map.get(opponent_player, "Unknown_Player")

            # Assign current XI
            current_playing_xi = data["info"]["players"].get(current_team, [])
            teammates = [p for p in current_playing_xi[:11] if p != player]
            for i, current_player in enumerate(teammates[:10]):
                player_stats[player][f"player{i+1}"] = player_id_map.get(current_player, "Unknown_Player")

        # Create 'player_data' folder if it does not exist
        os.makedirs(output_folder, exist_ok=True)

        # Write to individual player CSV files
        for player, stats in player_stats.items():
            # Get player ID from the map
            player_id = player_id_map.get(player, player)  # Fallback to player name if ID is not found

            # Define player file path
            player_file = os.path.join(output_folder, f"{player_id}.csv")
            
            # Check if file exists, if not create it with header
            file_exists = os.path.isfile(player_file)
            
            header = [
                "match_id", "date", "venue", "event", "match_type", "player_id", "team","gender",
                "runs_scored", "balls_faced", "boundaries", "sixes", "balls_bowled", "wickets", 
                "runs_given", "bowled_lbw", "maidens", "catches", "stumpings", "run_outs", 
                "role", "batting_style", "allrounder_status", "bowling_hand", "bowling_type",
                "batting_points", "bowling_points", "fielding_points", "total_fantasy_points",
                
                "player1", "player2", "player3", "player4", "player5", "player6", 
                "player7", "player8", "player9", "player10", 
                "opponentplayer1", "opponentplayer2", "opponentplayer3", "opponentplayer4",
                "opponentplayer5", "opponentplayer6", "opponentplayer7", "opponentplayer8",
                "opponentplayer9", "opponentplayer10", "opponentplayer11"
            ]
            
            # Open the player CSV file
            with open(player_file, 'a', newline='') as f:
                writer = csv.writer(f)
                # Write the header if the file does not exist
                if not file_exists:
                    writer.writerow(header)
                

                if match_type == "T20":
                    fantasy_calculator = fantasy_calculator_T20
                elif match_type == "ODI":
                    fantasy_calculator = fantasy_calculator_ODI
                elif match_type == "Test":
                    fantasy_calculator = fantasy_calculator_Test
                else:
                    raise ValueError(f"Unsupported match type: {match_type}")
                
                # Calculate fantasy points
                batting_points, bowling_points, fielding_points, total_fantasy_points = fantasy_calculator(stats)
                
                # Prepare data for this row
                # row = [stats.get(col, 0) for col in header[:21]]  # Exclude fantasy and player columns

                row = [
                    stats.get("match_id", ""),
                    stats.get("date", ""),
                    stats.get("venue", ""),
                    stats.get("event", ""),
                    stats.get("match_type", ""),
                    player_id,  # Replace with player_id
                    stats.get("team", ""),
                    stats.get("gender",""),
                    stats.get("runs_scored", 0),
                    stats.get("balls_faced", 0),
                    stats.get("boundaries", 0),
                    stats.get("sixes", 0),
                    stats.get("balls_bowled", 0),
                    stats.get("wickets", 0),
                    stats.get("runs_given", 0),
                    stats.get("bowled_lbw", 0),
                    stats.get("maidens", 0),
                    stats.get("catches", 0),
                    stats.get("stumpings", 0),
                    stats.get("run_outs", 0),
                    player_info.get(player_id, {}).get("role", "Unknown"),
                    player_info.get(player_id, {}).get("batting_style", "Unknown"),
                    player_info.get(player_id, {}).get("allrounder_status", "Unknown"),
                    player_info.get(player_id, {}).get("bowling_hand", "Unknown"),
                    player_info.get(player_id, {}).get("bowling_type", "Unknown")
                ]
                row[5] = player_id  # Replace with player_id
                row.extend([batting_points, bowling_points, fielding_points, total_fantasy_points])
                row.extend([stats.get(f"player{i+1}", "Unknown_Player") for i in range(10)])  # Add current XI
                row.extend([stats.get(f"opponentplayer{i+1}", "Unknown_Player") for i in range(11)])  # Add opponent XI
                writer.writerow(row)

    for match_type in x:
        folder_path = f"../data/raw/cricsheet_data/{match_type}"
        storage_path=f"../data/interim/{match_type}"
        if os.path.exists(storage_path+"_processed"):
            shutil.rmtree(storage_path+"_processed")  # Remove the folder if it exists
        os.makedirs(storage_path+"_processed")

        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Check if it is a file (not a folder)
            if os.path.isfile(file_path) & filename.endswith(".json"):
                process_match_data(file_path, storage_path+"_processed")

# #-----------------------------------------------------------------------------------------------------------------------------------------------------------

# # Code to create consolidated files
def create_consolidated_files(x):
    for match_type in x:
        # Path to the directory containing the CSV files
        data_dir = f"../data/interim/{match_type}_processed"

        # List all the CSV files in the directory
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

        # Initialize an empty DataFrame to store consolidated data
        consolidated_data = []

        # Function to calculate strike rate
        def calculate_strike_rate(runs, balls):
            return (runs / balls * 100) if balls > 0 else 0



        # Process each player's file
        for file in files:
            file_path = os.path.join(data_dir, file)

            # Read the CSV file
            player_data = pd.read_csv(file_path,parse_dates=['date'], quotechar='"', sep=',')

            # Sort by date to ensure chronological order
            player_data['date'] = pd.to_datetime(player_data['date'])
            player_data = player_data.sort_values(by='date')

            # Initialize a list for storing computed rows
            computed_rows = []

            # Loop through each row in the player's dataset
            for i in range(len(player_data)):
                row = player_data.iloc[i]

                # Calculate metrics
                strike_rate = calculate_strike_rate(row['runs_scored'], row['balls_faced'])  

                

            

                # Variance in total fantasy points before the current match

                # Append computed metrics
                computed_row = {
                    'match_id': row['match_id'],
                    'player_id':row['player_id'],
                    'date': row['date'],
                    'venue': row['venue'],
                    'strike_rate': strike_rate,
                    'wickets': row['wickets'],
                    'maidens': row['maidens'],
                    'bowled_lbw': row['bowled_lbw'],
                    'catches': row['catches'],
                    'stumpings': row['stumpings'],
                    'role':row['role'],
                    'batting_style':row['batting_style'],
                    'allrounder_status':row['allrounder_status'],
                    'bowling_hand':row['bowling_hand'],
                    'bowling_type':row['bowling_type'],
                    "player1":row['player1'],
                    "player2":row['player2'],
                    "player3":row['player3'],
                    "player4":row['player4'],
                    "player5":row['player5'],
                    "player6":row['player6'],
                    "player7":row['player7'],
                    "player8":row['player8'],
                    "player9":row['player9'],
                    "player10":row['player10'],
                    "opponentplayer1":row['opponentplayer1'],
                    "opponentplayer2":row['opponentplayer2'],
                    "opponentplayer3":row['opponentplayer3'],
                    "opponentplayer4":row['opponentplayer4'],
                    "opponentplayer5":row['opponentplayer5'],
                    "opponentplayer6":row['opponentplayer6'],
                    "opponentplayer7":row['opponentplayer7'],
                    "opponentplayer8":row['opponentplayer8'],
                    "opponentplayer9":row['opponentplayer9'],
                    "opponentplayer10":row['opponentplayer10'],
                    "opponentplayer11":row['opponentplayer11'],
                    "batting_points":row['batting_points'],
                    "bowling_points": row['bowling_points'],
                    "fielding_points": row['fielding_points'],
                    "total_fantasy_points": row['total_fantasy_points'],
                    "sixes":row["sixes"],
                    "boundaries":row["boundaries"]

                    
                }
                computed_rows.append(computed_row)

            # Add computed rows to the consolidated dataset
            consolidated_data.extend(computed_rows)

        # Create a DataFrame from the consolidated data
        consolidated_df = pd.DataFrame(consolidated_data)

        # Save the consolidated dataset to a CSV file
        consolidated_df.to_csv(f"../data/interim/new_dataset_{match_type}.csv", index=False)



# code to generate final datasets
def generate_final_dataset(x):
    for match_type in x:
        data = pd.read_csv(f"../data/interim/new_dataset_{match_type}.csv", parse_dates=['date'])

        # Ensure data is sorted by player and date
        data = data.sort_values(by=['player_id', 'date'])

        # List of numeric columns to compute averages
        numeric_columns = [
            'strike_rate', 'wickets', 'maidens', 'bowled_lbw',
            'catches', 'stumpings', 'sixes', 'boundaries'
        ]
        points_columns = ['batting_points', 'bowling_points', 'fielding_points']

        # Additional columns to extract for each player
        player_attributes = ['role', 'batting_style', 'allrounder_status', 'bowling_hand', 'bowling_type']

        # Initialize a new DataFrame for results
        results = []

        grouped = data.groupby('player_id')

        # Predefined lists for friend and opponent attributes
        friend_columns = [f"player{i}" for i in range(1, 11)]
        opponent_columns = [f"opponentplayer{i}" for i in range(1, 12)]

        # Compute rolling averages and variances for each player
        for player_id, group in grouped:
            group = group.sort_values(by='date').reset_index(drop=True)
            
            ewma_alpha = 0.3  # Optimal alpha value
            ewma_cols = {col: group[col].shift(1).ewm(alpha=ewma_alpha, adjust=False).mean() for col in numeric_columns + points_columns}
            
            for i, row in group.iterrows():
                previous_matches = group.iloc[:i]
                
                # Compute averages and variances
                stats = {}
                for window in [3, 5, 10]:
                    for col in numeric_columns + points_columns:
                        stats[f"{col}_avg_{window}"] = previous_matches[col].tail(window).mean() if not previous_matches.empty else np.nan
                        stats[f"{col}_var_{window}"] = previous_matches[col].tail(window).var() if not previous_matches.empty else np.nan
                
                # Add EWMA values
                for col, ewma_series in ewma_cols.items():
                    stats[f"{col}_ewma"] = ewma_series.iloc[i] if i > 0 else np.nan
                
                # Extract player attributes
                player_attrs = {f"player_{key}": row[key] for key in player_attributes}
                
                # Add friend and opponent attributes systematically
                for friend in friend_columns:
                    friend_id = row[friend]
                    if pd.notna(friend_id) and friend_id in grouped.groups:
                        friend_group = grouped.get_group(friend_id)
                        friend_matches = friend_group[friend_group['date'] < row['date']]
                        for window in [3, 5, 10]:
                            for col in points_columns:
                                stats[f"{friend}_{col}_avg_{window}"] = friend_matches[col].tail(window).mean() if not friend_matches.empty else np.nan
                                stats[f"{friend}_{col}_var_{window}"] = friend_matches[col].tail(window).var() if not friend_matches.empty else np.nan
                        for col in points_columns:
                            ewma_series = friend_group[col].shift(1).ewm(alpha=ewma_alpha, adjust=False).mean()
                            stats[f"{friend}_{col}_ewma"] = ewma_series.loc[friend_group['date'] < row['date']].iloc[-1] if not friend_matches.empty else np.nan
                        if not friend_matches.empty:
                            latest_row = friend_matches.iloc[-1]
                            for attr in player_attributes:
                                stats[f"{friend}_{attr}"] = latest_row[attr]
                
                for opponent in opponent_columns:
                    opponent_id = row[opponent]
                    if pd.notna(opponent_id) and opponent_id in grouped.groups:
                        opponent_group = grouped.get_group(opponent_id)
                        opponent_matches = opponent_group[opponent_group['date'] < row['date']]
                        for window in [3, 5, 10]:
                            for col in points_columns:
                                stats[f"{opponent}_{col}_avg_{window}"] = opponent_matches[col].tail(window).mean() if not opponent_matches.empty else np.nan
                                stats[f"{opponent}_{col}_var_{window}"] = opponent_matches[col].tail(window).var() if not opponent_matches.empty else np.nan
                        for col in points_columns:
                            ewma_series = opponent_group[col].shift(1).ewm(alpha=ewma_alpha, adjust=False).mean()
                            stats[f"{opponent}_{col}_ewma"] = ewma_series.loc[opponent_group['date'] < row['date']].iloc[-1] if not opponent_matches.empty else np.nan
                        if not opponent_matches.empty:
                            latest_row = opponent_matches.iloc[-1]
                            for attr in player_attributes:
                                stats[f"{opponent}_{attr}"] = latest_row[attr]
                
                # Combine all attributes and stats in desired order
                updated_row = {**row.to_dict(), **player_attrs, **stats}
                results.append(updated_row)
        # Create a DataFrame from the results
        averaged_data = pd.DataFrame(results)

        # Save the new dataset to a CSV file
        averaged_data.to_csv(f"../data/interim/{match_type}_final_dataset/{match_type}_final_dataset.csv", index=False)

        print(f"Averages and attributes of {match_type} computed, dataset saved.")

def perform_preprocessing(match_type_to_get):
    for match_type in match_type_to_get:
        input_path=f'../data/interim/{match_type}_final_dataset/{match_type}_final_dataset'
        df=pd.read_csv(input_path+".csv")
        all_columns=df.columns


        cols_needed=["player_id","match_id" ,"date","role","batting_points_avg_3","batting_points_avg_5","batting_points_avg_10","batting_points_ewma","bowling_points_avg_3","bowling_points_avg_5",
                     "bowling_points_avg_10","fielding_points_ewma","bowling_points_ewma","batting_points","bowling_points","fielding_points","total_fantasy_points",
                    
                    ]
        cols_needed=cols_needed + [f"opponentplayer{i}" for i in range (1,12) ] + [f"player{i}" for i in range (1,11)]
        cols_needed = cols_needed + [f"opponentplayer{i}_batting_points_avg_5" for i in range (1,12)] + [f"opponentplayer{i}_bowling_points_avg_5" for i in range (1,12)] + [[f"opponentplayer{i}_batting_points_ewma" for i in range (1,12)]] + [f"opponentplayer{i}_bowling_points_ewma" for i in range (1,12)]
        columns_to_sum1 = [f"opponentplayer{i}_batting_points_ewma" for i in range(1, 12)]
        columns_to_sum2 = [f"opponentplayer{i}_bowling_points_ewma" for i in range(1, 12)]
        columns_to_sum3 = [f"opponentplayer{i}_batting_points_avg_5" for i in range(1, 12)]
        columns_to_sum4 = [f"opponentplayer{i}_bowling_points_avg_5" for i in range(1, 12)]

        cols_needed = cols_needed + columns_to_sum1+ columns_to_sum2 +columns_to_sum3 +columns_to_sum4
        after_drop1= [item for item in all_columns if item in cols_needed]
        # for columns in after_drop1:
            # print(columns)

        df_2 = df[after_drop1]
        if set(columns_to_sum1).issubset(df_2.columns):
            df_2["OpponentTeam_batting_points_ewma"] = df_2[columns_to_sum1].sum(axis=1)
        if set(columns_to_sum2).issubset(df_2.columns):
            df_2["OpponentTeam_bowling_points_ewma"] = df_2[columns_to_sum2].sum(axis=1)
        if set(columns_to_sum3).issubset(df_2.columns):
            df_2["OpponentTeam_batting_points_avg_5"] = df_2[columns_to_sum3].sum(axis=1)
        if set(columns_to_sum4).issubset(df_2.columns):
            df_2["OpponentTeam_bowling_points_avg_5"] = df_2[columns_to_sum4].sum(axis=1)

        # Combine all columns to drop
        columns_to_drop = columns_to_sum1 + columns_to_sum2 + columns_to_sum3 + columns_to_sum4

        # Drop the columns safely
        df_3 = df_2.drop(columns=[col for col in columns_to_drop if col in df_2.columns], errors='ignore')

        # Print the transformed DataFrame
        # print(df_3.head())
        df_3.to_csv(f'{input_path}_after_drop.csv',index=False)

match_type_to_get=['ODI','Test','T20']
# split_data(match_type_to_get)
# extract_playerwise_datasets(match_type_to_get)
# create_consolidated_files(match_type_to_get)
# generate_final_dataset(match_type_to_get)
perform_preprocessing(match_type_to_get)