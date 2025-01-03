import csv
import time
import json
import random

def process_csv():
    players = []

    # Open and read the CSV file
    with open("api/test.csv", "r") as file:
        reader = csv.DictReader(file)  # Use DictReader to read rows as dictionaries
        for row in reader:
            players.append({"id": row["Player Name"], "explanation": f"{row['Player Name']} is a player in {row['Squad']}."})

    # Simulate a delay
    time.sleep(10)

    # Randomly select 11 players (or fewer if the file has less than 11 entries)
    selected_players = random.sample(players, min(len(players), 11))

    return selected_players

if __name__ == "__main__":
    # Process the CSV and get the result
    result = process_csv()

    # Output the result in JSON format
    print(json.dumps(result))

