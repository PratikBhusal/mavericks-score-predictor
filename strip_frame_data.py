import glob
import math
import os
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb


def get_players(frames_dataframe):

    # Return a list of unique players for a given team (home or away)
    def get_unique_players(frames_dataframe, places):
        players = []

        # Looks through each distance column (hp1, hp2, ap4, etc...), and gets a list
        #   of unique players in that column.
        # Looking through all columns for a team gives us all players on the team.
        for distance in places:
            for i in frames_dataframe[distance].unique().tolist():
                if not (math.isnan(i)):
                    players.append(i)

        return players

    # Lists of player ids for each home and away teams
    home_player_ids = get_unique_players(
        frames_dataframe, ["hp1", "hp2", "hp3", "hp4", "hp5"]
    )
    away_player_ids = get_unique_players(
        frames_dataframe, ["ap1", "ap2", "ap3", "ap4", "ap5"]
    )

    # Initialize the player dictionary.
    # Main key is player id. Holds all information on each player.
    players = {player_id: dict() for player_id in (home_player_ids + away_player_ids)}

    for player_id, _ in players.items():
        if player_id in home_player_ids:
            players[player_id]["team"] = "home"
        else:
            players[player_id]["team"] = "away"
        # players[player_id]["player_id"] = player_id

    return players


# x = pd.read_csv('')


def main():

    # current working directory
    path = Path.cwd()

    # names of files we want to read from
    file_names = list(path.glob("*.csv"))

    # output path
    output_path = str(path) + "/output/"

    for file_name in file_names:
        new_file_name_arr = str(file_name).split("/")
        new_file_name_arr.insert(-1, "output")
        new_file_name = "/".join(new_file_name_arr)
        if not (glob.glob(new_file_name)):
            # print(new_file_name)
            frame = pd.read_csv(file_name)
            players = get_players(frame)
            players_df = pd.DataFrame(players).transpose()
            players_df.to_csv(
                new_file_name, sep=",", encoding="utf-8", index_label="player_id"
            )


if __name__ == "__main__":
    main()


# frame = pd.read.csv("2016-11-21-ORL-MIL.csv")
# trimmed_frame = frame[['']]
