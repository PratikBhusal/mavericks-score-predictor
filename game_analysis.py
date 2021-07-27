# imports
import io
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Simple_markings folder. Holds the "events", e.g. 3PM, 2PM, PASS, FOUL, etc...


# Returns a dictionary containing all the players, with a key indicating whether
#    they are on the home or away team.
# input: The game_info object we read from one of the csv files
def get_players(frames_dataframe, home, away):

    """
    # Return a list of unique players for a given team (home or away)
    def get_unique_players(frames_dataframe, places):
      players = []

      # Looks through each distance column (hp1, hp2, ap4, etc...), and gets a list
      #   of unique players in that column.
      # Looking through all columns for a team gives us all players on the team.
      for distance in places:
        for i in frames_dataframe[distance].unique().tolist():
          if not(math.isnan(i)):
            players.append(i)

      return players
    """
    # print(home)
    # print(away)
    players = dict()
    for p_id in frames_dataframe.player_id:
        players[p_id] = {
            "team": home
            if frames_dataframe.loc[frames_dataframe.player_id == p_id, "team"].iloc[0]
            == "home"
            else away
        }
    return players
    """
  home_player_ids = get_unique_players(frames_dataframe, ['hp1', 'hp2', 'hp3', 'hp4', 'hp5'])
  away_player_ids = get_unique_players(frames_dataframe, ['ap1', 'ap2', 'ap3', 'ap4', 'ap5'])

  # Initialize the player dictionary.
  # Main key is player id. Holds all information on each player.
  players = {player_id: dict() for player_id in (home_player_ids + away_player_ids)}

  for player_id, data in players.items():
    if player_id in home_player_ids:
      data["team"] = "home"
    else:
      data["team"] = "away"

  return players
  """


def get_player_info(players, players_dataframe, attribute):
    for player_id, data in players.items():
        players[player_id][attribute] = players_dataframe.loc[
            players_dataframe.ids_id == player_id, attribute
        ].iloc[0]


def get_event_info(players, game_markings_dataframe, event):
    for player_id, data in players.items():
        players[player_id][event] = len(
            game_markings_dataframe.loc[
                (game_markings_dataframe.player_id == player_id)
                & (game_markings_dataframe.event == event)
            ]
        )


# Uses the game/player information gathered above to get the number of points made
#   by each team during actual gameplay. Since there is no information about free
#   throws made from foul plays, we cannot actually determine the final score or
#   which team won.
def get_scoreboard(game_df, home, away):
    home_score = 0
    away_score = 0

    for i in range(len(game_df["2PM"])):
        if game_df.iloc[i]["team"] == home:
            home_score += game_df["2PM"].iloc[i] * 2
        if game_df.iloc[i]["team"] == away:
            away_score += game_df["2PM"].iloc[i] * 2

    for i in range(len(game_df["3PM"])):
        if game_df.iloc[i]["team"] == home:
            home_score += game_df["3PM"].iloc[i] * 3
        if game_df.iloc[i]["team"] == away:
            away_score += game_df["3PM"].iloc[i] * 3

    return home_score, away_score


def get_game_names(game_file_names):
    game_names = []
    for game_file_name in game_file_names:
        game_file_name_arr = str(game_file_name).split("/")
        game_names.append(game_file_name_arr[-1])
    return game_names


def squash(game_dataframe, home, away, home_score, away_score):
    game = dict()
    game["home"] = home
    game["home_points"] = home_score
    game["away"] = away
    game["away_points"] = away_score

    # dataframe_dict = game_dataframe.to_dict("index"))
    team_stats = {key: dict() for key in ["home", "away"]}
    for player_id, player_stats in game_dataframe.to_dict("index").items():
        if player_stats["team"] == home:
            if player_stats["pos_name"] not in team_stats["home"]:
                team_stats["home"][player_stats["pos_name"]] = dict()
                team_stats["home"][player_stats["pos_name"]]["count"] = 0
                team_stats["home"][player_stats["pos_name"]]["2PM"] = 0
                team_stats["home"][player_stats["pos_name"]]["2PX"] = 0
                team_stats["home"][player_stats["pos_name"]]["3PM"] = 0
                team_stats["home"][player_stats["pos_name"]]["3PX"] = 0
                team_stats["home"][player_stats["pos_name"]]["PASS"] = 0
                team_stats["home"][player_stats["pos_name"]]["POSS"] = 0
                team_stats["home"][player_stats["pos_name"]]["TO"] = 0
            team_stats["home"][player_stats["pos_name"]]["count"] += 1
            team_stats["home"][player_stats["pos_name"]]["2PM"] += player_stats["2PM"]
            team_stats["home"][player_stats["pos_name"]]["2PX"] += player_stats["2PX"]
            team_stats["home"][player_stats["pos_name"]]["3PM"] += player_stats["3PM"]
            team_stats["home"][player_stats["pos_name"]]["3PX"] += player_stats["3PX"]
            team_stats["home"][player_stats["pos_name"]]["PASS"] += player_stats["PASS"]
            team_stats["home"][player_stats["pos_name"]]["POSS"] += player_stats["POSS"]
            team_stats["home"][player_stats["pos_name"]]["TO"] += player_stats["TO"]
        else:
            if player_stats["pos_name"] not in team_stats["away"]:
                team_stats["away"][player_stats["pos_name"]] = dict()
                team_stats["away"][player_stats["pos_name"]]["count"] = 0
                team_stats["away"][player_stats["pos_name"]]["2PM"] = 0
                team_stats["away"][player_stats["pos_name"]]["2PX"] = 0
                team_stats["away"][player_stats["pos_name"]]["3PM"] = 0
                team_stats["away"][player_stats["pos_name"]]["3PX"] = 0
                team_stats["away"][player_stats["pos_name"]]["PASS"] = 0
                team_stats["away"][player_stats["pos_name"]]["POSS"] = 0
                team_stats["away"][player_stats["pos_name"]]["TO"] = 0
            team_stats["away"][player_stats["pos_name"]]["count"] += 1
            team_stats["away"][player_stats["pos_name"]]["2PM"] += player_stats["2PM"]
            team_stats["away"][player_stats["pos_name"]]["2PX"] += player_stats["2PX"]
            team_stats["away"][player_stats["pos_name"]]["3PM"] += player_stats["3PM"]
            team_stats["away"][player_stats["pos_name"]]["3PX"] += player_stats["3PX"]
            team_stats["away"][player_stats["pos_name"]]["PASS"] += player_stats["PASS"]
            team_stats["away"][player_stats["pos_name"]]["POSS"] += player_stats["POSS"]
            team_stats["away"][player_stats["pos_name"]]["TO"] += player_stats["TO"]

    for team, team_info in team_stats.items():
        for position, position_stats in team_info.items():
            position_stats["2PM"] /= position_stats["count"]
            position_stats["2PX"] /= position_stats["count"]
            position_stats["3PM"] /= position_stats["count"]
            position_stats["3PX"] /= position_stats["count"]
            position_stats["PASS"] /= position_stats["count"]
            position_stats["POSS"] /= position_stats["count"]
            position_stats["TO"] /= position_stats["count"]
            del position_stats["count"]

    for team, team_info in team_stats.items():
        for position, position_stats in team_info.items():
            for stat in list(position_stats):
                game[team + "_" + position + "_" + stat] = position_stats[stat]
    return game


def create_dataset():
    players_dataframe = pd.read_csv("meta/players.csv")
    games_dataframe = pd.read_csv("meta/games.csv")
    teams_dataframe = pd.read_csv("meta/teams.csv")

    file_names = get_game_names(list((Path.cwd() / "simple-markings").glob("*.csv")))

    game_info = dict()
    for file_name in file_names:
        if os.path.isfile("simple-frames/" + file_name):
            file_name_array = file_name.split("-")
            away, home = file_name_array[-2], file_name_array[-1][:3]

            frames_dataframe = pd.read_csv("simple-frames/" + file_name)
            game_markings_dataframe = pd.read_csv("simple-markings/" + file_name)
            players = get_players(frames_dataframe, home, away)
            get_player_info(players, players_dataframe, "pos_name")
            get_event_info(players, game_markings_dataframe, "PASS")
            get_event_info(players, game_markings_dataframe, "2PM")
            get_event_info(players, game_markings_dataframe, "2PX")
            get_event_info(players, game_markings_dataframe, "3PM")
            get_event_info(players, game_markings_dataframe, "3PX")
            get_event_info(players, game_markings_dataframe, "POSS")
            get_event_info(players, game_markings_dataframe, "TO")
            game_dataframe = pd.DataFrame(players).transpose()
            home_score, away_score = get_scoreboard(game_dataframe, home, away)
            game_info[file_name] = squash(
                game_dataframe, home, away, home_score, away_score
            )

    game_info_dataframe = pd.DataFrame(game_info).transpose()
    print(game_info_dataframe)
    game_info_dataframe.to_csv(
        "all_game_info.csv", sep=",", encoding="utf-8", index_label="game_name"
    )


def learn():
    def calculate_loss(submission_dict):
        for game_index, game_stats in submission_dict.items():
            total_predicted_points = (
                game_stats["home_points_predicted"]
                + game_stats["away_points_predicted"]
            )
            total_actual_points = (
                game_stats["home_points_actual"] - game_stats["away_points_actual"]
            )
            # game_stats["percent_accuracy"] = (total_actual_points - loss) / total_actual_points * 100
            game_stats["loss"] = abs(total_predicted_points - total_actual_points)

    atts_to_exclude = ["home", "away", "game_name"]

    full_df = pd.read_csv("all_game_info.csv")  # grab the entire csv
    for (
        col
    ) in (
        full_df.columns.values
    ):  # we exclude all of the point making ones because that is too easy
        if "PM" in col:
            atts_to_exclude.append(col)

    train_df = full_df.sample(
        frac=0.8, random_state=200
    )  # split into training and testing

    test_df = full_df.drop(train_df.index)

    x_test = test_df.drop(["away_points", "home_points"] + atts_to_exclude, axis=1)
    x_train = train_df.drop(["away_points", "home_points"] + atts_to_exclude, axis=1)
    y_train_h = train_df["home_points"]
    y_train_a = train_df["away_points"]

    classifier_h = xgb.XGBClassifier(n_estimators=300, n_jobs=4, silent=1, eta=0.1)
    classifier_a = xgb.XGBClassifier(n_estimators=300, n_jobs=4, silent=1, eta=0.1)

    classifier_h.fit(x_train, y_train_h)
    classifier_a.fit(x_train, y_train_a)
    pred_h = classifier_h.predict(x_test)
    pred_a = classifier_a.predict(x_test)

    submission = pd.DataFrame(
        {
            "game_name": test_df.game_name,
            "home_points_predicted": pred_h,
            "away_points_predicted": pred_a,
            "home_points_actual": test_df.home_points,
            "away_points_actual": test_df.away_points,
        }
    )
    submission_dict = submission.to_dict("index")
    # print(submission_dict)
    calculate_loss(submission_dict)
    submission = pd.DataFrame(submission_dict).transpose()
    submission.to_csv(
        "answers.csv",
        index=False,
        columns=[
            "game_name",
            "home_points_predicted",
            "home_points_actual",
            "away_points_actual",
            "loss",
        ],
    )


def main():
    if not (os.path.isfile("all_game_info.csv")):
        create_dataset()
    learn()


if __name__ == "__main__":
    main()