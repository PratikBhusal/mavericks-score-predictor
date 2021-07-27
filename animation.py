
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import math
import sys  # for reading direction
from time import sleep


# Read everything from the file into a Panda dataframe
df = pd.read_csv('frames/2013-10-29-CHI-MIA.csv')
# print(len(df))

# Set up some matplotlib stuff
fig = plt.figure()
# fig.set_figheight(30)
# fig.set_figwidth(30)
ax = fig.add_subplot(111)

home_places = ['hp1', 'hp2', 'hp3', 'hp4', 'hp5']
home_x_dist = ['hp1_x', 'hp2_x', 'hp3_x', 'hp4_x', 'hp5_x']
home_y_dist = ['hp1_y', 'hp2_y', 'hp3_y', 'hp4_y', 'hp5_y']

away_places = ['ap1', 'ap2', 'ap3', 'ap4', 'ap5']
away_x_dist = ['ap1_x', 'ap2_x', 'ap3_x', 'ap4_x', 'ap5_x']
away_y_dist = ['ap1_y', 'ap2_y', 'ap3_y', 'ap4_y', 'ap5_y']

def get_closest_player(file_idx, places, x_dists, y_dists):
    # Get this particular row (file_idx) of the data frame
    frame_data = df.iloc[file_idx]
    #print(frame_data)

    closest = 100000  # The basketball court is going to be smaller than this, so we should always update it
    closest_player = ''

    # get each player's distance from the ball at this frame
    for idx, place in enumerate(places):
        x_dist = frame_data['ball_x'] - frame_data[x_dists[idx]]
        y_dist = frame_data['ball_y'] - frame_data[y_dists[idx]]
        distance = math.sqrt( pow(x_dist, 2) + pow(y_dist, 2) )

        if distance < closest:
            closest = distance
            closest_player = places[idx]
            print(closest_player)

    return closest_player

def animate(i, frame_number, players):

    # Interval is the amount of time we show at once.
    # Seconds * frames per second.
    interval = 1 * 24

    frame_number[0] += 6

    # Bound frame_number
    if frame_number[0] > len(df) - interval:
        frame_number[0] -= interval
    if frame_number[0] < 0:
        frame_number[0] = 0

    #print(frame_number[0])

    # Get the particular chunk of data we want to draw from.
    chunk = df[frame_number[0]: frame_number[0] + interval]

    # Get the closest player to the ball on the home team
    home_player_place = get_closest_player(frame_number, home_places, home_x_dist, home_y_dist)
    home_player_place_x = home_player_place + "_x"
    home_player_place_y = home_player_place + "_y"

    # Get the closest player to the ball on the away team
    away_player_place = get_closest_player(frame_number, away_places, away_x_dist, away_y_dist)
    away_player_place_x = away_player_place + "_x"
    away_player_place_y = away_player_place + "_y"

    # Update buffers for each of the players on the field.


    ax.clear()
    ax.plot(chunk["ball_x"], chunk["ball_y"], label="ball")
    ax.plot(chunk[home_player_place_x],  chunk[home_player_place_y],  label="home player")
    ax.plot(chunk[away_player_place_x],  chunk[away_player_place_y],  label="away player")
    ax.plot([-5], [-5])  # lower-left bound point
    ax.plot([99], [55])  # upper-right bound point

    # Nice that it automatically moves itself out of the way.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)


# Returns a dictionary containing all the players, with a key indicating whether
#    they are on the home or away team.
# input: The game_info object we read from one of the csv files
def get_players():
    # Return a list of unique players for a given team (home or away)
    def get_unique_players(df, places):
        players = []

        # Looks through each distance column (hp1, hp2, ap4, etc...), and gets a list
        #   of unique players in that column.
        # Looking through all columns for a team gives us all players on the team.
        for distance in places:
            for i in df[distance].unique().tolist():
                if not (math.isnan(i)):
                    players.append(i)

        return players

    home_player_ids = get_unique_players(df, home_places)
    away_player_ids = get_unique_players(df, away_places)

    # Initialize the player dictionary.
    # Main key is player id. Holds all information on each player.
    players = {player_id: dict() for player_id in (home_player_ids + away_player_ids)}

    for player_id, data in players.items():
        if player_id in home_player_ids:
            data["team"] = "home"
        else:
            data["team"] = "away"

    return players


def main():

    # frame_number that we are currently drawing
    frame_number = [0]

    players = get_players()
    print(players)

    # Create a matplotlib animation thing.
    #   the figure we want to draw,
    #   the animation update function,
    #   updated every x milliseconds.
    ani = anim.FuncAnimation(fig, animate, fargs=(frame_number,players,), interval=100)
    plt.show()


if __name__ == '__main__':
    #print("Hello, World!")
    main()

    # # Hrishi
    # x = pd.read_csv('frames\\2013-10-29-CHI-MIA.csv')
    # sns.scatterplot(x=x['ball_x'], y=x['ball_y'], size=x['ball_z'], sizes=(2, 20), hue=x['ball_z'])
    # plt.show()
