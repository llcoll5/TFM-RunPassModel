from dotenv import load_dotenv

import os
import pandas as pd
import numpy as np

from train_test_creator import get_train_test_from_final_dataset 

load_dotenv()

DATAFRAMES = ["players", "player_plays", "plays"]
DATAFRAMES_FEATURES = {
    "players": ["nflId", "height", "weight", "position"],
    "player_plays": ["gameId", "playId", "nflId", "teamAbbr"],
    "plays": ["gameId", "playId", "quarter", "down", "yardsToGo", "possessionTeam", 
              "defensiveTeam", "yardlineNumber", "gameClock", "preSnapHomeTeamWinProbability",
              "preSnapVisitorTeamWinProbability", "expectedPoints",
              "playClockAtSnap"]
}
PLAYERS_POSITIONS = {
    'Quarterback': ['QB'],
    'RunningBack': ['RB', 'FB'],
    'Receiver': ['WR'],
    'TightEnd': ['TE'],
    'OffensiveLine': ['G', 'C', 'T'],
    'DefensiveLine': ['DE', 'DT', 'NT'],
    'Linebacker': ['ILB', 'OLB', 'MLB', 'LB'],
    'DefensiveBack': ['CB', 'DB', 'SS', 'FS'],
}

def convert_feet_inches_to_cm(feet_inches):
    """
    Convert a string in the format "X-Y" (feet-inches) to centimeters.
    """
    feet, inches = map(int, feet_inches.split("-"))
    return (feet * 12 + inches) * 2.54

def convert_pounds_to_kg(pounds):
    """
    Convert pounds to kilograms.
    """
    return pounds * 0.453592

def convert_height_weight(df):
    """
    Convert height and weight columns in the dataframe to centimeters and kilograms.
    """
    df["height"] = df["height"].apply(convert_feet_inches_to_cm)
    df["weight"] = df["weight"].apply(convert_pounds_to_kg)
    return df

def get_imc(df):
    """
    Calculate the IMC (Body Mass Index) for each player in the dataframe.
    """
    df["imc"] = df["weight"] / ((df["height"] / 100) ** 2)
    return df

def get_is_offense(df):
    """
    Create a new column 'isOffense' in the dataframe based on the 'possessionTeam' column.
    """
    df["isOffense"] = (df["possessionTeam"] == df["teamAbbr"]).astype(int)
    return df

def get_is_defense(df):
    """
    Create a new column 'isDefense' in the dataframe based on the 'defensiveTeam' column.
    """
    df["isDefense"] = (df["defensiveTeam"] == df["teamAbbr"]).astype(int)
    return df

def get_is_ball(df):
    """
    Create a new column 'isBall' in the dataframe based on the 'event' column.
    """
    df["isBall"] = (df["teamAbbr"] == "football").astype(int)
    return df

def get_team_one_hot_encode(player_plays_df, plays_df):
    df = pd.merge(player_plays_df, plays_df, on=["gameId", "playId"], how="left")
    df = get_is_offense(df)
    df = get_is_defense(df)
    df = get_is_ball(df)
    columns_to_drop = ["teamAbbr"]
    return df.drop(columns=columns_to_drop)

def get_players_positions(df):
    """
    Create a new column 'position' in the dataframe based on the 'position' column.
    """
    for pos in PLAYERS_POSITIONS.keys():
        df["is_" + pos] = df["position"].isin(PLAYERS_POSITIONS[pos]).astype(int)
    df.drop(columns=["position"], inplace=True)
    return df

def get_game_clock_in_seconds(df):
    """
    Convert the 'gameClock' column to seconds.
    """
    df["gameClockInSecs"] = df.apply(lambda x: 15 * int(x.quarter) * 60 - (int(x.gameClock.split(":")[0]) * 60 + int(x.gameClock.split(":")[1])), axis=1)
    df["halfClockInSecs"] = df.apply(lambda x: x.gameClockInSecs if x.gameClockInSecs <= 1800 else x.gameClockInSecs - 1800 , axis=1)
    return df.drop(columns=["gameClock"])

def is_two_minute_warning(df):
    """
    Create a new column 'isTwoMinuteWarning' in the dataframe based on the 'gameClock' column.
    """
    df["isTwoMinuteWarning"] = ((df["halfClockInSecs"] >= 1680) & (df["halfClockInSecs"] < 1800)).astype(int)
    return df

def get_games_df():
    """
    Create a new dataframe with the games data.
    """
    games_df = pd.read_csv(os.getenv("games"))[["gameId","homeTeamAbbr","visitorTeamAbbr"]]
    return games_df

def get_home_away_team(df):
    """
    Create a new column 'homeTeam' in the dataframe based on the 'homeTeamAbbr' and 'visitorTeamAbbr' columns.
    """
    games_df = get_games_df()
    df = df.merge(games_df, on=["gameId"], how="left")
    df["homeTeam"] = df.apply(lambda x: 1 if x.possessionTeam == x.homeTeamAbbr else 0, axis=1)
    return df.drop(columns=["homeTeamAbbr", "visitorTeamAbbr"])

def get_possession_team_win_probability(df):
    """
    Create a new column 'preSnapPosTeamWP' in the dataframe based on the 'homeTeam' column.
    """
    print(f"head: {df.head()}")
    df["preSnapPosTeamWP"] = np.where(df["homeTeam"] == 1, df["preSnapHomeTeamWinProbability"], df["preSnapVisitorTeamWinProbability"])
    return df.drop(columns=["preSnapHomeTeamWinProbability", "preSnapVisitorTeamWinProbability", "homeTeam"])

def apply_players_functions(df):
    """
    Apply a list of functions to the players dataframe.
    """
    functions = [convert_height_weight, get_imc, get_players_positions]
    df = apply_functions_to_df(df, functions)
    return df.drop(columns=["height", "weight"])

def apply_plays_functions(df):
    """
    Apply a list of functions to the plays dataframe.
    """
    functions = [get_game_clock_in_seconds, is_two_minute_warning, get_home_away_team, get_possession_team_win_probability]
    df = apply_functions_to_df(df, functions)
    return df.drop(columns=["possessionTeam", "defensiveTeam"])

def apply_player_plays_functions(df, plays_df):
    """
    Apply a list of functions to the player_plays dataframe.
    """
    df = get_team_one_hot_encode(df, plays_df)
    return df.drop(columns=[col for col in list(plays_df.columns) if col not in ["gameId", "playId"]])

def apply_functions_to_df(df, functions):
    """
    Apply list of functions to the dataframe.
    """
    for func in functions:
        df = func(df)
    return df

def _main(players, player_plays, plays):
    """
    Main function to process the data.
    """
    players = apply_players_functions(players)
    player_plays = apply_player_plays_functions(player_plays, plays)
    player_plays = pd.merge(player_plays, players, on=["nflId"], how="left")
    plays = apply_plays_functions(plays)
    return player_plays, plays

if __name__ == "__main__":
    MAIN_FUNCTIONS = [
        get_game_clock_in_seconds,
        is_two_minute_warning,
        get_team_one_hot_encode,
        get_players_positions
    ]

    pandas_df = {var : pd.read_csv(os.getenv(var))[DATAFRAMES_FEATURES[var]] for var in DATAFRAMES if var is not None}


    new_dfs = _main(*pandas_df.values())
    df_names = ["player_plays", "plays"]
    for i, df in enumerate(new_dfs):
        df.name = df_names[i]
        df.to_csv(f"./data/final/{df.name}.csv", index=False)
        df_train, df_test = get_train_test_from_final_dataset(df)
        df_train.to_csv(f"./data/final/{df.name}_train.csv", index=False)
        df_test.to_csv(f"./data/final/{df.name}_test.csv", index=False)

