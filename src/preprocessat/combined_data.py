from dotenv import load_dotenv

import os
import pandas as pd

load_dotenv()

DATAFRAMES = ["players", "player_plays", "plays"]
TRACKING_PK = ["gameId", "playId", "nflId", "frameId"]
DATAFRAMES_FEATURES = {
    "players": ["nflId", "height", "weight", "position"],
    "player_plays": ["gameId", "playId", "nflId", "hadDropback", "inMotionAtBallSnap",
                     "shiftSinceLineset", "motionSinceLineset"],
    "plays": ["gameId", "playId", "quarter", "down", "yardsToGo", "possessionTeam", 
              "defensiveTeam", "yardlineNumber", "gameClock", "preSnapHomeScore", 
              "preSnapVisitorScore", "preSnapHomeTeamWinProbability",
              "preSnapVisitorTeamWinProbability", "expectedPoints",
              "playClockAtSnap", "isDropback"],
}
DEBUG = True
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

def merge_df_to_tracking(df_tracking, df_to_merge):
    """
    Merge a dataframe with the tracking dataframe on the specified column.
    """
    mergin_keys = [column for column in df_to_merge.columns if column in TRACKING_PK]
    df_tracking = df_tracking.merge(df_to_merge, how="left", on=mergin_keys)
    return df_tracking

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
    df["isOffense"] = (df["possessionTeam"] == df["club"]).astype(int)
    return df

def get_is_defense(df):
    """
    Create a new column 'isDefense' in the dataframe based on the 'defensiveTeam' column.
    """
    df["isDefense"] = (df["defensiveTeam"] == df["club"]).astype(int)
    return df

def get_is_ball(df):
    """
    Create a new column 'isBall' in the dataframe based on the 'event' column.
    """
    df["isBall"] = (df["club"] == "football").astype(int)
    return df

def get_team_one_hot_encode(df):
    df = get_is_offense(df)
    df = get_is_defense(df)
    df = get_is_ball(df)
    columns_to_drop = ["club", "possessionTeam", "defensiveTeam"]
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

def _main(df, functions):
    """
    Main function to process the tracking data and merge it with other dataframes.
    """
    for func in functions:
        df = func(df)
    return df

if __name__ == "__main__":
    MAIN_FUNCTIONS = [
        get_game_clock_in_seconds,
        is_two_minute_warning,
        get_team_one_hot_encode,
        get_players_positions
    ]
    
    print("Getting tracking_week_1-3...")
    if DEBUG:
        df = pd.read_csv("./data/debug_tracking.csv")
    else:
        df = pd.read_csv(os.getenv("tracking_week_1-3"))

    print(f"Dataframe shape: {df.shape}")
    print(f"Numero de jugades: {df['playId'].nunique()}")

    pandas_df = {var : pd.read_csv(os.getenv(var))[DATAFRAMES_FEATURES[var]] for var in DATAFRAMES if var is not None}

    # Convertim l'alçada i el pes a cm i kg
    pandas_df["players"] = convert_height_weight(pandas_df["players"])
    # Calculem l'IMC
    pandas_df["players"] = get_imc(pandas_df["players"])
    # Traiem les columnes inicials d'alçada i pes
    pandas_df["players"].drop(columns=["height", "weight"], inplace=True)

    for pd_df in pandas_df.values():
        print(f"Dataframe shape: {pd_df.shape}")
        df = merge_df_to_tracking(df, pd_df)

    _main(df)

    if DEBUG:
        df.to_csv("./data/debug_combined_data.csv", index=False)  
    else:    
        df.to_csv("./data/combined_data.csv", index=False)  
#change
