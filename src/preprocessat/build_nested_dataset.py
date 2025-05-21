import os 
import torch
import pandas as pd
import dask.dataframe as dd
from tqdm import tqdm

DATAFRAMES_FEATURES = {
    "players": ["isOffense", "isDefense", "isBall", "imc", "is_Quarterback", "is_RunningBack", "is_Receiver", 
                "is_TightEnd", "is_OffensiveLine", "is_DefensiveLine", "is_Linebacker", "is_DefensiveBack"], 
    "tracking": ["x", "y", "s", "a", "dis", "o", "dir"],
}


def build_nested_dataset_dask(
    plays_df,
    players_df,
    movements_ddf,
    player_feature_cols,
    movement_feature_cols,
    backup_every=100,
    save_path="jugadas_procesadas.pt"
):
    all_plays = []

    player_index = players_df.set_index(['gameId', 'playId', 'nflId'])

    for i, (_, play_row) in enumerate(tqdm(plays_df.iterrows(), total=len(plays_df), desc="Processant jugades")):
        game_id = play_row['gameId']
        play_id = play_row['playId']
        label = play_row['Y']

        filtered = movements_ddf[
            (movements_ddf.gameId == game_id) &
            (movements_ddf.playId == play_id)
        ]

        try:
            movements_df = filtered.compute()
            if movements_df.empty:
                continue
        except:
            continue

        players_list = []
        for nfl_id, group in movements_df.groupby("nflId"):
            try:
                player_row = player_index.loc[(game_id, play_id, nfl_id)]
                if isinstance(player_row, pd.Series):
                    player_row = player_row.to_frame().T
                player_feats = torch.tensor(
                    player_row[player_feature_cols].values[0],
                    dtype=torch.float32
                )
            except KeyError:
                continue

            move_df = group.sort_values("frameId")[movement_feature_cols]
            move_tensor = torch.tensor(move_df.values, dtype=torch.float32)

            players_list.append({
                "nflId": nfl_id,
                "features": player_feats,
                "movements": move_tensor
            })

        if players_list:
            all_plays.append({
                "gameId": game_id,
                "playId": play_id,
                "label": label,
                "players": players_list
            })

        if backup_every and (i + 1) % backup_every == 0:
            torch.save(all_plays, f"{save_path}.backup_{i+1}")

    torch.save(all_plays, save_path)
    return all_plays

if __name__ == "__main__":

    print("Carregant dades de test...")
    plays_df_test = pd.read_csv(os.path.join('data', 'final', 'plays_test.csv'))
    players_df_test = pd.read_csv(os.path.join('data', 'final', 'player_plays_test.csv'))
    tracking_df_test = dd.read_parquet(os.path.join('data', 'final', 'tracking_9_weeks_test'))
    dataset = build_nested_dataset_dask(
        plays_df=plays_df_test,
        players_df=players_df_test,
        movements_ddf=tracking_df_test,
        player_feature_cols=DATAFRAMES_FEATURES["players"],
        movement_feature_cols=DATAFRAMES_FEATURES["tracking"],
        save_path="nested_plays_test.pt"
    )


    print("Carregant dades de train...")
    plays_df_train = pd.read_csv(os.path.join('data', 'final', 'plays_train.csv'))
    players_df_train = pd.read_csv(os.path.join('data', 'final', 'player_plays_train.csv'))
    tracking_df_train = dd.read_parquet(os.path.join('data', 'final', 'tracking_9_weeks_train'))
    dataset = build_nested_dataset_dask(
        plays_df=plays_df_train,
        players_df=players_df_train,
        movements_ddf=tracking_df_train,
        player_feature_cols=DATAFRAMES_FEATURES["players"],
        movement_feature_cols=DATAFRAMES_FEATURES["tracking"],
        save_path="nested_plays_train.pt"
    )


