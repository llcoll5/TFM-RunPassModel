import os

import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

X_Y_TRAIN = pd.read_csv(os.path.join("data", "train_test", "X_Y_train.csv"))
X_Y_TEST = pd.read_csv(os.path.join("data", "train_test", "X_Y_test.csv"))

def get_train_test_from_final_dataset(df):
    """
    Get train and test data from the final dataset.
    :param df: Dataframe of the final dataset.
    :return: X_train, X_test, y_train, y_test
    """
    X_Y_train_df = df.merge(X_Y_TRAIN, on=["gameId", "playId"], how="inner")
    X_Y_test_df = df.merge(X_Y_TEST, on=["gameId", "playId"], how="inner")
    return X_Y_train_df, X_Y_test_df


if __name__ == "__main__":
    FILE_NAME = os.getenv("plays")
    plays = pd.read_csv(FILE_NAME)[["gameId", "playId", "isDropback"]]
    X = plays.drop(columns=["isDropback"])
    y = plays["isDropback"].astype("int")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    path = os.path.join("data", "train_test")
    X_Y_train = pd.concat([X_train, y_train], axis=1)
    X_Y_train.rename(columns={"isDropback": "Y"}, inplace=True)
    X_Y_test = pd.concat([X_test, y_test], axis=1)
    X_Y_test.rename(columns={"isDropback": "Y"}, inplace=True)
    X_Y_train.to_csv(os.path.join(path, "X_Y_train.csv"), index=False)
    X_Y_test.to_csv(os.path.join(path, "X_Y_test.csv"), index=False)
    print(f"S'ha creat els fitxers d'entrenament (amb {len(X_Y_train)} exemples) i de test (amb {len(X_Y_test)} exemples) a la carpeta {path}.")
    print(f"X_Y_train: {X_Y_train.shape}")
    print(f"X_Y_test: {X_Y_test.shape}")

    plt.figure(figsize=(12, 6))
    plt.suptitle("Passada vs Carrera")
    plt.subplot(1, 2, 1)
    plt.title("X_Y_train")
    plt.bar(["Passada", "Carrera"], X_Y_train["Y"].value_counts(), color='red', alpha=0.5)
    plt.subplot(1, 2, 2)
    plt.title("X_Y_test")
    plt.bar(["Passada", "Carrera"], X_Y_test["Y"].value_counts(), color='blue', alpha=0.5)
    plt.show()