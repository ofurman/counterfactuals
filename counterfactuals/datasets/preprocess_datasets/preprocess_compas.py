import numpy as np
import pandas as pd

from argparse import ArgumentParser


def preprocess_compas(raw_data: pd.DataFrame) -> pd.DataFrame:

    target_column = "class"

    raw_data["days_b_screening_arrest"] = np.abs(
        raw_data["days_b_screening_arrest"]
    )
    raw_data["c_jail_out"] = pd.to_datetime(raw_data["c_jail_out"])
    raw_data["c_jail_in"] = pd.to_datetime(raw_data["c_jail_in"])
    raw_data["length_of_stay"] = np.abs(
        (raw_data["c_jail_out"] - raw_data["c_jail_in"]).dt.days
    )
    raw_data["length_of_stay"].fillna(
        raw_data["length_of_stay"].value_counts().index[0], inplace=True
    )
    raw_data["days_b_screening_arrest"].fillna(
        raw_data["days_b_screening_arrest"].value_counts().index[0], inplace=True
    )
    raw_data["length_of_stay"] = raw_data["length_of_stay"].astype(int)
    raw_data["days_b_screening_arrest"] = raw_data[
        "days_b_screening_arrest"
    ].astype(int)
    # raw_data = raw_data[raw_data["score_text"] != "Medium"]
    # raw_data["class"] = pd.get_dummies(raw_data["score_text"])["High"].astype(int)
    raw_data["class"] = (
        raw_data["score_text"].map({"Low": 0, "Medium": 1, "High": 2}).astype(int)
    )
    raw_data.drop(["c_jail_in", "c_jail_out", "score_text"], axis=1, inplace=True)
    return raw_data


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--input_path", type=str, default="data/compas/compas_two_years.csv")
    args.add_argument("--output_path", type=str, default="data/compas/compas.csv")
    args = args.parse_args()
    raw_data = pd.read_csv(args.input_path)
    df = preprocess_compas(raw_data)
    df.to_csv(args.output_path, index=False)
