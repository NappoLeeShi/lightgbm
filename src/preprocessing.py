import pandas as pd
import numpy as np


def load_data(path):

    df = pd.read_csv(path)

    df = df.drop(
        "customerID",
        axis=1
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = (
        df["TotalCharges"]
        .fillna(
            df["TotalCharges"].median()
        )
    )

    df["Churn"] = df["Churn"].map({
        "No":0,
        "Yes":1
    })

    df = pd.get_dummies(
        df,
        drop_first=True
    )

    return df


def train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
):

    np.random.seed(
        random_state
    )

    idx = np.random.permutation(
        len(X)
    )

    test_n = int(
        len(X)*test_size
    )

    test_idx = idx[:test_n]

    train_idx = idx[test_n:]

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx]
    )