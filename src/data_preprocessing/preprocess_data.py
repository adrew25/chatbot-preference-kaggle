import pandas as pd


def preprocess_data(df):
    # Strip leading and trailing whitespaces
    df["prompt"] = df["prompt"].str.strip()
    df["response_a"] = df["response_a"].str.strip()
    df["response_b"] = df["response_b"].str.strip()

    # Convert text to lowercase (optional step)
    df["prompt"] = df["prompt"].str.lower()
    df["response_a"] = df["response_a"].str.lower()
    df["response_b"] = df["response_b"].str.lower()

    return df
