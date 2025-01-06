import pandas as pd
from src.data_preprocessing.feature_engineering import (
    sequence_feature_engineering_pipeline,
)
from src.data_preprocessing.tokenize import tokenize_data_in_batches


def feature_engineering_tokenization_pipeline():
    sequence_feature_engineering_pipeline()
    df = pd.read_parquet("data/processed/feature_engineered_data.parquet")
    tokenize_data_in_batches(df)
    return True


if __name__ == "__main__":
    feature_engineering_tokenization_pipeline()
