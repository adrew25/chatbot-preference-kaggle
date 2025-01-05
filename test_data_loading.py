from src.data_preprocessing.load_data import load_data
from src.data_preprocessing.preprocess_data import preprocess_data
from src.data_preprocessing.tokenize import tokenize_data_in_batches


def test_data_loading():
    # Load data
    df = load_data("data/raw/train.parquet")

    # Preprocess data
    df = preprocess_data(df)

    print(df.head())
    print(df.columns)


def test_tokenization():
    df = load_data("data/raw/train.parquet")

    df = preprocess_data(df)

    tokenized_data = tokenize_data_in_batches(df, batch_size=256)

    print(tokenized_data.keys())
    print("Tokenization complete. Shape:", tokenized_data["input_ids"].shape)


if __name__ == "__main__":
    # test_data_loading()
    test_tokenization()
