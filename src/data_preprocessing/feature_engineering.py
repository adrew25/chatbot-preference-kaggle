import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Add average word length features
def calculate_avg_word_length(response):
    words = response.split()
    if words:
        total_word_length = sum(len(word) for word in words)
        return total_word_length / len(words)
    else:
        return 0


def add_textual_features(df):
    # Text len features
    df["len_response_a"] = df["response_a"].apply(len)
    df["len_response_b"] = df["response_b"].apply(len)

    # Word count
    df["word_count_a"] = df["response_a"].apply(lambda x: len(x.split()))
    df["word_count_b"] = df["response_b"].apply(lambda x: len(x.split()))

    # Avg len word features
    df["avg_word_len_a"] = df["response_a"].apply(calculate_avg_word_length)
    df["avg_word_len_b"] = df["response_b"].apply(calculate_avg_word_length)

    return df


def add_model_features(df):
    # Encode model identities as categorical features
    le = LabelEncoder()

    # Apply encoding separately for 'model_a' and 'model_b'
    df["model_a_encoded"] = le.fit_transform(df["model_a"])
    df["model_b_encoded"] = le.transform(df["model_b"])

    return df


def add_language_bias_features(df):
    # One-hot encoding for language feature
    df = pd.get_dummies(df, columns=["language"], drop_first=True)

    return df


def add_interaction_features(df):
    # Difference in response lengths
    df["length_diff"] = df["len_response_a"] - df["len_response_b"]

    return df


def feature_engineering_pipeline(df):
    # Add features one by one
    df = add_textual_features(df)
    df = add_model_features(df)
    df = add_language_bias_features(df)
    df = add_interaction_features(df)

    return df


def sequence_feature_engineering_pipeline():
    print("🚀 Starting feature engineering pipeline... \n")
    raw_data = pd.read_parquet("data/raw/train.parquet")
    print(raw_data.head())

    # Ensure required columns are present
    if not {"prompt", "response_a", "response_b", "winner"}.issubset(raw_data.columns):
        raise ValueError("❌ Required columns are missing from raw data!")

    df = feature_engineering_pipeline(raw_data)

    # Correctly encode the winner (0 for response_a, 1 for response_b)
    df["winner"] = df["winner"].map({"model_a": 0, "model_b": 1})

    # Save feature-engineered data to disk
    df.to_parquet("data/processed/feature_engineered_data.parquet")
    df["winner"].to_pickle("data/processed/labels.pkl")

    print("\n✅ Feature engineering complete! Saved with corrected labels.")
