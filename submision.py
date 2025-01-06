import os
import torch
import pandas as pd
import torch.nn as nn
from transformers import DistilBertModel
from tqdm import tqdm
from transformers import AutoTokenizer

# Define the local path to the tokenizer
LOCAL_TOKENIZER_PATH = "path/to/local/tokenizer"

os.makedirs("data/predictions", exist_ok=True)


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


def add_interaction_features(df):
    # Difference in response lengths
    df["length_diff"] = df["len_response_a"] - df["len_response_b"]

    return df


def feature_engineering_pipeline(df):
    # Add features one by one
    df = add_textual_features(df)
    df = add_interaction_features(df)

    return df


def sequence_feature_engineering_pipeline():
    print("🚀 Starting feature engineering pipeline... \n")
    raw_data = pd.read_parquet("data/raw/test.parquet")
    print(raw_data.head())

    # Ensure required columns are present
    if not {"prompt", "response_a", "response_b"}.issubset(raw_data.columns):
        raise ValueError("❌ Required columns are missing from raw data!")

    df = feature_engineering_pipeline(raw_data)

    # Save feature-engineered data to disk
    df.to_parquet("data/predictions/test_feature_engineered_data.parquet")

    print("\n✅ Feature engineering complete")


def tokenize_data_in_batches(df, batch_size=128, cache_path="data/predictions/"):
    # Load the tokenizer from the local path
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_PATH)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    inputs = {
        "input_ids": [],
        "attention_mask": [],
    }

    # Ensure winner labels are handled correctly
    labels = df["winner"].tolist() if "winner" in df.columns else None

    # Tokenize in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Tokenizing Data"):
        batch_df = df.iloc[i : i + batch_size]

        batch_prompts = batch_df["prompt"].tolist()
        batch_resp_a = batch_df["response_a"].tolist()
        batch_resp_b = batch_df["response_b"].tolist()

        with torch.no_grad():
            batch_inputs = tokenizer(
                batch_prompts,
                batch_resp_a,
                batch_resp_b,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=256,
            )

        # Move tensors to CPU and store to avoid memory issues
        inputs["input_ids"].append(batch_inputs["input_ids"].cpu())
        inputs["attention_mask"].append(batch_inputs["attention_mask"].cpu())

    # Concatenate all batches
    inputs["input_ids"] = torch.cat(inputs["input_ids"], dim=0)
    inputs["attention_mask"] = torch.cat(inputs["attention_mask"], dim=0)

    torch.save(inputs, cache_path + "test_tokenized_data.pt")

    print(f"✅ Tokenization complete! Saved to {cache_path}")
    return inputs, labels


class DistilBERTClassifier(nn.Module):
    def __init__(self):
        super(DistilBERTClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(
            "distilbert-base-multilingual-cased"
        )
        self.classifier = nn.Linear(
            self.distilbert.config.hidden_size, 2
        )  # Assuming binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        return self.classifier(pooled_output)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and tokenizer
model_path = "src/models/saved_pths/DistilBERTClassifier_51.41.pth"
model = DistilBERTClassifier()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)


# model_a will be the 0 class, model_b will be the 1 class


# inference on test data
def inference_on_test_data():
    test_data = pd.read_parquet("data/predictions/test_feature_engineered_data.parquet")
    test_inputs, _ = tokenize_data_in_batches(test_data, cache_path="data/predictions/")

    test_dataset = torch.utils.data.TensorDataset(
        test_inputs["input_ids"], test_inputs["attention_mask"]
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)

    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference on Test Data"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    test_data["predicted_winner"] = [
        "model_a" if pred == 0 else "model_b" for pred in predictions
    ]

    submision_csv = test_data[["id", "predicted_winner"]]
    submision_csv.to_csv("submision.csv", index=False)

    print("\n✅ Inference on test data complete! Predictions saved.")


sequence_feature_engineering_pipeline()
inference_on_test_data()
