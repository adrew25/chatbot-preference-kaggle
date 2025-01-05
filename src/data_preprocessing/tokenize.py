import torch
from transformers import AutoTokenizer
import pandas as pd
import pickle
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-multilingual-cased", use_fast=True
)


def tokenize_data_in_batches(
    df, batch_size=100, cache_path="data/processed/tokenized_data.pkl"
):
    """
    Tokenizes data in batches and saves the tokenized outputs to disk.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    inputs = {
        "input_ids": [],
        "attention_mask": [],
    }

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

        batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
        inputs["input_ids"].append(batch_inputs["input_ids"].cpu())
        inputs["attention_mask"].append(batch_inputs["attention_mask"].cpu())

    inputs["input_ids"] = torch.cat(inputs["input_ids"], dim=0)
    inputs["attention_mask"] = torch.cat(inputs["attention_mask"], dim=0)

    # Save to processed data directory
    with open(cache_path, "wb") as f:
        pickle.dump(inputs, f)

    print(f"✅ Tokenization complete! Saved to {cache_path}")
    return inputs


def load_tokenized_data(cache_path="data/processed/tokenized_data.pkl"):
    with open(cache_path, "rb") as f:
        inputs = pickle.load(f)

    inputs = {
        key: value.to(device)
        for key, value in tqdm(inputs.items(), desc="Loading Tokenized Data")
    }
    print(f"✅ Loaded tokenized data from {cache_path}")
    return inputs
