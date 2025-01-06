import torch
import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_data_in_batches(
    df,
    batch_size=128,
    cache_path="data/processed/tokenized_data.pt",
    labels_cache_path="data/processed/labels.pkl",
):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

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

    torch.save(inputs, cache_path)
    if labels:
        pd.Series(labels).to_pickle(labels_cache_path)

    print(f"✅ Tokenization complete! Saved to {cache_path}")
    return inputs, labels
