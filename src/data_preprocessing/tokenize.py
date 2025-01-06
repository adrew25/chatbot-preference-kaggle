import torch
import pickle
import os
from transformers import AutoTokenizer  # Import here to avoid circular imports
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_data_in_batches(
    df,
    batch_size=128,
    cache_path="data/processed/tokenized_data.pkl",
    labels_cache_path="data/processed/labels.pkl",
):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    inputs = {
        "input_ids": [],
        "attention_mask": [],
    }

    labels = []

    # Tokenize in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Tokenizing Data"):
        batch_df = df.iloc[i : i + batch_size]

        batch_prompts = batch_df["prompt"].tolist()
        batch_resp_a = batch_df["response_a"].tolist()
        batch_resp_b = batch_df["response_b"].tolist()

        # Tokenizing prompt and response_a
        with torch.no_grad():
            inputs_a = tokenizer(
                batch_prompts,
                batch_resp_a,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_attention_mask=True,
            )
            inputs_b = tokenizer(
                batch_prompts,
                batch_resp_b,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_attention_mask=True,
            )

        # Concatenate the input_ids and attention_mask for both pairs
        inputs["input_ids"].append(
            torch.cat([inputs_a["input_ids"], inputs_b["input_ids"]], dim=0)
        )
        inputs["attention_mask"].append(
            torch.cat([inputs_a["attention_mask"], inputs_b["attention_mask"]], dim=0)
        )

        # Extract labels (same logic as before)
        batch_labels = (
            (batch_df["winner"] == "a").astype(int).tolist()
        )  # 1 if winner is 'a', 0 if winner is 'b'

        # Duplicate the labels to match both tokenized pairs (response_a and response_b)
        labels.extend(batch_labels * 2)  # Duplicate the labels for both responses

    # Concatenate all batches for inputs
    inputs["input_ids"] = torch.cat(inputs["input_ids"], dim=0)
    inputs["attention_mask"] = torch.cat(inputs["attention_mask"], dim=0)

    # Save tokenized inputs and labels
    with open(cache_path, "wb") as f:
        pickle.dump(inputs, f)

    with open(labels_cache_path, "wb") as f:
        pickle.dump(labels, f)

    print(f"✅ Tokenization complete! Saved to {cache_path}")
    print(f"✅ Labels saved to {labels_cache_path}")
    return inputs, labels
