import torch
from torch.utils.data import Dataset
import pandas as pd


class ChatbotDataset(Dataset):
    def __init__(
        self,
        data_path="data/processed/tokenized_data.pt",
        label_path="data/processed/labels.pkl",
    ):
        # Load tokenized data (now as a PyTorch tensor)
        tokenized_data = torch.load(data_path)

        # Load the labels (still as a Pandas Series)
        labels = torch.tensor(pd.read_pickle(label_path).values, dtype=torch.long)

        # Verify the tokenized data and labels match
        assert (
            len(tokenized_data["input_ids"]) == len(labels)
        ), f"Tokenized data length {len(tokenized_data['input_ids'])} does not match labels length {len(labels)}"

        # Assign input_ids, attention_mask, and labels as tensors
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
