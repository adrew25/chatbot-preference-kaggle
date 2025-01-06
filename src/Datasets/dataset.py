import pickle
from torch.utils.data import Dataset


class ChatbotDataset(Dataset):
    def __init__(
        self,
        data_path="data/processed/tokenized_data.pkl",
        label_path="data/processed/labels.pkl",
    ):
        # Load the tokenized data
        with open(data_path, "rb") as f:
            tokenized_data = pickle.load(f)

        # Load the labels
        with open(label_path, "rb") as f:
            labels = pickle.load(f)

        # Verify the tokenized data and labels match
        assert (
            len(tokenized_data["input_ids"]) == len(labels)
        ), f"Tokenized data length {len(tokenized_data['input_ids'])} does not match labels length {len(labels)}"

        # Assign input_ids, attention_mask, and labels
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
