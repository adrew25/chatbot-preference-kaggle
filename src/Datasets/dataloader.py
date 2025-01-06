from torch.utils.data import DataLoader
from src.Datasets.dataset import ChatbotDataset


def get_chatbot_dataloader(
    batch_size=32, shuffle=True, tokenized_data_path="data/processed/tokenized_data.pkl"
):
    dataset = ChatbotDataset(tokenized_data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Test use
if __name__ == "__main__":
    dataloader = get_chatbot_dataloader()
    for batch in dataloader:
        print(batch["input_ids"].shape, batch["attention_mask"].shape)
        break
