from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.Datasets.dataset import ChatbotDataset
import torch

# Load the pre-trained tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Function to check token lengths in the dataset
def check_token_lengths_from_loader(
    data_loader, tokenizer, max_length=512, num_samples=10
):
    checked_samples = 0

    for batch in tqdm(data_loader, desc="Checking token lengths"):
        input_ids = batch["input_ids"].to("cpu")

        for idx in range(len(input_ids)):
            # Decode to get original text
            token_length = len(input_ids[idx])

            print(f"Sample {checked_samples + 1}: Token length = {token_length}")
            if token_length > max_length:
                print(f"Warning: Token length exceeds {max_length} tokens!")

            checked_samples += 1
            if checked_samples >= num_samples:
                return


dataset = ChatbotDataset(data_path="data/processed/tokenized_data.pkl")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
check_token_lengths_from_loader(train_loader, tokenizer)
