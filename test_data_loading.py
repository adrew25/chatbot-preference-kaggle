from src.Datasets.dataset import ChatbotDataset
from torch.utils.data import DataLoader
import torch
from src.models.bert_classifier import BertClassifier

if __name__ == "__main__":
    # Load the data
    dataset = ChatbotDataset(
        data_path="data/processed/tokenized_data.pkl",
        label_path="data/processed/labels.pkl",
    )

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Initialize the data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the model
    model = BertClassifier(
        model_name="distilbert-base-multilingual-cased", num_classes=2
    )
    model.to(device)

    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Device: {device}")
