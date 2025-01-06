import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.Datasets.dataset import ChatbotDataset
from src.models.distilbert_classifier import DistilBERTClassifier
import os

os.makedirs("src/models/saved_pths", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dataset = ChatbotDataset()

# Split the dataset into training and validation sets 80% - 20%
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

print(f"Training set size: {len(train_loader.dataset)}")
print(f"Validation set size: {len(val_loader.dataset)}")


# for batch in train_loader:
#     input_ids, attention_mask, labels = (
#         batch["input_ids"],
#         batch["attention_mask"],
#         batch["labels"],
#     )
#     print(
#         "Input IDs Sample:", input_ids[0][:10]
#     )  # Print a small sample of the tokenized input
#     print("Labels Sample:", labels[:5])  # Check if labels are 0/1 correctly formatted
#     break

# import numpy as np

# labels = [batch["labels"] for batch in train_loader]
# all_labels = torch.cat(labels).cpu().numpy()

# unique, counts = np.unique(all_labels, return_counts=True)
# print("Class distribution in the dataset:", dict(zip(unique, counts)))

# import pandas as pd

# labels = pd.read_pickle("data/processed/labels.pkl")
# print(labels.value_counts())

# exit()
# Initialize model
model = DistilBERTClassifier()
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        if accuracy > 50:
            torch.save(
                model.state_dict(),
                f"src/models/saved_pths/DistilBERTClassifier_{accuracy:.2f}.pth",
            )
        print(f"Epoch {epoch+1} - Training Loss: {total_loss/len(train_loader):.4f}")
        print(
            f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%"
        )


train_model(model, train_loader, val_loader, optimizer, device)
