import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.Datasets.dataset import ChatbotDataset
from src.models.DistilBertClassifier import DistilBERTClassifier


import os

os.makedirs("models/saved_pths", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dataset = ChatbotDataset(
    data_path="data/processed/tokenized_data.pkl",
    label_path="data/processed/labels.pkl",
)

# Split the dataset into training and validation sets 80% - 20%
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

print(f"Training set size: {len(train_loader.dataset)}")
print(f"Validation set size: {len(val_loader.dataset)}")


# Initialize model
model = DistilBERTClassifier()
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=1):
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

        print(f"Epoch {epoch+1} - Training Loss: {total_loss/len(train_loader):.4f}")
        print(
            f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%"
        )

    # Save the model
    torch.save(model.state_dict(), "models/saved_pths/distilbert_model.pth")
    print("Model saved successfully")


train_model(model, train_loader, val_loader, optimizer, device)
