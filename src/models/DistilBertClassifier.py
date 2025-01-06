import torch.nn as nn
from transformers import DistilBertModel


class DistilBERTClassifier(nn.Module):
    def __init__(self):
        super(DistilBERTClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(
            "distilbert-base-multilingual-cased"
        )
        self.classifier = nn.Linear(
            self.distilbert.config.hidden_size, 2
        )  # Assuming binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        return self.classifier(pooled_output)
