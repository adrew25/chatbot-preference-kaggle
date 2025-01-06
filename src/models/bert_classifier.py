from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(
        self, model_name="distilbert-base-multilingual-cased", num_classes=1
    ):  # num_classes = 2 for binary classification
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # We use the [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits
