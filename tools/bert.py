import torch
import torch.nn as nn
from transformers import BertModel

class IntentClassifier(nn.Module):
    def __init__(self, num_classes, hidden_layer_size=128, dropout_rate=0.2):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_size, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_embedding)

        return logits