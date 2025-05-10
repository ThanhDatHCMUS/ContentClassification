import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

class PhoBERTClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # Nếu input có 3 chiều: (B, N, L), reshape về (B*N, L)
        if input_ids.dim() == 3:
            B, N, L = input_ids.shape
            input_ids = input_ids.view(-1, L)
            attention_mask = attention_mask.view(-1, L)
            reshape_back = True
        else:
            reshape_back = False

        # Đưa qua BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden)

        # Nếu cần reshape lại (dạng (B, N, hidden))
        if reshape_back:
            cls_embeddings = cls_embeddings.view(B, N, -1)
            sentence_embeddings = cls_embeddings.mean(dim=1)  # (B, hidden)
        else:
            sentence_embeddings = cls_embeddings  # (B, hidden) khi B = 1

        logits = self.classifier(self.dropout(sentence_embeddings))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


class ViTForSequenceClassification(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super(ViTForSequenceClassification, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
