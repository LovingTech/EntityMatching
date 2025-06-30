from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

DEVICE = "cpu"

class BERTTripletEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-uncased")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_emb

class EmbeddingModel:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
        self.model = BERTTripletEncoder()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.to(DEVICE)

    def encode(self, text: str):
        tokenized_result = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
        text_tokens, attention = tokenized_result["input_ids"], tokenized_result["attention_mask"]
        text_tokens.to(DEVICE)
        attention.to(DEVICE)
        return self.model(text_tokens, attention).squeeze().tolist()  # Returns a Python list
