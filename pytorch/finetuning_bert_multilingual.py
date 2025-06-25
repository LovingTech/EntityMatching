import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import Dataset, DataLoader
import random
from dataset import *
import mlflow
from tqdm import tqdm


mlflow.pytorch.autolog()

# === 1. Load BERT tokenizer and model ===
MODEL_NAME = "bert-base-multilingual-uncased"
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
MAX_LEN = 64
DEVICE = "mps"


tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert = BertModel.from_pretrained(MODEL_NAME)


# === 2. Dataset: assume you have (anchor, positive, negative) triplets ===

class TripletTextDataset(Dataset):
    def __init__(self, tokenizer, max_len=64):
        self.triplets = NameMatchingDataset()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]
        
        def encode(text):
            return self.tokenizer(
                text, 
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )

        return {
            "anchor": encode(a),
            "positive": encode(p),
            "negative": encode(n),
        }


class BERTTripletEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_emb

def main():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    dataset = TripletTextDataset(tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BERTTripletEncoder(MODEL_NAME).to(DEVICE)
    loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    with mlflow.start_run():    
        for epoch in range(EPOCHS):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                anchor = batch["anchor"]
                positive = batch["positive"]
                negative = batch["negative"]

                # Move tensors to device
                anchor_emb = model(
                    anchor["input_ids"].squeeze(1).to(DEVICE),
                    anchor["attention_mask"].squeeze(1).to(DEVICE),
                )
                positive_emb = model(
                    positive["input_ids"].squeeze(1).to(DEVICE),
                    positive["attention_mask"].squeeze(1).to(DEVICE),
                )
                negative_emb = model(
                    negative["input_ids"].squeeze(1).to(DEVICE),
                    negative["attention_mask"].squeeze(1).to(DEVICE),
                )

                loss = loss_fn(anchor_emb, positive_emb, negative_emb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            


            torch.save(model.state_dict(), f"bert_triplet_finetuned_e{epoch}.pt")
            print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "bert_triplet_finetuned.pt")
    print("Model saved as 'bert_triplet_finetuned.pt'")

if __name__ == "__main__":
    main()
