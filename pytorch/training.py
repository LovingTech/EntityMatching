# MLX imports
import torch
import torch.nn as nn
from torch.optim import Adam

# My imports
from models import *
from dataset import *
from utils import *

# Tokenizer imports
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, Sequence as NormalizerSequence

# Misc imports
import numpy as np
import random
import os

def getTokenizer(path):
    if os.path.exists(path):
        tokenizer = Tokenizer.from_file(path)
    else:
        dataset = NameMatchingDataset()
        texts = [x for triplet in dataset for x in triplet]

        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.normalizer = NormalizerSequence([NFD(), Lowercase()])
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<PAD>", "<UNK>"])
        tokenizer.train_from_iterator(texts[:100000], trainer)
        tokenizer.save(path)
    return tokenizer

### Parameters

# training params
tokenizer_path = "./tokenizer.json"
batch_size = 128
vocab_size = 15000
epochs = 20
learning_rate = 1e-2
prefetch = 8
# model params
d_out_model=768
num_att_heads=12
num_hidden_layers=12
d_hidden_ff=3072
max_seq_len=512

models_folder = "./models"
filename_fn =  lambda epoch, batch: f"model_e{epoch}_b{batch}_torch.pth"
model_file_name = lambda epoch, batch: f"{models_folder}/{filename_fn(epoch,batch)}"


def main():
    # Get main objects
    tokenizer = getTokenizer(tokenizer_path)
    raw_data = NameMatchingDataset()
    dataset = TokenizedDataset(raw_data, tokenizer)
    model = Model(
            vocab_size=vocab_size,
            d_model = d_out_model,
            num_heads = num_att_heads,
            num_layers = num_hidden_layers,
            d_ff = d_hidden_ff,
            max_len = max_seq_len
            )

    # Resume from latest checkpoint
    start_epoch, latest_checkpoint = get_latest_checkpoint(models_folder,filename_fn)
    if latest_checkpoint:
        model = torch.load(latest_checkpoint, weights_only = True)

    model.train()

    # Setup loss and optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.TripletMarginLoss()

    def step(a,p,n):
        loss = loss_fn(model(a), model(p), model(n))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    # Training loop
    print("Training Encoder")
    for epoch in range(start_epoch + 1, start_epoch + 1 + epochs):
        epoch_loss = 0

        data_iter = batch_iterator(dataset, batch_size=batch_size)
        prefetch_iter = PrefetchIterator(data_iter, prefetch=prefetch)

        for batch,(xa, xp, xn) in enumerate(prefetch_iter):
            if (batch+1) % 10 == 0:
                print(f"Data: {(batch+1)*batch_size}/{len(dataset)}")

            if (batch+1) % 500 == 0:
                torch.save(model, model_file_name(epoch, batch+1))

            epoch_loss = float(step(xa,xp,xn).item())

        # Print stats about epoch and attempt to save model
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f} Memory used: {torch.cuda.max_memory_allocated() / (1024**3):.4f}GB")
        torch.cuda.reset_max_memory_allocated()
        try:
            torch.save(model, model_file_name(epoch, int(len(dataset)/128)))
        except:
            print("Model save failed")

if __name__ == '__main__':
    main()
