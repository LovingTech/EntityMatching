# MLX imports
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam

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
import multiprocessing as mp
from functools import partial
import mlflow

def triplet_loss(anchor, positive, negative, margin=2.0):
    pos_dist = mx.sum((anchor - positive) ** 2, axis=1)
    neg_dist = mx.sum((anchor - negative) **2, axis=1)
    loss = mx.maximum(pos_dist-neg_dist + margin, 0.0)
    return mx.mean(loss)

def loss_fn(m, a, p, n):
    va = m(a)
    vp = m(p)
    vn = m(n)
    return triplet_loss(va, vp, vn)

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
filename_fn =  lambda epoch, batch: f"model_e{epoch}_b{batch}.safetensors"
model_file_name = lambda epoch, batch: f"{models_folder}/{filename_fn(epoch,batch)}" 


def main():
    mlflow.set_experiment("entity-matching-encoder")  # Optional: set experiment name
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "vocab_size": vocab_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "d_model": d_out_model,
            "num_heads": num_att_heads,
            "num_layers": num_hidden_layers,
            "d_ff": d_hidden_ff,
            "max_seq_len": max_seq_len
        })

        # Tokenizer, dataset, and model
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

        start_epoch, latest_checkpoint = get_latest_checkpoint(models_folder, filename_fn) 
        if latest_checkpoint:
            model.load_weights(latest_checkpoint)

        model.train()
        optimizer = Adam(learning_rate=learning_rate)
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        @partial(mx.compile, inputs=(model.state, optimizer.state), outputs=(model.state, optimizer.state))
        def step(a, p, n):
            loss, grads = loss_and_grad_fn(model, a, p, n)
            optimizer.update(model, grads)
            return loss

        print("Training Encoder")
        for epoch in range(start_epoch + 1, start_epoch + 1 + epochs):
            epoch_loss = 0.0
            data_iter = batch_iterator(dataset, batch_size=batch_size)
            prefetch_iter = PrefetchIterator(data_iter, prefetch=prefetch)
           
            
            for batch, (xa, xp, xn) in enumerate(prefetch_iter):
                if (batch + 1) % 10 == 0:
                    print(f"Data: {(batch + 1) * batch_size}/{len(dataset)}")

                loss_val = float(step(xa, xp, xn).item())
                epoch_loss += loss_val

                # Log metrics every 500 batches
                if (batch + 1) % 500 == 0:
                    memory_used = mx.get_peak_memory() / (1024**3)
                    mlflow.log_metric("batch_loss", loss_val, step=epoch * 10_000 + batch + 1)  # Unique step
                    mlflow.log_metric("batch_memory_gb", memory_used, step=epoch * 10_000 + batch + 1)
                    print(f"[Batch {batch+1}] Loss: {loss_val:.4f} Memory: {memory_used:.4f} GB")

                    # Save model checkpoint
                    model_path = model_file_name(epoch, batch + 1)
                    try:
                        mx.eval(model.parameters())
                        model.save_weights(model_path)
                        mlflow.log_artifact(model_path)
                    except Exception as e:
                        print(f"Model save failed at batch {batch+1}: {e}")

            memory_used = mx.get_peak_memory() / (1024**3)
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f} Memory: {memory_used:.4f} GB")

            # Log metrics to MLflow
            mlflow.log_metric("loss", epoch_loss, step=epoch)
            mlflow.log_metric("memory_gb", memory_used, step=epoch)

            mx.reset_peak_memory()

            try:
                mx.eval(model.parameters())
                model_path = model_file_name(epoch, int(len(dataset)/128))
                model.save_weights(model_path)
                mlflow.log_artifact(model_path)
            except Exception as e:
                print(f"Model save failed: {e}")

if __name__ == '__main__':
    main()
