import psycopg2
import os
from models import *
from utils import *
from training import *
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
from finetuning_bert_multilingual import *
import dotenv

dotenv.load_dotenv()


DB_NAME = "lei"
DB_USER = "postgres"
DB_PASSWORD =  os.getenv("POSTGRES_PASSWORD")
DB_HOST = "127.0.0.1"
DB_PORT = "5432"

VERSION = "2"

conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
)

conn.rollback()
with conn.cursor() as curr:
    curr.execute("""DROP TABLE IF EXISTS Embed_v{}""".format(VERSION))
    curr.execute(
    """CREATE TABLE Embed_v{} (
        id uuid DEFAULT gen_random_uuid() PRIMARY KEY REFERENCES Names(id),
        embedding vector(768) NOT NULL
                )""".format(VERSION))
conn.commit()


# tokenizer params
tokenizer_path = "./tokenizer.json"
# DB params
BATCH_SIZE = 100
DEVICE = "mps"

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
# model setup
model = BERTTripletEncoder(MODEL_NAME)
model.load_state_dict(torch.load("./models/bert_triplet_finetuned_v3.pt", weights_only=True))
model.train(False)
model.to(DEVICE)


# Tokenizer utility
def encode(text, max_len):
    return tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

def encode_batch(texts):
    max_len = 64
    token_ids = []
    attention = []
    for t in texts:
        result_tokenzied = encode(t, max_len)
        token_ids.append(result_tokenzied["input_ids"].squeeze(1).tolist())
        attention.append(result_tokenzied["attention_mask"].squeeze(1).tolist())
    return torch.tensor(token_ids).squeeze(1).to(DEVICE), torch.tensor(attention).squeeze(1).to(DEVICE)

# Main loop with paging
offset = 3238600
while True:
    with conn.cursor() as curr:
        curr.execute("""
            SELECT id, name FROM Names
            ORDER BY id
            LIMIT %s OFFSET %s
        """, (BATCH_SIZE, offset))
        rows = curr.fetchall()

    if not rows:
        break  # Done paging

    ids, names = zip(*rows)
    tokens, attention = encode_batch(names)
    embeddings = model(tokens, attention)  # [B, 768]
    print(embeddings.shape)
    with conn.cursor() as curr:
        args_str = ",".join(
            curr.mogrify("(%s, %s)", (id_, emb.tolist())).decode("utf-8") for id_, emb in zip(ids, embeddings))
        curr.execute(f"INSERT INTO Embed_v{VERSION} (id, embedding) VALUES {args_str}")

    conn.commit()
    offset += BATCH_SIZE
    print(f"Inserted batch at offset {offset}")
