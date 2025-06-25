import psycopg2
import os
from models import *
from utils import *
from training import *
from dataset import encode
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import dotenv

dotenv.load_dotenv()


DB_NAME = "lei"
DB_USER = "postgres"
DB_PASSWORD =  os.getenv("POSTGRES_PASSWORD")
DB_HOST = "db"
DB_PORT = "5432"

conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
)

conn.rollback()
with conn.cursor() as curr:
    curr.execute("""DROP TABLE IF EXISTS Embed_v1""")
    curr.execute(
    """CREATE TABLE Embed_v1 (
        id uuid DEFAULT gen_random_uuid() PRIMARY KEY REFERENCES Names(id),
        embedding vector(768) NOT NULL
                )""")
conn.commit()


# model params
d_out_model=768
num_att_heads=12
num_hidden_layers=12
d_hidden_ff=3072
max_seq_len=512
# tokenizer params
tokenizer_path = "./tokenizer.json"
# DB params
BATCH_SIZE = 1000

tokenizer = getTokenizer(tokenizer_path)
model = Model(
    vocab_size=vocab_size,
    d_model = d_out_model,
    num_heads = num_att_heads,
    num_layers = num_hidden_layers,
    d_ff = d_hidden_ff,
    max_len = max_seq_len
    )

model.load_weights("./models/model_finetuned_e11_b7372.safetensors")

model.eval()

# Tokenizer utility
def encode_batch(texts):
    max_len = 64
    token_ids = [
        encode(t,tokenizer, max_len)
        for t in texts
    ]
    return mx.array(np.array(token_ids))

# Main loop with paging
offset = 0
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
    tokens = encode_batch(names)
    embeddings = np.array(model(tokens))  # [B, 768]
    print(embeddings.shape)
    with conn.cursor() as curr:
        args_str = ",".join(
            curr.mogrify("(%s, %s)", (id_, emb.tolist())).decode("utf-8") for id_, emb in zip(ids, embeddings))
        curr.execute(f"INSERT INTO Embed_v1 (id, embedding) VALUES {args_str}")

    conn.commit()
    offset += BATCH_SIZE
    print(f"Inserted batch at offset {offset}")
