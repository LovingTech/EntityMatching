from fastapi import FastAPI
from pydantic import BaseModel
from model import EmbeddingModel
from db import DB
import os
import dotenv

dotenv.load_dotenv()

DB_NAME = "lei"
DB_USER = "postgres"
DB_PASSWORD =  os.getenv("POSTGRES_PASSWORD")
DB_HOST = "127.0.0.1"
DB_PORT = "5432"

app = FastAPI()

# Setup
model = EmbeddingModel("bert_triplet_finetuned_v3.pt")
db = DB(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")  # Replace with actual DSN

class QueryRequest(BaseModel):
    text: str
    top_k: int = 50

@app.post("/search")
def search(req: QueryRequest):
    embedding = model.encode(req.text)
    results = db.search(embedding, top_k=req.top_k)
    return {"results": [{"text": text, "score": float(score)} for text, score in results]}
