import csv
import numpy as np
import psycopg2

def encode(text, tokenizer, max_len):
    ids = tokenizer.encode(text).ids
    return (ids + [0] * max(0, max_len - len(ids)))[:max_len]

""" A dataset class for name matching tasks, loading triplets from a database using a SQL query. The anchor, positive pair is determined already in a view and the negative case is selected based on the levenstein distance."""
class DatabaseNameMatchingDataset:
    def __init__(self, db_config):
        self.connection = psycopg2.connect(**db_config)
        self.query = """
        SELECT anchor, positive, negative"""
        self.triplets = self.load_triplets()

    def load_triplets(self):
        with self.connection.cursor() as cursor:
            cursor.execute(self.query)
            return cursor.fetchall()

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


class NameMatchingDataset:
    def __init__(self, data_csv="training.csv"):
        with open(data_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            self.triplets = [(row[1],row[2], row[3]) for row in reader]

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]
        
class TokenizedDataset:
    def __init__(self, dataset, tokenizer, data_csv="training.csv", max_len=64):
        super().__init__(data_csv)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        return tuple(np.array(encode(a, self.tokenizer, self.max_len), dtype=np.int32) for a in self.dataset[idx])


    
