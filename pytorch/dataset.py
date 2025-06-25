import csv
import numpy as np

def encode(text, tokenizer, max_len):
    ids = tokenizer.encode(text).ids
    return (ids + [0] * max(0, max_len - len(ids)))[:max_len]


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
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        return tuple(np.array(encode(a, self.tokenizer, self.max_len), dtype=np.int32) for a in self.dataset[idx])


    
