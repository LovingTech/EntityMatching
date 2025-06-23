from collections import Counter, defaultdict
import heapq
import json



def getTokenizer():
    if os.path.exists(tokenizer_path):
        print("Tokenizer Loaded")
        return t.load(tokenizer_path)
    else:
        all_texts = [t for triplet in texts for t in triplet]
        tokenizer = BPE(num_merges=15000)
        print("Tokenizer Training")
        tokenizer.train(all_texts[:50000])
        tokenizer.save(tokenizer_path)
        print("Tokenizer Saved")
        return tokenizer


class BPE:
    def __init__(self, num_merges=100):
        self.num_merges = num_merges
        self.merges = []
        self.token2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2token = {0: "<PAD>", 1: "<UNK>"}

    def train(self, corpus):
        tokens = [tuple(word) + ("</w>",) for word in corpus]
        vocab = Counter(tokens)

        # Initial pair frequency table
        pair_freqs = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i+1])] += freq

        # Heap of pairs for fast max lookup
        pair_heap = [(-count, pair) for pair, count in pair_freqs.items()]
        heapq.heapify(pair_heap)

        for i in range(self.num_merges):
            print(f"Merge: {i}")
            if not pair_heap:
                break

            # Get the most frequent pair
            _, best_pair = heapq.heappop(pair_heap)
            if pair_freqs[best_pair] == 0:
                continue

            self.merges.append(best_pair)
            new_vocab = Counter()

            affected_pairs = defaultdict(int)

            for word, freq in vocab.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                        new_word.append(''.join(best_pair))
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                new_word = tuple(new_word)
                new_vocab[new_word] += freq

                # Update affected pair frequencies
                for i in range(len(new_word) - 1):
                    pair = (new_word[i], new_word[i+1])
                    affected_pairs[pair] += freq

            vocab = new_vocab
            pair_freqs = affected_pairs

            # Rebuild heap with updated pairs
            pair_heap = [(-count, pair) for pair, count in pair_freqs.items()]
            heapq.heapify(pair_heap)

        # Build final vocab
        final_tokens = set()
        for word in vocab:
            final_tokens.update(word)

        for tok in sorted(final_tokens):
            if tok not in self.token2id:
                idx = len(self.token2id)
                self.token2id[tok] = idx
                self.id2token[idx] = tok


    def _replace(self, token, pair):
        merged = ''.join(pair)
        i = 0
        result = []
        while i < len(token):
            if i < len(token) - 1 and (token[i], token[i+1]) == pair:
                result.append(merged)
                i += 2
            else:
                result.append(token[i])
                i += 1
        return result

    def get_stats(self, tokens):
        pairs = Counter()
        for token in tokens:
            for i in range(len(token)-1):
                pair = (token[i], token[i+1])
                pairs[pair] += 1
        return pairs

    def tokenize(self, word):
        word = list(word) + ["</w>"]
        for merge in self.merges:
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i+1]) == merge:
                    word[i:i+2] = ["".join(merge)]
                else:
                    i += 1
        return word

    def encode(self, text, max_len=25):
        tokens = self.tokenize(text)
        ids = [self.token2id.get(t, self.token2id["<UNK>"]) for t in tokens]
        return ids[:max_len] + [0] * (max_len - len(ids))  # pad

    def save(self, filepath):
        data = {
            "num_merges": self.num_merges,
            "merges": self.merges,
            "token2id": self.token2id,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        bpe = cls(num_merges=data["num_merges"])
        bpe.merges = [tuple(m) for m in data["merges"]]
        bpe.token2id = {k: int(v) for k, v in data["token2id"].items()}
        bpe.id2token = {v: k for k, v in bpe.token2id.items()}
        return bpe
