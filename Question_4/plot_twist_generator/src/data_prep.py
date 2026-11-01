"""Data preparation and dataset loading for Question 4: The Plot-Twist Generator.

Handles:
- Loading the TinyStories dataset from Hugging Face (`roneneldan/TinyStories`).
- Caching splits locally to disk (`data/`) for reproducible, fast subsequent loading.
- Story-level splitting to prevent cross-story sequence overlap / data leakage.
- Normalization and punctuation-preserving tokenization.
- Shared vocabulary building with special tokens (<PAD>, <UNK>, <BOS>, <EOS>).
- Constructing next-token prediction pairs (x, y) chunked strictly within story boundaries.
- Batch collation with right-padding and sequence lengths.
- Dataset statistics, OOV analysis, and sanity checks.
"""

import os
import re
import json
import random
from collections import Counter
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

SPECIAL_TOKENS: Dict[str, int] = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def clean_and_tokenize(text: str) -> List[str]:
    """Normalize whitespace/quotes, lowercase, and tokenize words + punctuation.
    
    Preserving punctuation allows language models to learn sentence boundaries,
    dialogue quotes, and natural cadence.
    """
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("—", " - ")
    text = re.sub(r"\s+", " ", text.strip().lower())
    return TOKEN_PATTERN.findall(text)


class Vocabulary:
    """Shared vocabulary for word-level language modeling."""

    def __init__(self, max_size: int = 10000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.stoi: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.itos: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.token_counts: Dict[str, int] = {}
        self.raw_vocab_size: int = 0

    def build_from_stories(self, stories: List[str]):
        """Construct vocabulary from training stories only (never touching val/test)."""
        counter = Counter()
        for story in stories:
            counter.update(clean_and_tokenize(story))

        self.token_counts = dict(counter)
        self.raw_vocab_size = len(counter)

        # Sort by frequency descending, then alphabetically for determinism
        sorted_tokens = sorted(
            [tok for tok, freq in counter.items() if freq >= self.min_freq],
            key=lambda x: (-counter[x], x)
        )

        for tok in sorted_tokens[: self.max_size - len(SPECIAL_TOKENS)]:
            if tok not in self.stoi:
                idx = len(self.stoi)
                self.stoi[tok] = idx
                self.itos[idx] = tok

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Convert a text string to a list of token IDs."""
        tokens = clean_and_tokenize(text)
        ids = [self.stoi.get(t, SPECIAL_TOKENS["<UNK>"]) for t in tokens]
        if add_bos:
            ids = [SPECIAL_TOKENS["<BOS>"]] + ids
        if add_eos:
            ids = ids + [SPECIAL_TOKENS["<EOS>"]]
        return ids

    def decode(self, ids: List[int], stop_at_eos: bool = True) -> str:
        """Convert a list of token IDs back into readable text."""
        words = []
        for idx in ids:
            if idx == SPECIAL_TOKENS["<EOS>"] and stop_at_eos:
                break
            if idx in (SPECIAL_TOKENS["<PAD>"], SPECIAL_TOKENS["<BOS>"]):
                continue
            words.append(self.itos.get(idx, "<UNK>"))
        text = " ".join(words)
        # Detokenize basic punctuation attached to preceding words
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text

    def compute_oov_rate(self, stories: List[str]) -> Tuple[float, int, int]:
        """Compute the percentage and count of out-of-vocabulary (<UNK>) tokens in a story split."""
        total_tokens = 0
        oov_tokens = 0
        unk_id = SPECIAL_TOKENS["<UNK>"]
        for story in stories:
            encoded = self.encode(story, add_bos=False, add_eos=False)
            total_tokens += len(encoded)
            oov_tokens += sum(1 for tok in encoded if tok == unk_id)
        rate = (oov_tokens / max(1, total_tokens)) * 100.0
        return rate, oov_tokens, total_tokens

    def save(self, file_path: str):
        """Save vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "max_size": self.max_size,
                "min_freq": self.min_freq,
                "raw_vocab_size": self.raw_vocab_size,
                "stoi": self.stoi,
            }, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: str) -> "Vocabulary":
        """Load vocabulary from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(max_size=data["max_size"], min_freq=data["min_freq"])
        vocab.raw_vocab_size = data.get("raw_vocab_size", len(data["stoi"]))
        vocab.stoi = data["stoi"]
        vocab.itos = {int(v): k for k, v in data["stoi"].items()}
        return vocab

    def __len__(self) -> int:
        return len(self.stoi)


class StoryLMDataset(Dataset):
    """PyTorch Dataset for next-token prediction language modeling.
    
    Preserves story boundaries by creating training sequences strictly within
    individual stories, preventing cross-story narrative leakage.
    """

    def __init__(self, stories: List[str], vocab: Vocabulary, max_seq_len: int = 128):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for story in stories:
            token_ids = vocab.encode(story, add_bos=True, add_eos=True)
            if len(token_ids) < 2:
                continue

            # Slice strictly within this story's token boundaries
            for start in range(0, len(token_ids) - 1, max_seq_len):
                chunk = token_ids[start : start + max_seq_len + 1]
                if len(chunk) < 2:
                    continue
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad batch of variable length sequences and return lengths for packed sequences."""
    xs, ys = zip(*batch)
    pad_id = SPECIAL_TOKENS["<PAD>"]
    x_padded = pad_sequence(xs, batch_first=True, padding_value=pad_id)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=pad_id)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    return x_padded, y_padded, lengths


def load_and_split_tinystories(
    num_train: int = 20000,
    num_val: int = 1000,
    num_test: int = 1000,
    seed: int = 42,
    min_story_len: int = 20,
    cache_dir: Optional[str] = "data",
) -> Tuple[List[str], List[str], List[str]]:
    """Load TinyStories from Hugging Face and split into train, val, and test subsets.
    
    If cached files exist in `cache_dir`, loads them directly. Otherwise, streams
    from Hugging Face 'roneneldan/TinyStories' and caches locally for subsequent runs.
    
    Splitting is performed strictly at the full-story level before any chunking.
    """
    if cache_dir is not None:
        train_path = os.path.join(cache_dir, "train_stories.json")
        val_path = os.path.join(cache_dir, "val_stories.json")
        test_path = os.path.join(cache_dir, "test_stories.json")
        if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
            print(f"Loading cached splits from directory: '{cache_dir}'...")
            with open(train_path, "r", encoding="utf-8") as f:
                train_stories = json.load(f)
            with open(val_path, "r", encoding="utf-8") as f:
                val_stories = json.load(f)
            with open(test_path, "r", encoding="utf-8") as f:
                test_stories = json.load(f)
            if len(train_stories) >= num_train and len(val_stories) >= num_val and len(test_stories) >= num_test:
                print(f"Successfully loaded cached data: {len(train_stories[:num_train])} train, "
                      f"{len(val_stories[:num_val])} val, {len(test_stories[:num_test])} test stories.")
                return train_stories[:num_train], val_stories[:num_val], test_stories[:num_test]

    random.seed(seed)
    total_needed_from_train = num_train + num_val
    print(f"Streaming {total_needed_from_train} stories from Hugging Face 'roneneldan/TinyStories' (train split)...")

    train_stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    collected_train_pool: List[str] = []

    for item in train_stream:
        text = item.get("text", "").strip()
        if len(text) >= min_story_len:
            collected_train_pool.append(text)
            if len(collected_train_pool) >= total_needed_from_train:
                break

    # Shuffle with fixed seed for deterministic split
    random.shuffle(collected_train_pool)
    train_stories = collected_train_pool[:num_train]
    val_stories = collected_train_pool[num_train : num_train + num_val]

    print(f"Streaming {num_test} stories from Hugging Face 'roneneldan/TinyStories' (validation split) as held-out test...")
    val_stream = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    test_stories: List[str] = []

    for item in val_stream:
        text = item.get("text", "").strip()
        if len(text) >= min_story_len:
            test_stories.append(text)
            if len(test_stories) >= num_test:
                break

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Caching story splits to directory: '{cache_dir}'...")
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_stories, f, ensure_ascii=False)
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_stories, f, ensure_ascii=False)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_stories, f, ensure_ascii=False)

    print(f"Successfully loaded and cached: {len(train_stories)} train, {len(val_stories)} val, {len(test_stories)} test stories.")
    return train_stories, val_stories, test_stories


def get_dataloaders(
    train_stories: List[str],
    val_stories: List[str],
    test_stories: List[str],
    vocab: Optional[Vocabulary] = None,
    max_vocab_size: int = 10000,
    min_freq: int = 2,
    max_seq_len: int = 128,
    batch_size: int = 64,
    vocab_save_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """Build vocabulary and create PyTorch DataLoaders for all three splits."""
    if vocab is None:
        vocab = Vocabulary(max_size=max_vocab_size, min_freq=min_freq)
        print("Building vocabulary from training stories...")
        vocab.build_from_stories(train_stories)
        if vocab_save_path:
            vocab.save(vocab_save_path)
            print(f"Vocabulary saved to {vocab_save_path} (size: {len(vocab)})")

    train_ds = StoryLMDataset(train_stories, vocab, max_seq_len=max_seq_len)
    val_ds = StoryLMDataset(val_stories, vocab, max_seq_len=max_seq_len)
    test_ds = StoryLMDataset(test_stories, vocab, max_seq_len=max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, val_loader, test_loader, vocab
