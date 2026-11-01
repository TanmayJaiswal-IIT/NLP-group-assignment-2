"""Phase 1 End-to-End Execution Script: Dataset Preparation & Next-Token Prediction.

Executes:
1. Fixed seed initialization (seed=42).
2. Streaming and story-level splitting of TinyStories (20,000 train, 1,000 val, 1,000 test).
3. Local disk caching for instant subsequent loading.
4. Shared vocabulary construction with special tokens (<PAD>, <UNK>, <BOS>, <EOS>).
5. Vocabulary export to `checkpoints/vocab.json`.
6. Dataset & DataLoader construction for all splits with right-padding.
7. Verification of next-token prediction alignment (y[t] = x[t+1]) and padding integrity.
8. Calculation of OOV rates, story lengths, token distributions, and chunk counts.
9. Generation of exploratory distribution plots in `outputs/plots/`.
10. Export of dataset summary statistics to `outputs/tables/`.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure local modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seed, CONFIG
from src.data_prep import (
    load_and_split_tinystories,
    clean_and_tokenize,
    Vocabulary,
    StoryLMDataset,
    get_dataloaders,
    SPECIAL_TOKENS,
)


def run_phase1_pipeline():
    print("=" * 70)
    print("EXECUTING PHASE 1: DATA PREPARATION & NEXT-TOKEN PREDICTION")
    print("=" * 70)

    # Step 1: Set seed for full determinism
    seed = CONFIG.get("seed", 42)
    set_seed(seed)
    print(f"[1/7] Initialized random seed: {seed}")

    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    plots_dir = os.path.join(base_dir, "outputs", "plots")
    tables_dir = os.path.join(base_dir, "outputs", "tables")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    vocab_file = os.path.join(checkpoints_dir, "vocab.json")

    # Step 2: Load & split stories
    num_train = CONFIG.get("train_stories", 20000)
    num_val = CONFIG.get("val_stories", 1000)
    num_test = CONFIG.get("test_stories", 1000)

    print(f"\n[2/7] Loading and splitting stories ({num_train} train, {num_val} val, {num_test} test)...")
    train_stories, val_stories, test_stories = load_and_split_tinystories(
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        seed=seed,
        cache_dir=data_dir,
    )

    # Step 3: Vocabulary Construction on Training Split
    max_vocab_size = CONFIG.get("max_vocab_size", 10000)
    min_freq = CONFIG.get("min_freq", 2)
    print(f"\n[3/7] Building shared vocabulary (max_size={max_vocab_size}, min_freq={min_freq})...")

    vocab = Vocabulary(max_size=max_vocab_size, min_freq=min_freq)
    vocab.build_from_stories(train_stories)
    vocab.save(vocab_file)
    print(f"Vocabulary successfully built and saved to: {vocab_file}")
    print(f"  - Total unique words encountered in train split: {vocab.raw_vocab_size:,}")
    print(f"  - Final vocabulary size (including special tokens): {len(vocab):,}")

    # Step 4: Token Length & OOV Statistics
    print("\n[4/7] Computing dataset statistics and OOV rates...")
    train_lens = [len(clean_and_tokenize(s)) for s in train_stories]
    val_lens = [len(clean_and_tokenize(s)) for s in val_stories]
    test_lens = [len(clean_and_tokenize(s)) for s in test_stories]

    val_oov_rate, val_oov_count, val_total_toks = vocab.compute_oov_rate(val_stories)
    test_oov_rate, test_oov_count, test_total_toks = vocab.compute_oov_rate(test_stories)

    stats = {
        "num_train_stories": len(train_stories),
        "num_val_stories": len(val_stories),
        "num_test_stories": len(test_stories),
        "vocab_size": len(vocab),
        "raw_vocab_size": vocab.raw_vocab_size,
        "min_freq": min_freq,
        "special_tokens": SPECIAL_TOKENS,
        "story_length_tokens": {
            "train": {
                "min": int(np.min(train_lens)),
                "max": int(np.max(train_lens)),
                "mean": float(np.mean(train_lens)),
                "median": float(np.median(train_lens)),
                "p90": float(np.percentile(train_lens, 90)),
                "total_tokens": int(np.sum(train_lens)),
            },
            "val": {
                "min": int(np.min(val_lens)),
                "max": int(np.max(val_lens)),
                "mean": float(np.mean(val_lens)),
                "median": float(np.median(val_lens)),
                "p90": float(np.percentile(val_lens, 90)),
                "total_tokens": int(np.sum(val_lens)),
            },
            "test": {
                "min": int(np.min(test_lens)),
                "max": int(np.max(test_lens)),
                "mean": float(np.mean(test_lens)),
                "median": float(np.median(test_lens)),
                "p90": float(np.percentile(test_lens, 90)),
                "total_tokens": int(np.sum(test_lens)),
            },
        },
        "oov_evaluation": {
            "val_oov_rate_pct": float(val_oov_rate),
            "val_oov_tokens": val_oov_count,
            "val_total_tokens": val_total_toks,
            "test_oov_rate_pct": float(test_oov_rate),
            "test_oov_tokens": test_oov_count,
            "test_total_tokens": test_total_toks,
        },
    }

    print(f"  - Train story length: Mean={stats['story_length_tokens']['train']['mean']:.1f}, "
          f"Median={stats['story_length_tokens']['train']['median']:.1f}, "
          f"90th%={stats['story_length_tokens']['train']['p90']:.1f}, "
          f"Max={stats['story_length_tokens']['train']['max']}")
    print(f"  - Val OOV Rate: {val_oov_rate:.2f}% ({val_oov_count}/{val_total_toks} tokens)")
    print(f"  - Test OOV Rate: {test_oov_rate:.2f}% ({test_oov_count}/{test_total_toks} tokens)")

    # Step 5: Dataset & DataLoader Construction
    max_seq_len = CONFIG.get("max_seq_len", 128)
    batch_size = CONFIG.get("batch_size", 64)
    print(f"\n[5/7] Creating StoryLMDataset & DataLoaders (max_seq_len={max_seq_len}, batch_size={batch_size})...")

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        train_stories=train_stories,
        val_stories=val_stories,
        test_stories=test_stories,
        vocab=vocab,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )

    stats["num_train_chunks"] = len(train_loader.dataset)
    stats["num_val_chunks"] = len(val_loader.dataset)
    stats["num_test_chunks"] = len(test_loader.dataset)

    print(f"  - Train chunks (sequences): {stats['num_train_chunks']:,} across {len(train_loader)} batches")
    print(f"  - Val chunks (sequences):   {stats['num_val_chunks']:,} across {len(val_loader)} batches")
    print(f"  - Test chunks (sequences):  {stats['num_test_chunks']:,} across {len(test_loader)} batches")

    # Step 6: Next-Token Prediction & Padding Sanity Checks
    print("\n[6/7] Verifying next-token prediction alignment & padding mechanics...")
    sample_x, sample_y, lengths = next(iter(train_loader))

    assert sample_x.shape == sample_y.shape, "Shape mismatch between inputs and targets!"
    assert sample_x.size(0) <= batch_size, "Batch size exceeds configured limit!"
    
    # Verify next-token offset on non-padded slice
    first_x = sample_x[0]
    first_y = sample_y[0]
    first_len = lengths[0].item()

    # The target at time t must equal input at time t+1 for all valid steps
    if first_len > 1:
        match_count = (first_x[1:first_len] == first_y[:first_len - 1]).sum().item()
        assert match_count == first_len - 1, "Input-Target alignment broken!"
        print("  [PASSED] Target is exactly input shifted by 1 token (y[t] == x[t+1])")

    # Verify padding
    pad_id = SPECIAL_TOKENS["<PAD>"]
    if sample_x.size(1) > first_len:
        assert (first_x[first_len:] == pad_id).all(), "Padding position mismatch in x!"
        assert (first_y[first_len:] == pad_id).all(), "Padding position mismatch in y!"
        print(f"  [PASSED] All positions >= length are properly padded with <PAD> (ID={pad_id})")

    # Example tokenization walkthrough
    sample_story = train_stories[0][:200]
    tokenized = clean_and_tokenize(sample_story)
    encoded = vocab.encode(sample_story, add_bos=True, add_eos=True)
    decoded = vocab.decode(encoded, stop_at_eos=True)

    print("\n--- Example Story Walkthrough ---")
    print(f"Raw Snippet:     {sample_story[:100]}...")
    print(f"Tokenized:       {tokenized[:12]}...")
    print(f"Encoded IDs:     {encoded[:12]}... (starts with BOS={SPECIAL_TOKENS['<BOS>']})")
    print(f"Input x[:5]:     {encoded[:5]}")
    print(f"Target y[:5]:    {encoded[1:6]}")
    print(f"Decoded:         {decoded[:100]}...")

    # Step 7: Export Summary Tables and Exploratory Plots
    print("\n[7/7] Generating distribution plots and exporting metadata...")

    # Save JSON summary
    summary_json_path = os.path.join(tables_dir, "phase1_dataset_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Summary statistics saved to: {summary_json_path}")

    # Plot 1: Story Length Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(train_lens, bins=40, color="#1f77b4", alpha=0.75, edgecolor="black", label="Train Stories")
    ax.axvline(stats["story_length_tokens"]["train"]["median"], color="red", linestyle="--", linewidth=1.5,
               label=f"Median ({stats['story_length_tokens']['train']['median']:.0f})")
    ax.axvline(stats["story_length_tokens"]["train"]["p90"], color="orange", linestyle=":", linewidth=1.5,
               label=f"90th Percentile ({stats['story_length_tokens']['train']['p90']:.0f})")
    ax.axvline(max_seq_len, color="green", linestyle="-.", linewidth=1.5,
               label=f"Max Seq Len ({max_seq_len})")
    ax.set_title("Story Length Distribution (in Tokens)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Tokens per Story", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    length_plot_path = os.path.join(plots_dir, "story_length_distribution.png")
    fig.savefig(length_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {length_plot_path}")

    # Plot 2: Top Vocabulary Word Frequencies
    top_words = sorted(
        [(k, v) for k, v in vocab.token_counts.items() if k not in SPECIAL_TOKENS],
        key=lambda x: -x[1]
    )[:25]
    words, counts = zip(*top_words)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(words)), counts, color="#2ca02c", edgecolor="black", alpha=0.8)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right", fontsize=9)
    ax.set_title("Top 25 Most Frequent Tokens in Training Corpus", fontsize=13, fontweight="bold")
    ax.set_xlabel("Token", fontsize=11)
    ax.set_ylabel("Frequency Count", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    vocab_plot_path = os.path.join(plots_dir, "top_vocab_frequencies.png")
    fig.savefig(vocab_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {vocab_plot_path}")

    print("\n" + "=" * 70)
    print("PHASE 1 EXECUTION COMPLETE AND FULLY VERIFIED!")
    print("=" * 70)


if __name__ == "__main__":
    run_phase1_pipeline()

