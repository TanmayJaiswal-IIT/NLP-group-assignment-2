"""Training pipeline and quantitative evaluation for Question 4: The Plot-Twist Generator.

Implements:
- Cross-entropy training with padding token masking (ignore_index=0).
- True corpus-level Perplexity calculation strictly on valid non-padded tokens.
- AdamW optimizer, gradient clipping (max norm = 1.0), and ReduceLROnPlateau scheduler.
- Early stopping with best checkpoint restoration.
- Wall-clock training duration and parameter tracking.
- Comparative plotting and table generation for Model A (Vanilla RNN) vs Model B (LSTM).
"""

import os
import sys
import time
import math
import json
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure local modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import VanillaRNNLanguageModel, LSTMLanguageModel, count_parameters
from src.data_prep import SPECIAL_TOKENS, Vocabulary


def evaluate_loss_and_ppl(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    pad_id: int,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate cross-entropy loss and perplexity strictly over valid non-padded tokens.
    
    PPL = exp( sum(CE_loss) / total_valid_tokens )
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y, lengths in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x, lengths)
            vocab_size = logits.size(-1)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            non_pad = (y != pad_id).sum().item()
            total_loss += loss.item()
            total_tokens += non_pad

    avg_loss = total_loss / max(1, total_tokens)
    # Clip exponent to avoid float overflow during early epochs
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def train_single_model(
    model: nn.Module,
    model_name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    save_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Train a single language model with early stopping and learning rate scheduling."""
    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    pad_id = SPECIAL_TOKENS["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-5))
    grad_clip = float(config.get("grad_clip", 1.0))
    epochs = int(config.get("epochs", 15))
    patience = int(config.get("patience", 2))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-5
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "epoch_time_sec": [],
        "learning_rates": [],
    }

    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    best_epoch = 0
    patience_counter = 0
    total_train_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_tokens = 0

        for batch_idx, (x, y, lengths) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits, _ = model(x, lengths)
            vocab_size = logits.size(-1)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            non_pad = (y != pad_id).sum().item()

            # Backward pass scaled by valid tokens in mini-batch
            (loss / non_pad).backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item()
            running_tokens += non_pad

        epoch_time = time.time() - t0
        train_loss = running_loss / max(1, running_tokens)
        val_loss, val_ppl = evaluate_loss_and_ppl(model, val_loader, pad_id, device)
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["epoch_time_sec"].append(epoch_time)
        history["learning_rates"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f} | "
            f"LR: {current_lr:.1e} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl = val_ppl
            best_epoch = epoch
            patience_counter = 0
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_ppl": best_val_ppl,
                },
                save_path,
            )
            print(f"  --> Saved new best checkpoint to {save_path} (Val PPL: {val_ppl:.2f})")
        else:
            patience_counter += 1
            print(f"  --> No improvement for {patience_counter}/{patience} epoch(s).")
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch}.")
                break

    total_duration = time.time() - total_train_start

    # Restore best weights for final evaluation on test split
    print(f"Restoring best checkpoint from epoch {best_epoch} for test evaluation...")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_ppl = evaluate_loss_and_ppl(model, test_loader, pad_id, device)
    print(f"Held-out Test Evaluation -> Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:.2f}")

    return {
        "model_name": model_name,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_ppl": best_val_ppl,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "total_train_time_sec": total_duration,
        "avg_epoch_time_sec": float(sum(history["epoch_time_sec"]) / max(1, len(history["epoch_time_sec"]))),
    }


def plot_training_curves(rnn_results: Dict, lstm_results: Dict, output_path: str):
    """Plot side-by-side training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curves
    epochs_rnn = range(1, len(rnn_results["history"]["train_loss"]) + 1)
    epochs_lstm = range(1, len(lstm_results["history"]["train_loss"]) + 1)

    ax1.plot(epochs_rnn, rnn_results["history"]["train_loss"], "b--", label="Vanilla RNN (Train)")
    ax1.plot(epochs_rnn, rnn_results["history"]["val_loss"], "b-", linewidth=2, label="Vanilla RNN (Val)")
    ax1.plot(epochs_lstm, lstm_results["history"]["train_loss"], "r--", label="LSTM (Train)")
    ax1.plot(epochs_lstm, lstm_results["history"]["val_loss"], "r-", linewidth=2, label="LSTM (Val)")

    ax1.set_title("Training & Validation Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss (Sum / Valid Tokens)", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(alpha=0.3)

    # Perplexity Curves
    ax2.plot(epochs_rnn, rnn_results["history"]["val_ppl"], "b-o", linewidth=2, label="Vanilla RNN (Val PPL)")
    ax2.plot(epochs_lstm, lstm_results["history"]["val_ppl"], "r-s", linewidth=2, label="LSTM (Val PPL)")

    ax2.set_title("Validation Perplexity Over Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Perplexity (Lower is Better)", fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved training curves to: {output_path}")

