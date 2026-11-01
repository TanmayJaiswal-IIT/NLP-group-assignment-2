"""Neural Language Models for Question 4: The Plot-Twist Generator.

Implements:
1. Model A: Vanilla RNN Language Model
   Architecture: Embedding -> Vanilla RNN (tanh) -> Linear -> Vocabulary Logits
2. Model B: LSTM Language Model
   Architecture: Embedding -> LSTM (gated cell state) -> Linear -> Vocabulary Logits

Both models share identical embedding dimension, hidden dimension, number of layers,
dropout, vocabulary, and loss formulation to ensure rigorous, fair comparison as required
by the DSL 504 rubric.
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BaseRecurrentLanguageModel(nn.Module):
    """Base class for autoregressive recurrent language models."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_id: int = 0,
        model_type: str = "rnn",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pad_id = pad_id
        self.model_type = model_type.lower()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)

        if self.model_type == "rnn":
            self.recurrent = nn.RNN(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                nonlinearity="tanh",
            )
        elif self.model_type == "lstm":
            self.recurrent = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Expected 'rnn' or 'lstm'.")

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hidden=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input token IDs of shape (batch_size, seq_len)
            lengths: Optional 1D tensor of valid sequence lengths before padding
            hidden: Previous recurrent state (h for RNN, (h, c) for LSTM)
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden: Updated recurrent state
        """
        emb = self.drop(self.embedding(x))

        if lengths is not None:
            # Pack sequence to bypass computation over padding tokens
            packed = pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, hidden = self.recurrent(packed, hidden)
            out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=x.size(1)
            )
        else:
            out, hidden = self.recurrent(emb, hidden)

        logits = self.fc(self.drop(out))
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize zero recurrent state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        if self.model_type == "lstm":
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h0, c0)
        return h0


class VanillaRNNLanguageModel(BaseRecurrentLanguageModel):
    """Model A: Standard Elman Vanilla RNN Language Model.
    
    Update equations:
        h_t = tanh(W_ih * e_t + b_ih + W_hh * h_{t-1} + b_hh)
        y_hat_t = W_out * h_t + b_out
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_id: int = 0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=pad_id,
            model_type="rnn",
        )


class LSTMLanguageModel(BaseRecurrentLanguageModel):
    """Model B: Long Short-Term Memory (LSTM) Language Model.
    
    Update equations:
        i_t = sigmoid(W_ii * e_t + b_ii + W_hi * h_{t-1} + b_hi)   (Input Gate)
        f_t = sigmoid(W_if * e_t + b_if + W_hf * h_{t-1} + b_hf)   (Forget Gate)
        g_t = tanh(W_ig * e_t + b_ig + W_hg * h_{t-1} + b_hg)      (Candidate Cell)
        o_t = sigmoid(W_io * e_t + b_io + W_ho * h_{t-1} + b_ho)   (Output Gate)
        c_t = f_t * c_{t-1} + i_t * g_t                           (Additive Cell State Highway)
        h_t = o_t * tanh(c_t)
        y_hat_t = W_out * h_t + b_out
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_id: int = 0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=pad_id,
            model_type="lstm",
        )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Breakdown of trainable parameter counts by module type."""
    emb_params = sum(p.numel() for n, p in model.named_parameters() if "embedding" in n and p.requires_grad)
    rec_params = sum(p.numel() for n, p in model.named_parameters() if "recurrent" in n and p.requires_grad)
    fc_params = sum(p.numel() for n, p in model.named_parameters() if "fc" in n and p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "embedding_params": emb_params,
        "recurrent_params": rec_params,
        "linear_params": fc_params,
        "total_params": total_params,
    }

