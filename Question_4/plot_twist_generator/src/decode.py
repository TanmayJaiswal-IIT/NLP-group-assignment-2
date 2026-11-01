"""Decoding strategies and story continuation generation for Question 4.

Implements the three core generation strategies required by Part 3:
1. Greedy Decoding: argmax over vocabulary logits at each step (deterministic).
2. Temperature Sampling: scaling logits by temperature T in {0.5, 0.8, 1.2} before softmax.
3. Top-k Sampling: filtering to the top-k highest probability tokens (e.g. k=10, k=40)
   before sampling.

Handles:
- Conditioning on an arbitrary opening prompt (hidden-state warm-up).
- Strict masking of <PAD> and <BOS> tokens to prevent degenerate generation.
- Optional <UNK> suppression.
- Autoregressive step-by-step unrolling with recurrent state caching.
- Early termination upon generating <EOS>.
- Clean detokenization and spacing restoration.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F

from src.data_prep import Vocabulary, SPECIAL_TOKENS


@torch.no_grad()
def generate_continuation(
    model: torch.nn.Module,
    vocab: Vocabulary,
    prompt: str,
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_k: int = 40,
    max_new_tokens: int = 100,
    ban_unk: bool = True,
    device: torch.device = torch.device("cpu"),
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate a single continuation for a given prompt using the specified decoding strategy.
    
    Args:
        model: Trained VanillaRNNLanguageModel or LSTMLanguageModel
        vocab: Shared Vocabulary object
        prompt: Opening story string
        strategy: One of 'greedy', 'temperature', 'top_k'
        temperature: Sampling temperature (e.g. 0.5, 0.8, 1.2)
        top_k: Truncation threshold for top-k sampling (e.g. 10, 40)
        max_new_tokens: Maximum continuation length
        ban_unk: Whether to prevent generation of <UNK> tokens
        device: CPU or CUDA device
        seed: Optional RNG seed for deterministic reproduction of stochastic sampling
        
    Returns:
        dict containing prompt, continuation, full_story, token IDs, length, and EOS status.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()

    # Encode prompt with <BOS> but WITHOUT <EOS> so the model continues the story
    prompt_ids = vocab.encode(prompt, add_bos=True, add_eos=False)
    if len(prompt_ids) == 0:
        prompt_ids = [SPECIAL_TOKENS["<BOS>"]]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Warm up recurrent hidden state on the prompt prefix
    logits, hidden = model(input_ids)
    next_logits = logits[:, -1, :]  # Shape: (1, vocab_size)

    pad_id = SPECIAL_TOKENS["<PAD>"]
    unk_id = SPECIAL_TOKENS["<UNK>"]
    bos_id = SPECIAL_TOKENS["<BOS>"]
    eos_id = SPECIAL_TOKENS["<EOS>"]

    generated_ids: List[int] = []
    eos_triggered = False

    for _ in range(max_new_tokens):
        step_logits = next_logits.clone()

        # Prevent emitting <PAD> or <BOS>
        step_logits[:, pad_id] = -float("inf")
        step_logits[:, bos_id] = -float("inf")
        if ban_unk:
            step_logits[:, unk_id] = -float("inf")

        if strategy == "greedy":
            next_token = torch.argmax(step_logits, dim=-1, keepdim=True)

        elif strategy == "temperature":
            t = max(temperature, 1e-4)
            scaled_logits = step_logits / t
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        elif strategy == "top_k":
            t = max(temperature, 1e-4)
            scaled_logits = step_logits / t
            k = min(top_k, scaled_logits.size(-1))
            top_vals, top_indices = torch.topk(scaled_logits, k=k, dim=-1)
            filtered_logits = torch.full_like(scaled_logits, -float("inf"))
            filtered_logits.scatter_(1, top_indices, top_vals)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        else:
            raise ValueError(f"Unknown strategy: '{strategy}'. Choose 'greedy', 'temperature', or 'top_k'.")

        token_id = next_token.item()
        if token_id == eos_id:
            eos_triggered = True
            break

        generated_ids.append(token_id)

        # Autoregressive single-step forward pass
        logits, hidden = model(next_token, hidden=hidden)
        next_logits = logits[:, -1, :]

    continuation_text = vocab.decode(generated_ids, stop_at_eos=True)
    clean_prompt = prompt.strip()
    full_story = f"{clean_prompt} {continuation_text}".strip()

    return {
        "prompt": prompt,
        "strategy": strategy,
        "temperature": temperature if strategy in ("temperature", "top_k") else 1.0,
        "top_k": top_k if strategy == "top_k" else None,
        "continuation": continuation_text,
        "full_story": full_story,
        "generated_token_ids": generated_ids,
        "num_generated_tokens": len(generated_ids),
        "eos_triggered": eos_triggered,
    }


def generate_multiple_continuations(
    model: torch.nn.Module,
    vocab: Vocabulary,
    prompt: str,
    strategy: str = "temperature",
    temperature: float = 0.8,
    top_k: int = 40,
    num_samples: int = 3,
    max_new_tokens: int = 100,
    ban_unk: bool = True,
    device: torch.device = torch.device("cpu"),
    base_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate multiple stochastic continuations for the same prompt under different random seeds."""
    samples = []
    for i in range(num_samples):
        # Vary seed across samples for stochastic diversity
        sample_seed = base_seed + (i * 100) if strategy != "greedy" else None
        res = generate_continuation(
            model=model,
            vocab=vocab,
            prompt=prompt,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            ban_unk=ban_unk,
            device=device,
            seed=sample_seed,
        )
        res["sample_index"] = i + 1
        samples.append(res)
        if strategy == "greedy":
            # Greedy is completely deterministic; 1 sample is sufficient
            break
    return samples

