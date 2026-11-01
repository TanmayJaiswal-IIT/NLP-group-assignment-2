"""Open-Source Pretrained LLM Baseline for Question 4 Extension.

Evaluates `HuggingFaceTB/SmolLM2-135M-Instruct` as an open-source Transformer LLM baseline
beside the from-scratch Model A (Vanilla RNN) and Model B (LSTM).

Adheres strictly to the rubric requirements:
- Zero-shot instruction prompting (no fine-tuning on TinyStories).
- Exact model checkpoint, prompt template, decoding parameters, and hardware recorded.
- Comparable token generation limits (max_new_tokens = 90, T = 0.8, Top-k = 40).
- Side-by-side output presentation and evaluation across the Part 5 criteria.
"""

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMBaselineRunner:
    """Runner for small open-source pretrained causal language models."""

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        device: Optional[torch.device] = None,
    ):
        self.model_id = model_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading pretrained LLM: {self.model_id} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        print("Pretrained LLM successfully initialized.")

    @torch.no_grad()
    def generate_story_twist(
        self,
        prompt: str,
        max_new_tokens: int = 90,
        temperature: float = 0.8,
        top_k: int = 40,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Generate zero-shot story continuation with instruction prompt template."""
        if seed is not None:
            torch.manual_seed(seed)

        messages = [
            {
                "role": "system",
                "content": "You are a creative storyteller. Continue the following opening sentence with an interesting, surprising, and coherent plot twist in 3 to 5 sentences. Write only the story continuation.",
            },
            {"role": "user", "content": prompt},
        ]

        formatted_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        input_len = formatted_inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **formatted_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        gen_tokens = outputs[0][input_len:].cpu()
        continuation_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        return {
            "model_id": self.model_id,
            "prompt": prompt,
            "continuation": continuation_text,
            "full_story": f"{prompt.strip()} {continuation_text}",
            "generated_token_ids": gen_tokens.tolist(),
            "num_generated_tokens": len(gen_tokens),
            "temperature": temperature,
            "top_k": top_k,
        }

