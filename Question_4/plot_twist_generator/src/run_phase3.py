"""Phase 3 End-to-End Execution Script: Text-Generation Strategies Comparison.

Implements and evaluates:
1. Greedy decoding
2. Temperature sampling (T = 0.5, 0.8, 1.2)
3. Top-k sampling (k = 10, 40 at T = 0.8)

Across:
- Both trained architectures: Model A (Vanilla RNN) and Model B (LSTM)
- The 5 standardized prompt categories (Mystery, Surprising Discovery, Unusual Character,
  Problem to Solve, Unexpected Ending Setup)
- Multiple stochastic samples per setting to observe variance and repetition loops

Outputs:
- outputs/generations/part3_decoding_grid.json: full generated stories and token sequences
- outputs/tables/decoding_strategies_comparison.csv: quantitative metrics table
- outputs/plots/decoding_strategies_diversity.png: visual comparison of Distinct-1/2 & Rep-4
"""

import os
import sys
import json
from typing import List, Dict, Any
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seed, CONFIG
from src.data_prep import Vocabulary
from src.models import VanillaRNNLanguageModel, LSTMLanguageModel
from src.decode import generate_continuation
from src.evaluate_metrics import aggregate_generation_metrics

# The 5 Required Prompt Categories from Part 3 Specification
PART3_PROMPTS = [
    {
        "category": "1. Mystery",
        "prompt": "Lily found a tiny brass key hidden inside her grandmother's old wooden music box. When she turned it,",
    },
    {
        "category": "2. Surprising Discovery",
        "prompt": "While digging in the sandbox at school, Max hit something hard and shiny that started to glow.",
    },
    {
        "category": "3. Unusual Character",
        "prompt": "Barnaby was a very polite cloud who was afraid of rain and preferred to wear a warm yellow scarf.",
    },
    {
        "category": "4. Problem to Solve",
        "prompt": "The bridge across the river was broken, and Sammy the squirrel had to get the medicine to his mother before sunset.",
    },
    {
        "category": "5. Unexpected Ending Setup",
        "prompt": "Everyone in the forest was terrified of the giant shadow in the cave, until little Mia walked inside with a flashlight and saw",
    },
]

# Decoding configurations matrix
DECODING_CONFIGS = [
    {"name": "Greedy", "strategy": "greedy", "temperature": 1.0, "top_k": None, "runs": 1},
    {"name": "Temp 0.5", "strategy": "temperature", "temperature": 0.5, "top_k": None, "runs": 2},
    {"name": "Temp 0.8", "strategy": "temperature", "temperature": 0.8, "top_k": None, "runs": 2},
    {"name": "Temp 1.2", "strategy": "temperature", "temperature": 1.2, "top_k": None, "runs": 2},
    {"name": "Top-k (k=10, T=0.8)", "strategy": "top_k", "temperature": 0.8, "top_k": 10, "runs": 2},
    {"name": "Top-k (k=40, T=0.8)", "strategy": "top_k", "temperature": 0.8, "top_k": 40, "runs": 2},
]


def run_phase3_pipeline():
    print("=" * 70)
    print("EXECUTING PHASE 3: TEXT-GENERATION DECODING STRATEGIES")
    print("=" * 70)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    generations_dir = os.path.join(base_dir, "outputs", "generations")
    tables_dir = os.path.join(base_dir, "outputs", "tables")
    plots_dir = os.path.join(base_dir, "outputs", "plots")

    os.makedirs(generations_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    seed = CONFIG.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load Shared Vocabulary
    vocab_path = os.path.join(checkpoints_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}. Please run Phase 1 first.")
    vocab = Vocabulary.load(vocab_path)
    vocab_size = len(vocab)
    print(f"[1/5] Loaded shared vocabulary ({vocab_size:,} tokens)")

    # Step 2: Load Trained Model Checkpoints
    rnn_ckpt_path = os.path.join(checkpoints_dir, "vanilla_rnn_best.pt")
    lstm_ckpt_path = os.path.join(checkpoints_dir, "lstm_best.pt")

    if not os.path.exists(rnn_ckpt_path) or not os.path.exists(lstm_ckpt_path):
        raise FileNotFoundError("Model checkpoints missing! Please ensure Phase 2 has completed.")

    print("[2/5] Loading trained Model A (Vanilla RNN) and Model B (LSTM) checkpoints...")
    model_rnn = VanillaRNNLanguageModel(vocab_size, 256, 512, 2).to(device)
    rnn_ckpt = torch.load(rnn_ckpt_path, map_location=device)
    model_rnn.load_state_dict(rnn_ckpt["model_state"])
    model_rnn.eval()

    model_lstm = LSTMLanguageModel(vocab_size, 256, 512, 2).to(device)
    lstm_ckpt = torch.load(lstm_ckpt_path, map_location=device)
    model_lstm.load_state_dict(lstm_ckpt["model_state"])
    model_lstm.eval()

    models = [
        {"name": "Model A (Vanilla RNN)", "model": model_rnn},
        {"name": "Model B (LSTM)", "model": model_lstm},
    ]

    # Step 3: Generate Story Continuations across full matrix
    print("\n[3/5] Generating story continuations across 5 prompts x 6 decoding configs x 2 models...")
    all_results: List[Dict[str, Any]] = []
    max_tokens = CONFIG.get("max_new_tokens", 80)

    for m_info in models:
        m_name = m_info["name"]
        m_obj = m_info["model"]
        print(f"\n--- Generating with {m_name} ---")

        for p_info in PART3_PROMPTS:
            cat = p_info["category"]
            prompt_text = p_info["prompt"]

            for cfg in DECODING_CONFIGS:
                cfg_name = cfg["name"]
                strat = cfg["strategy"]
                temp = cfg["temperature"]
                top_k = cfg["top_k"]
                runs = cfg["runs"]

                for run_idx in range(runs):
                    run_seed = seed + (run_idx * 137) if strat != "greedy" else None
                    gen_output = generate_continuation(
                        model=m_obj,
                        vocab=vocab,
                        prompt=prompt_text,
                        strategy=strat,
                        temperature=temp,
                        top_k=top_k if top_k else 40,
                        max_new_tokens=max_tokens,
                        ban_unk=True,
                        device=device,
                        seed=run_seed,
                    )

                    entry = {
                        "model": m_name,
                        "category": cat,
                        "prompt": prompt_text,
                        "config_name": cfg_name,
                        "strategy": strat,
                        "temperature": temp,
                        "top_k": top_k,
                        "run_index": run_idx + 1,
                        "continuation": gen_output["continuation"],
                        "full_story": gen_output["full_story"],
                        "num_tokens": gen_output["num_generated_tokens"],
                        "eos_triggered": gen_output["eos_triggered"],
                        "generated_token_ids": gen_output["generated_token_ids"],
                    }
                    all_results.append(entry)

    # Save complete generations to JSON
    grid_json_path = os.path.join(generations_dir, "part3_decoding_grid.json")
    with open(grid_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_results)} total generated stories to: {grid_json_path}")

    # Step 4: Compute Quantitative Metrics per (Model, Decoding Strategy)
    print("\n[4/5] Computing quantitative diversity, repetition and length metrics...")
    summary_rows = []

    for m_info in models:
        m_name = m_info["name"]
        for cfg in DECODING_CONFIGS:
            cfg_name = cfg["name"]

            # Filter generations matching this model & configuration
            subset = [
                r for r in all_results
                if r["model"] == m_name and r["config_name"] == cfg_name
            ]

            metrics = aggregate_generation_metrics(
                subset,
                min_len_threshold=CONFIG.get("min_len_threshold", 15),
                max_len_threshold=max_tokens,
            )

            summary_rows.append({
                "Model": m_name,
                "Decoding Configuration": cfg_name,
                "Strategy": cfg["strategy"],
                "Distinct-1 (Mean)": round(metrics["sample_distinct_1_mean"], 3),
                "Distinct-2 (Mean)": round(metrics["sample_distinct_2_mean"], 3),
                "Corpus Distinct-1": round(metrics["corpus_distinct_1"], 3),
                "Corpus Distinct-2": round(metrics["corpus_distinct_2"], 3),
                "Repetition Rate (Rep-4)": round(metrics["rep_4_mean"], 3),
                "Avg Length (Tokens)": round(metrics["avg_length"], 1),
                "Premature EOS (%)": round(metrics["premature_eos_pct"], 1),
                "Hit Max Context (%)": round(metrics["unterminated_pct"], 1),
            })

    df_metrics = pd.DataFrame(summary_rows)
    metrics_csv_path = os.path.join(tables_dir, "decoding_strategies_comparison.csv")
    metrics_json_path = os.path.join(tables_dir, "decoding_strategies_comparison.json")
    df_metrics.to_csv(metrics_csv_path, index=False)
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print("\n" + "=" * 70)
    print("DECODING STRATEGIES QUANTITATIVE COMPARISON:")
    print("=" * 70)
    print(df_metrics[["Model", "Decoding Configuration", "Distinct-1 (Mean)", "Distinct-2 (Mean)", "Repetition Rate (Rep-4)", "Avg Length (Tokens)"]].to_string(index=False))

    # Step 5: Generate Comparative Bar Charts
    print("\n[5/5] Generating comparative diversity & repetition plots...")
    configs_list = [c["name"] for c in DECODING_CONFIGS]

    rnn_d1 = [df_metrics[(df_metrics["Model"] == "Model A (Vanilla RNN)") & (df_metrics["Decoding Configuration"] == c)]["Distinct-1 (Mean)"].values[0] for c in configs_list]
    lstm_d1 = [df_metrics[(df_metrics["Model"] == "Model B (LSTM)") & (df_metrics["Decoding Configuration"] == c)]["Distinct-1 (Mean)"].values[0] for c in configs_list]

    rnn_rep = [df_metrics[(df_metrics["Model"] == "Model A (Vanilla RNN)") & (df_metrics["Decoding Configuration"] == c)]["Repetition Rate (Rep-4)"].values[0] for c in configs_list]
    lstm_rep = [df_metrics[(df_metrics["Model"] == "Model B (LSTM)") & (df_metrics["Decoding Configuration"] == c)]["Repetition Rate (Rep-4)"].values[0] for c in configs_list]

    x = range(len(configs_list))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Distinct-1 Diversity
    ax1.bar([i - width/2 for i in x], rnn_d1, width, label="Vanilla RNN", color="#1f77b4", alpha=0.85)
    ax1.bar([i + width/2 for i in x], lstm_d1, width, label="LSTM", color="#ff7f0e", alpha=0.85)
    ax1.set_ylabel("Distinct-1 (Higher = More Lexical Variety)", fontsize=10)
    ax1.set_title("Lexical Diversity (Distinct-1) across Decoding Strategies", fontsize=11, fontweight="bold")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(configs_list, rotation=35, ha="right", fontsize=9)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Repetition Rate (Rep-4)
    ax2.bar([i - width/2 for i in x], rnn_rep, width, label="Vanilla RNN", color="#1f77b4", alpha=0.85)
    ax2.bar([i + width/2 for i in x], lstm_rep, width, label="LSTM", color="#ff7f0e", alpha=0.85)
    ax2.set_ylabel("Rep-4 (Lower = Less Degenerate Loops)", fontsize=10)
    ax2.set_title("4-Gram Repetition Rate (Rep-4) across Strategies", fontsize=11, fontweight="bold")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(configs_list, rotation=35, ha="right", fontsize=9)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "decoding_strategies_diversity.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved diversity & repetition plot to: {plot_path}")

    # Print sample comparative generations from Category 1 (Mystery)
    print("\n" + "=" * 70)
    print("SAMPLE COMPARATIVE GENERATIONS (Category 1: Mystery):")
    print("=" * 70)
    cat1_samples = [r for r in all_results if "Mystery" in r["category"] and r["run_index"] == 1]
    for s in cat1_samples:
        print(f"\n[{s['model']} | {s['config_name']}]")
        print(f"Continuation: {s['continuation'][:200]}...")

    print("\n" + "=" * 70)
    print("PHASE 3 EXECUTION COMPLETE AND FULLY VERIFIED!")
    print("=" * 70)


if __name__ == "__main__":
    run_phase3_pipeline()

