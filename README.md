# NLP Group Assignment 2 — Group 14

**Group 14 | DSL504 Natural Language Processing, IIT Bhilai | Group Assignment 2**  
Kinshuk Gupta (12341190), Harshit Kandpal (B24CS505), Utkarsh Chattarjee (B24CS050), Sujal Som (B24DS032), Tanmay Jaiswal (B24DS033)

---

## Summary

This repository contains four complete NLP experiments: (Q1) English→Hindi neural machine translation with Bahdanau attention on the IITB corpus; (Q2) five-class Amazon review sentiment classification with BiGRU+GloVe vs fine-tuned RoBERTa; (Q3) abstractive news summarisation with a from-scratch Transformer on CNN/DailyMail; and (Q4) creative story and plot-twist generation with Vanilla RNN vs LSTM trained on TinyStories.

---

## Repository Structure

```
NLP-group-assignment-2/
├── README.md                                  ← This file
├── requirements.txt                           ← Unified environment
├── make_submission.sh                         ← Creates Group14_Submission.zip
├── report/
│   ├── extract_facts.py                       ← Reads results JSON → facts.json
│   ├── facts.json                             ← All numbers used in reports
│   ├── generate_pdf.py                        ← Builds combined PDF report
│   ├── sections/
│   │   ├── q1.md  q2.md  q3.md  q4.md        ← Shared prose sections
│   └── Group14_Combined_Technical_Report.pdf  ← Combined report (auto-generated)
├── Question1/                                 ← Q1: EN→HI NMT (IITB corpus)
│   ├── question_1.ipynb
│   ├── scratch/results.json
│   ├── figures/
│   ├── checkpoints/                           ← baseline_model.pt (44 MB), attention_model.pt (73 MB)
│   ├── README.md
│   └── NLP_Group_Assignment_2.pdf            ← Assignment spec
├── Question_2/                                ← Q2: Amazon sentiment (BiGRU vs RoBERTa)
│   ├── Question_2.ipynb
│   ├── report.md
│   └── README.md
├── Question_3/                                ← Q3: CNN/DailyMail summarisation
│   ├── summarizer.ipynb
│   ├── summarizer_export.py
│   ├── results.json  predictions.json
│   ├── figures/
│   ├── REPORT.docx
│   └── README.md
└── Question_4/plot_twist_generator/           ← Q4: TinyStories RNN vs LSTM
    ├── configs/config.yaml
    ├── src/                                   ← Modular pipeline scripts
    ├── notebooks/                             ← Interactive notebooks
    ├── outputs/                               ← Tables, plots, generated stories
    ├── checkpoints/vocab.json                 ← Shared vocabulary (176 KB)
    ├── requirements.txt
    ├── report/technical_report.md
    └── README.md
```

---

## Global Setup

**Python version:** 3.10+ (tested on 3.12.3)

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. GPU recommended for Q1, Q2, Q3 (CUDA); Q4 is feasible on CPU with subset
```

**HuggingFace datasets** (Q2, Q3, Q4) are auto-downloaded on first run to `~/.cache/huggingface/`.  
**GloVe** (Q2): download `glove.6B.100d.txt` from [nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) and set the path in the Q2 notebook.  
**Colab/Kaggle:** Each notebook contains instructions for mounting Google Drive or Kaggle datasets. All notebooks were originally run on T4/A100 GPUs.

---

## Question 1 — English → Hindi NMT (IITB Corpus)

**Task:** Translate English sentences to Hindi using the IITB parallel corpus. Compare a baseline LSTM encoder-decoder with a Bahdanau attention model.

**Commands (in order):**
```bash
# Install Q1 deps (covered by root requirements.txt)
# Run notebook end-to-end (Colab/Kaggle recommended for GPU):
#   Question1/question_1.ipynb → Run All
# CPU quick-test: uncomment QUICK_TEST = True in config cell (500 pairs)
```

**Runtime/Hardware:** ~2–3 hours on T4 GPU (8 epochs × 35,000 pairs). CPU: ~8–12 h or use QUICK_TEST.

**Output locations:** `Question1/scratch/results.json`, `Question1/figures/`, `Question1/checkpoints/`

### Key Results

| Model | BLEU |
|---|---:|
| Baseline LSTM | 2.44 |
| Attention LSTM | 3.38 |
| Delta | +0.94 |

**Length-bucket performance:**

| Bucket | BLEU Baseline | BLEU Attention |
|---|---:|---:|
| 1–5 tokens | 2.91 | 4.20 |
| 6–10 tokens | 2.27 | 3.38 |
| 11–15 tokens | 5.88 | 8.85 |
| 16+ tokens | 0.89 | 1.27 |

**Sample translation:**
- **English:** `the tea party is aghast .`
- **Reference:** `चाय पार्टी भौचक्की है।`
- **Attention pred:** `टी एच सी . . . टी .` (OOV collapse on "tea party")

**Conclusion:** Bahdanau attention improves BLEU by 0.94 points (38.5% relative). Gains are largest in the 11–15-token bucket. Both models struggle on 16+ token sentences due to vocabulary sparsity.

**Checkpoints:** `Question1/checkpoints/baseline_model.pt` (44 MB), `Question1/checkpoints/attention_model.pt` (73 MB) — both included (< 95 MB limit).

---

## Question 2 — Amazon Sentiment: BiGRU vs RoBERTa

**Task:** Five-class Amazon review rating prediction (1–5 stars). Two distinct pipelines: GloVe-BiGRU and fine-tuned `roberta-base`.

**Commands:**
```bash
# 1. Download GloVe-6B-100d to glove.6B/ (see notebook cell 1 for path config)
# 2. Run Question_2/Question_2.ipynb → Run All (Colab/Kaggle recommended)
# Dataset: Hugging Face amazon_reviews_multi (auto-downloaded)
```

**Runtime/Hardware:** BiGRU ~30 min on T4; RoBERTa ~90 min on T4 (3 epochs × 2 seeds).

**Output locations:** `Question_2/Question_2.ipynb` (all figures/tables in notebook output), `Question_2/report.md`

### Key Results (mean ± std, 2 seeds: 42 and 123)

| Model | Accuracy | Macro F1 | MAE (stars) |
|---|---:|---:|---:|
| BiGRU + additive attention | 0.5684 ± 0.0000 | 0.5699 ± 0.0008 | 0.5191 ± 0.0038 |
| RoBERTa + masked mean | 0.5380 ± 0.0037 | 0.5307 ± 0.0031 | 0.6123 ± 0.0021 |

**Sample input→output:**
- Review: *"Ran entirely too small."* → Label 1★ → BiGRU: 1★ ✓, RoBERTa: 2★ ✗

**Conclusion:** BiGRU outperforms RoBERTa in this run. RoBERTa training had not converged (val macro-F1 rising through final epoch). Full report in `Question_2/report.md`.

---

## Question 3 — Abstractive Summarisation (CNN/DailyMail)

**Task:** Train a 13.3M-parameter Transformer from scratch on CNN/DailyMail. Tokeniser only from `bert-base-uncased` (no pretrained model weights). Evaluate against Lead-3.

**Commands:**
```bash
# Full run (GPU required, ~2–2.5 h on T4):
#   Question_3/summarizer.ipynb → Run All
#   Auth Google Drive when prompted (Colab only)

# CPU smoke test (< 5 min):
export SUMMARIZER_PRESET=smoke
# Then: Question_3/summarizer.ipynb → Run All
```

**Runtime/Hardware:** ~110 min training on CUDA GPU. Full eval (11,490 test examples) ~15 min.

**Output locations:** `Question_3/results.json`, `Question_3/predictions.json`, `Question_3/figures/`

### Key Results

| System | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---:|---:|---:|---:|
| **Our Transformer** | **34.28** | **13.94** | **24.50** | **31.82** |
| Lead-3 | 40.05 | 17.48 | 25.02 | 36.35 |

Source ablation gap: **+2.70 nats/token** (confirms source usage). Training: 6 epochs (epoch 6 partial: 50,009/53,838 steps), best val loss 2.7997.

**Sample:**
- Article: *"The Palestinian Authority officially became the 123rd member of the International Criminal Court..."*
- Generated: *"the palestinian authority officially became 123rd member of the international criminal court. the formal accession marked with a ceremony at the hague in the netherlands. the court is based in the netherlands, where the court is based."*

**Conclusion:** The model learns genuine abstractive patterns (+2.70 nat source-ablation gap) but falls below Lead-3 due to insufficient training epochs and greedy decoding without repetition control (68.1% of outputs contain repeated trigrams).

**Checkpoints:** Not included (assignment policy forbids Q3 checkpoint submission).  
**Regeneration:** Run `summarizer.ipynb` on Colab with T4/A100 (~110 min training).

---

## Question 4 — Plot-Twist Generator (TinyStories)

**Task:** Train Vanilla RNN (Model A) and LSTM (Model B) from scratch on TinyStories. Compare perplexity, decoding strategies, and human-rated story quality. Evaluate on adult-style plot-twist prompts.

**Commands (in order):**
```bash
cd Question_4/plot_twist_generator

# Phase 1: Data preparation
python src/run_phase1.py

# Phase 2: Train RNN and LSTM (GPU ~38 min RNN, ~80 min LSTM)
# CPU subset (3,000 stories, ~15 min RNN):
#   Edit configs/config.yaml: train_stories → 3000
python src/run_phase2.py

# Phase 3: Decoding strategies
python src/run_phase3.py

# Phase 4: Plot-twist challenge
python src/run_phase4.py

# Phase 5: Quantitative + human evaluation
python src/run_phase5.py

# Phase 7: LLM extension (optional, downloads SmolLM2-135M)
python src/run_phase7_llm_extension.py
```

**Runtime/Hardware:** Phase 2: RNN ~38 min, LSTM ~80 min on GPU. All phases on CPU with subset ~30 min total.

**Output locations:** `outputs/tables/`, `outputs/plots/`, `outputs/generations/`

### Key Results

| Model | Test PPL | Human Overall (1–5) | Distinct-2 (top-k) |
|---|---:|---:|---:|
| Vanilla RNN | 18.30 | 3.10 ± 0.06 | 0.904 |
| LSTM | 21.17 | **3.81 ± 0.12** | 0.909 |

**Sample (LSTM, Top-k k=40 T=0.8):**
- Prompt: *"It was a dark and stormy night when the detective arrived at the old mansion."*
- Generated: *"Inside, a little owl named Pip sat by the fire, wondering who left muddy footprints on the floor. The detective looked very carefully and found a trail leading to the garden shed."*

**Conclusion:** LSTM outperforms Vanilla RNN on human-rated coherence (+1.01) and creativity (+0.88) despite higher perplexity — a clear demonstration of the "likelihood trap." Top-k (k=40, T=0.8) is the recommended decoding strategy.

**Checkpoints:** RNN/LSTM `.pt` files excluded from repository (`.gitignore: *.pt`). Regenerate with `python src/run_phase2.py`. Vocab file `checkpoints/vocab.json` (176 KB) is included.

---

## Reproducibility

| Setting | Value |
|---|---|
| Global random seed | 42 (all questions) |
| Q2 additional seed | 123 |
| PyTorch determinism | `torch.manual_seed`, `torch.backends.cudnn.deterministic=True` where applicable |
| HuggingFace tokeniser | `bert-base-uncased` (Q3), `roberta-base` (Q2) — auto-downloaded |

> **Determinism caveat:** CUDA operations may introduce small floating-point non-determinism across hardware. Results may vary slightly between GPU models.

---

## Checkpoint Policy

| Question | Checkpoints | Location | Size |
|---|---|---|---|
| Q1 | ✅ Included | `Question1/checkpoints/` | 44 MB + 73 MB |
| Q2 | Not required (outputs in notebook) | — | — |
| Q3 | ❌ Excluded (assignment policy) | Regenerate via notebook | — |
| Q4 | ❌ Excluded (`.gitignore: *.pt`) | Regenerate via `run_phase2.py` | — |

---

## Coverage and Limitations

- **Q3 checkpoints absent** per assignment policy. Regeneration requires ~110 min on a GPU.
- **Q4 checkpoints absent** due to `.gitignore`. Vocab file is included; regeneration ~80–118 min on GPU.
- **Q2 figures** are embedded in notebook output cells (not separate PNG files).
- **Q2 GloVe embeddings** must be downloaded separately (not in repository; ~822 MB).

---

## Submission Checklist

| Item | Status |
|---|---|
| Q1 notebook + checkpoints + figures | ✅ |
| Q2 notebook + report.md | ✅ |
| Q3 notebook + results.json + figures | ✅ |
| Q4 src/ + notebooks + outputs + vocab | ✅ |
| Root requirements.txt | ✅ |
| Root README.md | ✅ |
| Per-question README.md | ✅ |
| Combined PDF report | ✅ `report/Group14_Combined_Technical_Report.pdf` |
| make_submission.sh | ✅ |

---

## Combined Technical Report

[`report/Group14_Combined_Technical_Report.pdf`](report/Group14_Combined_Technical_Report.pdf)
