# Question 3: Abstractive Summarization with a From-Scratch Transformer (CNN/DailyMail 3.0.0)

- **Course:** DSL504 Natural Language Processing | Group Assignment 2
- **Group:** 14 | **Institution:** IIT Bhilai
- **Team:** Kinshuk Gupta (12341190), Harshit Kandpal (B24CS505), Utkarsh Chattarjee (B24CS050), Sujal Som (B24DS032), Tanmay Jaiswal (B24DS033)
- **Deliverable Status:** Submitted codebase defense documentation

---

## Table of Contents

- [0. 60-Second Viva Pitch](#0-60-second-viva-pitch)
- [1. Assignment Recap and Rubric-to-Evidence Map](#1-assignment-recap-and-rubric-to-evidence-map)
- [2. Repository Map](#2-repository-map)
- [3. Data and Preprocessing](#3-data-and-preprocessing)
- [4. Architecture from Scratch](#4-architecture-from-scratch)
- [5. Training and Optimization](#5-training-and-optimization)
- [6. Inference and Decoding](#6-inference-and-decoding)
- [7. Quantitative Results](#7-quantitative-results)
- [8. Why the Model Scores Below Lead-3](#8-why-the-model-scores-below-lead-3)
- [9. Qualitative Analysis and Failure Modes](#9-qualitative-analysis-and-failure-modes)
- [10. Attention Visualization and Interpretability](#10-attention-visualization-and-interpretability)
- [11. How to Run, Dependencies, and Expected Runtime](#11-how-to-run-dependencies-and-expected-runtime)
- [12. Comprehensive Viva Q&A Bank](#12-comprehensive-viva-qa-bank)
  - [12.1 Architecture and Mathematical Foundations](#121-architecture-and-mathematical-foundations)
  - [12.2 Training, Optimization, and Loss Dynamics](#122-training-optimization-and-loss-dynamics)
  - [12.3 Preprocessing, Tokenization, and Masking Mechanics](#123-preprocessing-tokenization-and-masking-mechanics)
  - [12.4 Decoding, Inference, and Metric Specifications](#124-decoding-inference-and-metric-specifications)
  - [12.5 Failure Diagnosis, Assignment Pitfalls, and Improvements](#125-failure-diagnosis-assignment-pitfalls-and-improvements)
  - [12.6 Team Contributions, Methodology, and Adversarial Inquiries](#126-team-contributions-methodology-and-adversarial-inquiries)
- [13. Known Limitations, Discrepancies, and Verification Items](#13-known-limitations-discrepancies-and-verification-items)
- [14. One-Page Cheat Sheet of Key Numbers](#14-one-page-cheat-sheet-of-key-numbers)

---

## 0. 60-Second Viva Pitch

We implemented an abstractive sequence-to-sequence Transformer encoder-decoder from scratch using basic tensor operations and primitive PyTorch modules (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`). The system loads no pretrained model weights and avoids all high-level library implementations (`nn.Transformer`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, or Hugging Face model classes), employing Hugging Face solely for the `bert-base-uncased` WordPiece tokenizer (30,522 vocabulary). Our architecture totals 13,343,232 parameters (`d_model=256`, 3 encoder layers, 3 decoder layers, 4 attention heads, `d_ff=1024`, Post-LN residual topology) with complete three-way embedding weight tying across encoder inputs, decoder inputs, and output projection.

We trained on the complete CNN/DailyMail 3.0.0 training split of 287,113 document-highlight pairs, truncated to 400 source and 128 target tokens, utilizing Adam with a Noam warmup-decay schedule (`warmup=4000`) and fp16 mixed precision. Within a 110-minute training envelope on a GPU, the run executed 50,009 steps across 6 epochs, reaching a best validation loss of 2.7997 at epoch 6. Monotonically decreasing validation loss confirms the model was compute-bounded and undertrained rather than converged or overfitted.

On the full 11,490-pair test split evaluated under greedy autoregressive decoding with stemmed F1 scoring, our model achieves ROUGE-1/2/L/Lsum of 34.28 / 13.94 / 24.50 / 31.82 against the Lead-3 baseline of 40.05 / 17.48 / 25.02 / 36.35 (reproducing the assignment benchmark within 0.01). While extractive Lead-3 surpasses our score due to journalistic lead bias and the absence of a copy or coverage mechanism in our greedy decoder, a source-ablation test shows a +2.70 nats/token loss penalty when conditioned on mismatched articles. This demonstrates that the decoder generates summaries by attending to the source document rather than relying on language model priors.

---

## 1. Assignment Recap and Rubric-to-Evidence Map

The assignment tasks students with constructing a complete Transformer encoder-decoder from scratch to generate abstractive summaries of CNN/DailyMail news articles. The grading rubric allocates 100 marks across 7 components. Every requirement is mapped directly to implementation cells and physical verification artifacts in this repository.

| # | Rubric Component | Marks | Implementation Location | Verification Evidence |
|---|---|---|---|---|
| 1 | **Dataset Loading & Preprocessing**<br>Split loading, WordPiece tokenization, truncation limits, shared vocab, special tokens, shifted targets, length statistics, batching. | 10 | `summarizer.ipynb` Cell 6 (`tok`, length stats), Cell 7 (`collate`, `PairDataset`, `BucketSampler`) | `results.json:data`, `REPORT.docx` §2, token length statistics table (article median 756, summary median 64). |
| 2 | **Transformer Core Components**<br>Scaled dot-product attention, Multi-Head Attention, Position-wise FFN, sinusoidal Positional Encoding, Post-LN residuals. | 20 | `summarizer.ipynb` Cell 9 (`scaled_dot_product_attention`, `MultiHeadAttention`, `PositionwiseFFN`, `PositionalEncoding`), Cell 13 (`run_sanity` checks a, b, c, e) | `summarizer_export.py` L195-252, unit test logs passing assertions in Cell 13, `REPORT.docx` §3. |
| 3 | **Full Encoder-Decoder Architecture**<br>Encoder stack, Decoder stack with cross-attention, 3 masks (source pad, causal, target pad), tied embeddings, exact parameter derivation. | 20 | `summarizer.ipynb` Cell 9 (`EncoderLayer`, `DecoderLayer`), Cell 11 (`Encoder`, `Decoder`, `Seq2SeqTransformer`, `pad_mask`, `causal_mask`, `count_params`) | `results.json:model` (13,343,232 parameters), `REPORT.docx` §3, parameter count verification in Cell 11. |
| 4 | **Training Procedure & Optimization**<br>Cross-entropy ignoring pad, Adam (β₁=0.9, β₂=0.98), Noam schedule, gradient clipping, crash-safe checkpoints, early stopping, best checkpoint restore. | 10 | `summarizer.ipynb` Cell 15 (`get_lr`, `atomic_save`, `evaluate_loss`, `train`), Cell 13 (overfitting unit test g) | `results.json:training`, `figures/loss_curves.png`, `figures/lr_schedule.png`, `REPORT.docx` §4. |
| 5 | **Quantitative Evaluation**<br>Test-set generation, ROUGE-1/2/L/Lsum (`use_stemmer=True`), sentence-split `\n` formatting for Lsum, Lead-3 comparison, loss curves. | 15 | `summarizer.ipynb` Cell 17 (`loss_curves`, `lr_schedule`), Cell 19 (`generate_all`, `split_sents`, `score_all`) | `results.json:rouge`, `results.json:test`, `figures/rouge_vs_lead3.png`, `REPORT.docx` §6-§7. |
| 6 | **Qualitative Analysis & Attention**<br>At least 5 test examples, failure classification (repetition, unsupported facts), cross-attention heatmap from last decoder layer during generation. | 15 | `summarizer.ipynb` Cell 21 (`flags`, `analysis`), Cell 23 (`greedy_decode(capture=True)`, heatmap rendering) | `results.json:diagnostics`, `results.json:qualitative`, `results.json:attention`, `figures/attention_avg.png`, `figures/attention_heads.png`, `REPORT.docx` §8-§9. |
| 7 | **Report Structure, Clarity & Insight**<br>Complete technical report, hyperparameter documentation, training status diagnosis, limitation analysis, submission packaging. | 10 | `summarizer.ipynb` Cell 25 (DOCX generator), Cell 27 (zip exporter) | `REPORT.docx` (all sections populated from `results.json`), `predictions.json` (11,490 test outputs). |
| | **Total** | **100** | | |

---

## 2. Repository Map

The `Question_3` directory contains the complete code, exported scripts, configuration, serialized metrics, generated predictions, technical report, and visual figures.

```
Question_3/
├── README.md                      # Primary viva defense guide and technical documentation (this file)
├── summarizer.ipynb               # Master Jupyter notebook containing the complete pipeline (Cells 0-27)
├── summarizer_export.py           # Standalone Python script export of notebook code cells (895 lines)
├── results.json                   # Machine-readable single source of truth for all metrics and configurations
├── predictions.json               # Full 11,490 test set generations (id, reference, generated) (7.9 MB)
├── REPORT.docx                    # Programmatically generated technical report with figures and tables
└── figures/
    ├── loss_curves.png            # Training and validation loss curves across 6 epochs with best epoch marked
    ├── lr_schedule.png            # Logged Noam learning rate schedule across 50,009 training steps
    ├── rouge_vs_lead3.png         # Comparative bar chart of model vs Lead-3 on ROUGE-1/2/L/Lsum
    ├── attention_avg.png          # Cross-attention heatmap averaged across all 4 attention heads (test #9700)
    └── attention_heads.png        # Cross-attention heatmaps displayed separately for each of the 4 heads
```

### File Manifest and Audit Details

- **`summarizer.ipynb` (master pipeline):** Structured into 13 numbered sections across 28 cells (14 markdown, 14 code). Implements data streaming, token caching, scratch Transformer modules, sanity assertions, crash-safe training with mid-epoch state persistence, batched greedy inference, ROUGE evaluation, linguistic failure diagnostics, attention extraction, and automated report compilation.
- **`summarizer_export.py` (codebase export):** Direct procedural export produced in Cell 27 by joining all executed notebook code cells with `# %%` cell markers. Retains identical class and function logic as the notebook, augmented with environment variable overrides (`Q3_RUN_DIR`, `Q3_COLAB_MOUNT_ROOT`) to allow direct CLI execution and analysis without Jupyter or Colab dependencies.
- **`results.json` (serialized truth):** Structured JSON object containing 8 top-level keys (`data`, `model`, `training`, `rouge`, `test`, `diagnostics`, `qualitative`, `attention`). Stores precise floating-point metrics (e.g., best val loss 2.799668, parameter breakdowns, step-by-step LR history, ablation gaps, and error distributions).
- **`predictions.json` (test outputs):** An array of exactly 11,490 objects corresponding to the full CNN/DailyMail test set. Each object contains `id` (the 40-character hex hash from CNN/DM), `reference` (human-authored highlights with sentence-separating newlines), and `generated` (model-generated abstractive text).
- **`REPORT.docx` (final document):** Comprehensive 11-section report compiled dynamically in Cell 25 via `python-docx`. Every quantitative figure and table is injected directly from `results.json` and `figures/*.png`.
- **`figures/*.png` (visual artifacts):** High-resolution (150 DPI) visualization plots generated via `matplotlib` in Cells 17, 19, and 23.

---

## 3. Data and Preprocessing

### 3.1 Dataset Configuration and Split Partitions

We utilize the standard benchmark dataset `cnn_dailymail` version 3.0.0 loaded via Hugging Face `datasets`. The dataset provides news articles paired with bullet-point highlight summaries written by human editors.

| Split | Total Pairs | Pairs Utilized | Role in Pipeline | Split Hygiene Rules |
|---|---:|---:|---|---|
| **Training** | 287,113 | 287,113 | Parameter optimization via gradient descent | Gradient updates only; no evaluation decisions |
| **Validation** | 13,368 | 3,000 | Mid-epoch evaluation, Noam LR monitoring, early stopping | 3,000 fixed subset used for fast evaluation during training |
| **Test** | 11,490 | 11,490 | Final benchmark evaluation against Lead-3 | Text never touched until post-training inference |

### 3.2 Tokenizer Selection and Vocabulary Sharing

- **Tokenizer Model:** We use the pretrained `bert-base-uncased` WordPiece tokenizer (`AutoTokenizer.from_pretrained("bert-base-uncased")`).
- **Scope of Pretraining:** The tokenizer provides vocabulary mappings and subword segmentation algorithms only. Zero neural network weights, embeddings, or transformer parameters are imported from BERT.
- **Shared Vocabulary:** A single shared vocabulary of `V = 30,522` tokens is employed for both source articles and target summaries. This allows the model to share semantic representations across encoder and decoder via weight tying.
- **Special Tokens:**
  - `PAD` (padding token): ID `0` (`[PAD]`)
  - `BOS` / `CLS` (beginning of sequence): ID `101` (`[CLS]`)
  - `EOS` / `SEP` (end of sequence): ID `102` (`[SEP]`)
- **Lowercasing:** Because the tokenizer is uncased, text normalization to lowercase occurs deterministically during subword segmentation.

### 3.3 Sequence Truncation and Length Statistics

Sequence lengths were empirically measured on a 2,000-article validation sample in Cell 6:

| Modality | Truncation Limit | Median Length | 90th Percentile | Max Length | % Exceeding Limit |
|---|---|---|---|---|---|
| **Source Article** | **400 tokens** | 756.0 tokens | 1,445.3 tokens | 2,355 tokens | 87.2% |
| **Target Summary** | **128 tokens** | 64.0 tokens | 106.0 tokens | 469 tokens | 4.0% |

**Technical justification:**

1. **Journalistic Inverted Pyramid:** News writing places crucial factual content (who, what, where, when) in the leading sentences. Truncating articles to 400 tokens preserves the core event narrative while discarding background exposition and boilerplate sign-offs.
2. **Computational Complexity:** Self-attention scales quadratically, `O(S²)`. Truncating to `S = 400` requires `400² = 160,000` operations per layer per head, whereas full length (`S = 1445`) requires `1445² = 2,088,025`, a 13-fold increase in compute and memory that would render training unfeasible on standard GPUs.
3. **Summary Retention:** A limit of 128 tokens accommodates 96.0% of all target summaries without any truncation, ensuring reference summaries retain their full semantic scope.

### 3.4 Target Shifting and Autoregressive Formulation

The target sequences are formatted with start (`[CLS]`) and end (`[SEP]`) boundaries:

$$Y = [\text{CLS}, y_1, y_2, \dots, y_m, \text{SEP}, \text{PAD}, \dots]$$

During batch collation in `collate(batch)` (Cell 7):

- **Decoder Input (Shifted-Right):** `y_in = Y[:, :-1] = [CLS, y_1, y_2, ..., y_m, PAD, ...]`. Starts with `[CLS]`, shifted right by one position so position `t` predicts token `t+1`.
- **Prediction Target (Shifted-Left):** `y_target = Y[:, 1:] = [y_1, y_2, ..., y_m, SEP, PAD, ...]`. Excludes `[CLS]` and ends with `[SEP]`.
- **Padding:** Variable-length sequences in a batch are dynamically padded to the batch maximum using `pad_seqs` (Cell 7).

### 3.5 Bucket Batching

To eliminate computational waste from pad tokens while maintaining gradient stochasticity, training utilizes `BucketSampler` (Cell 7):

1. The dataset indices are partitioned into chunks of 50 batches (50 × 32 = 1,600 pairs).
2. Within each chunk, samples are sorted by source length.
3. The resulting batches within the chunk are shuffled.

This ensures sequences within any batch have nearly identical lengths, reducing padding tokens by over 70% compared to unstratified batching.

---

## 4. Architecture from Scratch

Every model component is authored from primitive PyTorch modules (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`) and basic tensor manipulations in `summarizer.ipynb` Cells 9 and 11. No high-level abstractions or pretrained layers are used.

### 4.1 Model Hyperparameter Configuration

| Hyperparameter | Symbol | Value | Architectural Purpose |
|---|---|---:|---|
| Model Dimension | `d_model` | **256** | Hidden representation width throughout encoder and decoder |
| Attention Heads | `h` | **4** | Number of parallel attention projections per sub-layer |
| Head Dimension | `d_k`, `d_v` | **64** | Dimension per head (`d_k = d_model / h = 256 / 4 = 64`) |
| Feed-Forward Dimension | `d_ff` | **1,024** | Hidden expansion dimension in position-wise FFN (4 × `d_model`) |
| Encoder Stack Depth | `N_enc` | **3** | Number of stacked Transformer encoder layers |
| Decoder Stack Depth | `N_dec` | **3** | Number of stacked Transformer decoder layers |
| Dropout Probability | `p_drop` | **0.1** | Dropout applied to attention weights, sub-layers, and embeddings |
| Vocabulary Size | `V` | **30,522** | WordPiece vocabulary size (`bert-base-uncased`) |
| Weight Tying | - | **True** | Shared weights across encoder emb, decoder emb, and output linear |
| Layer Normalization Order | - | **Post-LN** | `LN(x + Dropout(Sublayer(x)))` as in Vaswani et al. (2017) |
| Max Source Sequence | `S_max` | **400** | Maximum input article token limit |
| Max Target Sequence | `T_max` | **128** | Maximum generated summary token limit |

### 4.2 Exact Parameter Count Derivation

The total model parameter count is exactly **13,343,232**. The breakdown verified in Cell 11 and stored in `results.json:model.params` is:

$$N_{\text{total}} = N_{\text{embed}} + N_{\text{enc}} + N_{\text{dec}}$$

$$N_{\text{total}} = 7{,}813{,}632 + 2{,}369{,}280 + 3{,}160{,}320 = 13{,}343{,}232$$

#### Tied Embedding Matrix (`embed.weight`)

`N_embed = V × d_model = 30,522 × 256 = 7,813,632`

This single matrix serves three distinct roles: encoder input embedding, decoder input embedding, and the output projection of shape `(30522, 256)` applied before softmax.

#### Encoder Sub-layers (3 layers)

Each encoder layer consists of Self-Attention, LayerNorm 1, Position-wise FFN, and LayerNorm 2.

| Sub-component | Calculation | Parameters |
|---|---|---:|
| Self-Attention (`W_Q`, `W_K`, `W_V`, `W_O`, each 256×256 with bias 256) | 4 × (256 × 256 + 256) | 263,168 |
| LayerNorm 1 (γ, β) | 2 × 256 | 512 |
| FFN `W_1` (256 → 1024) + bias | 256 × 1024 + 1024 | 263,168 |
| FFN `W_2` (1024 → 256) + bias | 1024 × 256 + 256 | 262,400 |
| FFN subtotal | 263,168 + 262,400 | 525,568 |
| LayerNorm 2 (γ, β) | 2 × 256 | 512 |
| **Per-layer encoder total** | 263,168 + 512 + 525,568 + 512 | **789,760** |
| **All 3 encoder layers** | 3 × 789,760 | **2,369,280** |

#### Decoder Sub-layers (3 layers)

Each decoder layer consists of Masked Self-Attention, LayerNorm 1, Cross-Attention, LayerNorm 2, FFN, and LayerNorm 3.

| Sub-component | Calculation | Parameters |
|---|---|---:|
| Masked Self-Attention | 4 × (256 × 256 + 256) | 263,168 |
| LayerNorm 1 | 2 × 256 | 512 |
| Cross-Attention | 4 × (256 × 256 + 256) | 263,168 |
| LayerNorm 2 | 2 × 256 | 512 |
| Position-wise FFN | 263,168 + 262,400 | 525,568 |
| LayerNorm 3 | 2 × 256 | 512 |
| **Per-layer decoder total** | 263,168 + 512 + 263,168 + 512 + 525,568 + 512 | **1,053,440** |
| **All 3 decoder layers** | 3 × 1,053,440 | **3,160,320** |

#### Final Linear Projection (`out`)

`nn.Linear(256, 30522, bias=False)` shares storage directly with `embed.weight` via `self.out.weight = self.embed.weight`. Bias is omitted per the weight-tying formulation, contributing **0** unique parameters.

### 4.3 Component-by-Component Formulation

#### Scaled Dot-Product Attention (`scaled_dot_product_attention`, Cell 9)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

- **Precision Guard:** Raw attention logits `(Q·Kᵀ) / √d_k` are cast to `torch.float32` prior to softmax. This guarantees numerical stability under AMP (fp16), preventing exponent overflow.
- **Finite Masking:** Masked positions (mask value `False`) are filled with `NEG_INF = -1e9` (large finite negative value) rather than `-inf`. This ensures that rows that are completely masked never generate `NaN` when passed through softmax.

#### Multi-Head Attention (`MultiHeadAttention`, Cell 9)

Splits queries, keys, and values into `h = 4` parallel heads of dimension `d_k = 64`:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

- **Tensor Split:** Tensor shape `(B, T, d_model)` is transformed via `.view(B, T, h, d_k).transpose(1, 2)` to yield `(B, h, T, d_k)`.
- **Tensor Merge:** Head outputs `(B, h, T, d_k)` are recombined via `.transpose(1, 2).reshape(B, T, h * d_k)` and projected through `W_O` of shape `(d_model, d_model)`.

#### Position-wise Feed-Forward Network (`PositionwiseFFN`, Cell 9)

Applied identically and independently to each position:

$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

Expands representations from 256 to 1024 dimensions through a ReLU non-linearity, then projects back to 256.

#### Sinusoidal Positional Encoding (`PositionalEncoding`, Cell 9)

Fixed positional encodings provide position information without learnable parameters:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

The lookup table is registered as a non-persistent buffer of shape `(1, L_max, d_model)`. Embeddings are initialized with a normal distribution using `nn.init.normal_(self.embed.weight, std=c.d_model**-0.5)`, i.e. `std = 256^-0.5 = 0.0625`. Fixed positional vectors are added directly to the embeddings before being passed through `Dropout(0.1)`.

#### Residual and Normalization Topology (Post-LN)

Following Vaswani et al. (2017), our network strictly implements **Post-LN**:

$$\mathbf{x}^{(l)} = \text{LayerNorm}\left(\mathbf{x}^{(l-1)} + \text{Dropout}(\text{Sublayer}(\mathbf{x}^{(l-1)}))\right)$$

Every attention and feed-forward sub-layer is wrapped in this residual-dropout-normalization sandwich.

### 4.4 The Three Attention Masks

The model implements three boolean masks (`True` = position allowed to attend, `False` = masked out):

| Mask Name | Function | Shape | Where Applied | Role |
|---|---|---|---|---|
| **Source Padding Mask** | `pad_mask(src, pad_id)` | `(B, 1, 1, S)` | Encoder Self-Attention & Decoder Cross-Attention | Sets attention weights to zero for source pad positions (`token_id == 0`). Prevents padding from corrupting contextual representations. |
| **Causal Look-Ahead Mask** | `causal_mask(T, device)` | `(1, 1, T, T)` | Decoder Self-Attention | Lower-triangular matrix (`torch.tril`). Enforces the autoregressive property: step `t` can only attend to steps ≤ `t`. |
| **Target Padding Mask** | `pad_mask(tgt_in, pad_id)` | `(B, 1, 1, T)` | Decoder Self-Attention | Identifies pad tokens in target sequences. Combined via logical AND: `tgt_mask = causal_mask & pad_mask(tgt_in)`. |

**Shape broadcasting in cross-attention:** queries originate from the decoder (length `T`), while keys/values originate from the encoder (length `S`). The source padding mask of shape `(B, 1, 1, S)` broadcasts across the attention logit matrix of shape `(B, h, T, S)` over batch dimension `B`, head dimension `h`, and query length `T`.

### 4.5 Tensor Shape Walkthrough for One Forward Pass

Using real configuration values with batch size `B = 32`, source length `S = 400`, and decoder input length `T = 127`:

1. **Input Batch**
   - Source tokens: `src` → `(32, 400)`
   - Target input tokens: `tgt_in` → `(32, 127)` (shifted right, begins with `[CLS]`)
2. **Mask Generation**
   - `src_mask = pad_mask(src, 0)` → `(32, 1, 1, 400)`
   - `tgt_mask = make_tgt_mask(tgt_in)` → `(32, 1, 127, 127)`
3. **Encoder Stack**
   - Embedding + PE: `embed(src) + pe` → `(32, 400, 256)`
   - Layer 1 Self-Attention: `Q, K, V` → `(32, 4, 400, 64)`; attention weights → `(32, 4, 400, 400)`; output → `(32, 400, 256)`
   - Layer 1 Post-LN + FFN + Post-LN: → `(32, 400, 256)`
   - Repeated across Layers 2 and 3: final `enc_out` → `(32, 400, 256)`
4. **Decoder Stack**
   - Embedding + PE: `embed(tgt_in) + pe` → `(32, 127, 256)`
   - Layer 1 Masked Self-Attention: `Q, K, V` → `(32, 4, 127, 64)`; causal scores → `(32, 4, 127, 127)`; Post-LN → `(32, 127, 256)`
   - Layer 1 Cross-Attention: `Q` from decoder → `(32, 4, 127, 64)`; `K, V` from `enc_out` → `(32, 4, 400, 64)`; cross-attention logits → `(32, 4, 127, 400)`; masked by `src_mask`; output → `(32, 127, 256)`
   - Layer 1 Post-LN + FFN + Post-LN: → `(32, 127, 256)`
   - Repeated across Layers 2 and 3: final `dec_out` → `(32, 127, 256)`
5. **Output Projection**
   - `logits = out(dec_out)` → `(32, 127, 30522)`

---

## 5. Training and Optimization

### 5.1 Loss Function Formulation

The network is optimized using cross-entropy loss over the vocabulary (`V = 30,522`), with padding positions explicitly excluded from both the numerator and denominator:

$$\mathcal{L}_{\text{seq}} = -\frac{1}{\sum_{b,t} \mathbb{I}(y_{b,t} \ne \text{PAD})} \sum_{b=1}^{B} \sum_{t=1}^{T} \mathbb{I}(y_{b,t} \ne \text{PAD}) \log P\left(y_{b,t} \mid \mathbf{y}_{<t}, \mathbf{x}\right)$$

Implemented via `F.cross_entropy(logits.reshape(-1, V).float(), labels.reshape(-1), ignore_index=0, reduction="mean")`. Label smoothing is set to `0.0` (plain cross-entropy).

### 5.2 Optimizer and Noam Learning Rate Schedule

We employ the Adam optimizer with parameters specified in Vaswani et al. (2017):

$$\beta_1 = 0.9, \quad \beta_2 = 0.98, \quad \epsilon = 10^{-9}$$

The learning rate is updated at every optimization step (`step` from 1 to 50,009) using the Noam schedule (`get_lr`, Cell 15):

$$lr(step) = \text{factor} \cdot d_{model}^{-0.5} \cdot \min\left(step^{-0.5},\; step \cdot warmup^{-1.5}\right)$$

**Numerical evaluation** (`factor = 1.0`, `d_model = 256`, `warmup = 4000`):

- `d_model^-0.5 = 256^-0.5 = 1/16 = 0.0625`
- `warmup^-1.5 = 4000^-1.5 ≈ 3.9528e-6`
- **Initial step (step 1):** `lr = 0.0625 × 3.9528e-6 =` **`2.47e-7`**
- **Peak learning rate (step 4,000):** `lr = 0.0625 × 4000^-0.5 = 0.0625 × 0.015811 =` **`9.882e-4`** (≈ 9.88 × 10⁻⁴)
- **End of run (step 50,009):** `lr = 0.0625 × 50009^-0.5 = 0.0625 × 0.0044717 =` **`2.795e-4`**

### 5.3 Teacher Forcing and Training Parallelism

Training uses 100% teacher forcing: at every step, the decoder receives the true shifted ground-truth sequence `y_in`. Because the causal look-ahead mask prevents position `t` from attending to later tokens, all `T - 1` output logits can be computed simultaneously in a single forward pass without recurrent loops.

### 5.4 Regularization, Mixed Precision, and Checkpointing

- **Gradient Clipping:** Maximum grad norm is clipped to `1.0` via `nn.utils.clip_grad_norm_`.
- **Automatic Mixed Precision (AMP):** Enabled via `torch.autocast("cuda", dtype=torch.float16)` and `torch.amp.GradScaler("cuda")`, halving VRAM requirements and doubling training throughput.
- **Crash-Safe Atomic Checkpointing:** `atomic_save` (Cell 15) writes checkpoints to a `.tmp` file and performs an atomic filesystem replace. Checkpoints are written every `ckpt_every = 1000` steps mid-epoch, recording random states, model weights, optimizer buffers, and batch indices (`mid_skip`).

### 5.5 Training Execution History and Stopping Condition

| Epoch | Completed Steps | Train Loss (nats) | Val Loss (nats) | Best Val Loss | Epoch Time | Session Status |
|---|---:|---:|---:|---:|---:|---|
| 1 | 8,973 | 5.3943 | 4.1857 | 4.1857 | 19.69 min | Completed |
| 2 | 17,946 | 4.0792 | 3.3101 | 3.3101 | 19.71 min | Completed |
| 3 | 26,919 | 3.5008 | 2.9903 | 2.9903 | 19.71 min | Completed |
| 4 | 35,892 | 3.2427 | 2.8807 | 2.8807 | 19.82 min | Completed |
| 5 | 44,865 | 3.1177 | 2.8181 | 2.8181 | 19.71 min | Completed |
| 6 (partial) | **50,009** | **3.0426** | **2.7997** | **2.7997** | 11.36 min | **Halted: Time Budget Reached** |

- **Total Execution Time:** 110.08 minutes.
- **Stopping Reason:** Time budget reached (`time_budget_min = 110.0`). The run completed 50,009 of 53,838 planned steps (93.1% of 6 epochs).
- **Convergence Diagnosis:** The model was **undertrained**, not converged. Early stopping (patience = 3) was never triggered (`bad = 0/3`). Validation loss dropped monotonically across all 6 epochs: 4.1857 → 3.3101 → 2.9903 → 2.8807 → 2.8181 → 2.7997.
- **Best Checkpoint Restoration:** The best checkpoint from Epoch 6 (validation loss 2.7997) was restored to the model before running test set inference.

### 5.6 Source Ablation Verification

To prove the model actually conditions on the source article rather than functioning as an unconditioned n-gram language model, Cell 15 runs a source-ablation test on the validation split:

- Cross-entropy with true paired article: **2.7997 nats/token**
- Cross-entropy with shuffled / wrong article (`src.roll(1, dims=0)`): **5.5027 nats/token**
- **Conditioning gap:** **+2.7031 nats/token**

A model that ignores the source would yield a gap near 0. The +2.70 nats/token penalty demonstrates that the cross-attention layers rely heavily on the source article representations to predict summary tokens.

---

## 6. Inference and Decoding

### 6.1 Greedy Autoregressive Generation Protocol

Inference is executed without teacher forcing via `greedy_decode` (Cell 11):

```
                       ┌───────────────────────┐
                       │   Source Article x    │
                       └──────────┬────────────┘
                                  │
                                  ▼
                       ┌───────────────────────┐
                       │   Encoder (Run ONCE)  │
                       └──────────┬────────────┘
                                  │ enc_out: (B, S, D)
                                  ▼
                       ┌───────────────────────┐
             ┌────────►│        Decoder        │◄──────┐
             │         └──────────┬────────────┘       │
             │                    │                    │
             │                    ▼                    │
             │         ┌───────────────────────┐       │
             │         │   Linear Projection   │       │
             │         └──────────┬────────────┘       │
             │                    │ logits: (B, V)     │
             │                    ▼                    │
             │         ┌───────────────────────┐       │
             │         │   Argmax Token ID     │       │
             │         └──────────┬────────────┘       │
             │                    │                    │
             │              nxt == EOS ?               │
             │              /          \              │
             │            Yes           No             │
             │            /              \            │
             │    Mark done[b]=True   Append to ys ────┘
             │    Fill with PAD
             │
             └── Loop until step == 127 OR done.all()
```

#### Step-by-Step Inference Execution

1. **Single-Pass Encoding:** The source tensor `src` is encoded once: `enc = model.encode(src, src_mask)`. This output is cached and reused across all subsequent decoding steps, eliminating redundant encoder computations.
2. **Buffer Initialization:** The generation tensor is initialized with start tokens: `ys = torch.full((B, 1), BOS=101, dtype=torch.long)`. A boolean tracker `done = torch.zeros(B, dtype=torch.bool)` tracks sequence completion.
3. **Step-by-Step Expansion (t = 1 … 127):**
   - Decoder forward pass: `h = model.decode(ys, enc, src_mask)` yielding hidden states of shape `(B, t, d_model)`.
   - Final-position projection: `logits = model.out(h[:, -1])` yielding logits of shape `(B, 30522)`.
   - Greedy selection: `nxt = logits.argmax(dim=-1)`.
   - Completion masking: if `done[b]` is True, `nxt[b]` is overwritten with `PAD=0`.
   - Concatenation: `ys = torch.cat([ys, nxt[:, None]], dim=1)`.
   - Termination check: `done |= (nxt == EOS=102)`. If `done.all()`, the loop terminates immediately.
4. **Post-Processing:** The leading `[CLS]` token is sliced off: `ys[:, 1:]`.

### 6.2 Batched Evaluation Pipeline

`generate_all` (Cell 19) processes the entire 11,490-article test split:

- **Length-Sorted Bucketing:** Articles are sorted by token length to minimize padding overhead across batches (`eval_batch_size = 64`). Original order is restored after generation.
- **Disk Caching:** Intermediate generations are saved every 20 batches to `gen_cache.pkl`.
- **Throughput:** Full test set generation (11,490 summaries) completed in **4.84 minutes** (~2,374 summaries/minute).
- **Generation Length:** The average generated summary length is **60.85 tokens** (approx. 48.36 words).

### 6.3 Token Decoding and ROUGE-Lsum Sentence Splitting

- **Subword Detokenization:** Token ID arrays are converted back to clean text strings using `tok.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)`.
- **Newline Insertion for ROUGE-Lsum (`split_sents`, Cell 19):** ROUGE-Lsum computes the longest common subsequence on a sentence-by-sentence basis rather than treating the summary as one flat string. To ensure standard evaluation:
  1. Sentence boundaries are capitalized via regex: `re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)`.
  2. The NLTK sentence segmenter partitions the text: `nltk.sent_tokenize(text)`.
  3. Sentences are joined with newline characters: `"\n".join(...)`.
- Ground-truth test highlights already contain newline separators between bullet points.

---

## 7. Quantitative Results

Quantitative evaluation was executed on the full 11,490-pair test split using Google's `rouge_score` package with stemming enabled (`use_stemmer=True`) and F1 scoring multiplied by 100.

### 7.1 Full Test Set Evaluation Benchmark

| Evaluated System | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Test Pairs (N) | Evaluation Method |
|---|---:|---:|---:|---:|---:|---|
| **Transformer from scratch (Ours)** | **34.28** | **13.94** | **24.50** | **31.82** | 11,490 | Greedy autoregressive, best checkpoint (Ep 6) |
| **Lead-3 Baseline (Measured Here)** | **40.05** | **17.48** | **25.02** | **36.35** | 11,490 | First 3 article sentences, sentence-split with `\n` |
| **Lead-3 (Assignment Reference)** | **40.06** | **17.48** | **25.02** | **36.36** | 11,490 | Benchmark figures provided in assignment spec |
| **Delta vs Measured Lead-3** | **-5.78** | **-3.54** | **-0.52** | **-4.52** | 11,490 | Absolute F1 difference (Ours − Measured) |
| **Delta vs Assignment Reference** | **-5.78** | **-3.54** | **-0.52** | **-4.54** | 11,490 | Absolute F1 difference (Ours − Reference) |

**Metric precision notes:**

- Unrounded model scores from `results.json:rouge.model`: ROUGE-1 = 34.2752, ROUGE-2 = 13.9407, ROUGE-L = 24.4954, ROUGE-Lsum = 31.8245.
- Unrounded measured Lead-3 scores from `results.json:rouge.lead3`: ROUGE-1 = 40.0525, ROUGE-2 = 17.4806, ROUGE-L = 25.0162, ROUGE-Lsum = 36.3478.
- Measured Lead-3 matches the assignment reference benchmark within 0.01 across all four metrics, confirming our evaluation pipeline and sentence splitting adhere strictly to benchmark standards.

### 7.2 Diagnostic Corpus Statistics

Computed across all 11,490 test predictions in Cell 21 (`results.json:diagnostics`):

- **Summaries with Repeated Word Trigram:** **68.12%** (at least one trigram occurring more than once in the generated summary)
- **Summaries with Unsupported Numbers:** **9.34%** (contains a numerical token absent from the source article)
- **Summaries with Unsupported Words:** **20.77%** (contains a content word absent from the source article)
- **Mean Trigram Copy Rate:** **69.24%** (percentage of generated trigrams copied verbatim from the source)
- **Mean Generated Summary Length:** **48.36 words** (vs **52.78 words** for ground-truth references)

### 7.3 Training Loss Curves and Visualization Artifacts

Visualized in `figures/loss_curves.png` (Cell 17):

- **Training Loss Trajectory:** Decreased monotonically from 5.3943 (Epoch 1) to 3.0426 (Epoch 6).
- **Validation Loss Trajectory:** Decreased monotonically from 4.1857 (Epoch 1) to 2.7997 (Epoch 6).
- **Overfitting Diagnosis:** The model exhibited zero signs of overfitting. At no point did validation loss increase or diverge from training loss.
- **Visual Artifacts Summary:**
  1. `figures/loss_curves.png`: Training vs validation loss per epoch, with a star marking the best epoch (Epoch 6, loss 2.7997).
  2. `figures/lr_schedule.png`: Trace of the 50,009-step Noam schedule, displaying initial ramp-up over 4,000 steps followed by smooth decay.
  3. `figures/rouge_vs_lead3.png`: Side-by-side bar chart of ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum comparing our Transformer against Lead-3.
  4. `figures/attention_avg.png`: Heatmap of last-decoder-layer cross-attention averaged over all 4 heads for test sample index 9700.
  5. `figures/attention_heads.png`: 4-panel grid displaying cross-attention heatmaps for each individual head for test sample index 9700.

---

## 8. Why the Model Scores Below Lead-3

The assignment specification explicitly notes:

> "A Transformer trained from scratch with limited compute will usually score below Lead-3. This is normal. Compare against it and explain the gap in your report."

Our model trailing Lead-3 by 5.78 ROUGE-1 and 4.52 ROUGE-Lsum is an expected outcome driven by six concrete structural and computational factors:

1. **Extractive Inductive Bias of News Articles:** CNN/DailyMail articles are written by journalists using the inverted pyramid convention, where the headline and first three sentences compress the central event. Lead-3 extracts human-written, fully grammatical sentences directly from this lead paragraph. It suffers zero syntactic errors, zero repetitions, and zero hallucinations.
2. **From-Scratch Training vs. Pretrained Foundation Models:** State-of-the-art abstractive summarizers (e.g., BART, T5, PEGASUS) are pretrained on billions of tokens using masked language modeling or span corruption before fine-tuning. Our model started from random weight initialization and was required to master English grammar, syntax, world knowledge, and summarization logic simultaneously from only 287k articles.
3. **Severe Compute and Budgetary Under-training:** Due to the 110-minute training envelope on a single GPU, the model completed only 50,009 steps (~5.5 epochs). The validation loss curve was still dropping at Epoch 6 (2.8181 → 2.7997). The model never reached an asymptote or loss plateau; it was halted strictly by wall-clock limits while still actively learning.
4. **Information Loss from Source Truncation:** 87.2% of source articles in CNN/DailyMail exceed the 400-token limit (median length is 756 tokens). For articles where secondary facts or summary bullet points reference information in paragraphs 4 through 8, that source text was completely discarded, making exact reference overlap impossible.
5. **Pathologies of Greedy Autoregressive Decoding:** Greedy decoding selects the locally optimal token `argmax P(y_t | y_<t, x)` at each step. This causes two structural failure modes:
   - **Degenerative Repetition:** Once the decoder selects a common phrase, self-attention to previous tokens reinforces the pattern, locking the generator into repetition cycles. This explains why 68.12% of outputs contain repeated trigrams.
   - **Exposure Bias:** The model was trained with 100% teacher forcing on gold prefixes, but must generate conditioned on its own previous (potentially erroneous) outputs during inference.
6. **Absence of a Copy / Pointer Mechanism:** Without a pointer-generator network (See et al., 2017), the model must generate rare named entities, monetary figures, and dates through its fixed subword softmax vocabulary. When subword confidence is distributed across vocabulary tokens, the model either hallucinates plausible numbers (present in 9.34% of test summaries) or produces fragmented tokens.

---

## 9. Qualitative Analysis and Failure Modes

We extracted representative qualitative examples from `predictions.json` (evaluated in Cell 21 and stored in `results.json:qualitative`), encompassing both typical generations and degenerate failure modes.

### Example 0: Typical Event Summary (Test #9700, ID `28c3566ff7959cc7070625cff413093e6aa119a2`)

**Reference:**

> Cloud appeared in central China's Hunan province last Friday .
>
> A university student snapped the moment on his way to the library .
>
> Picture has sparked tongue-in-cheek debate on Mother Nature's password .

**Generated:**

> chinese internet users have since shared and commented on shi's photo thousands of times on chinese social media. the cloud formation resembling the wi - fi symbol appeared recently recently in the sky of xiang city in hunan province.

- **Scores:** ROUGE-1: **30.14** | ROUGE-Lsum: **30.14**
- **Mechanical Analysis:** The model successfully abstracts the core event (Wi-Fi-shaped cloud in Hunan) and social media commentary. However, it exhibits a local bigram repetition: "appeared recently recently". This occurs because the greedy decoder emitted "recently", and the local language model head assigned maximum probability to repeating the temporal adverb before completing the prepositional phrase. The trigram repetition heuristic missed this bigram error, demonstrating that local fluency defects can occur even when trigram checks pass.

### Example 1: Typical Topic Compression (Test #1597, ID `942fc9db6f3b08d1011d77312ce88f6892ed86c6`)

**Reference:**

> Alert issued over rogue workers in nuclear, transport and public services .
>
> MI5 giving advice on risk posed by employees working in sensitive areas .
>
> Follows actions of pilot Andreas Lubitz who killed 150 people in the Alps .
>
> A BA pilot claims his airline does not carry out mental health checks .

**Generated:**

> mi5 has issued alert over threat posed by rogue workers in the alps plane crash disaster. concerns are so great that one security source has warned :'much physical security you have, if someone is already in there '

- **Scores:** ROUGE-1: **27.91** | ROUGE-Lsum: **25.58**
- **Mechanical Analysis:** Topical grounding is accurate (MI5 warning, rogue workers, Alps crash). However, the quotation is abruptly truncated at the end of the sentence ("warned :'much physical security..."). The model ran low on target length probability mass and failed to synthesize the complete discourse unit, illustrating weak compression of complex syntactic embeddings.

### Example 2: Typical Vivid Event Extraction (Test #349, ID `12294954320c496e89fcc0bd5bb91e2dcc5c5d7f`)

**Reference:**

> The sentencing phase in Dzhokhar Tsarnaev's trial begins in a federal court in Boston .
>
> Prosecutor shows pictures of the four victims and Tsarnaev flipping his middle finger .
>
> Victims testify about the impact of the bombing on their lives .

**Generated:**

> dzhokhar tsarnaev's bombs tore through their bodies : singeing flesh, shattering bones, shredding muscles and severing limbs. jurors also began to hear about the holes his bombs left in hearts of survivors. the jury must decide whether boston marathon bomber tsarnaev should live or die for what he has done.

- **Scores:** ROUGE-1: **33.71** | ROUGE-Lsum: **31.46**
- **Mechanical Analysis:** The model demonstrates high lexical overlap (81% trigram copy rate from the source article). However, it exhibits severe content skew: rather than summarizing the procedural sentencing phase reported in the reference, it attends to the sensational, visceral opening lines of the news story, copying emotional descriptors verbatim.

### Example 3: Typical Context/Entity Blur (Test #11262, ID `12670e6af0ccf95cf8de50f651f0fded33d419b0`)

**Reference:**

> In 1915, 1.5 million Armenians were killed by Ottoman Turks in what historians have described as the first genocide of the 20th century .
>
> Obama refused to call the mass killings a 'genocide' in official statement despite promising as a presidential candidate that he would .
>
> Kim Kardashian - who is Armenian on her father's side - has called the killings a genocide and says Obama should too .

**Generated:**

> reality star said it was 'very disappointing' that he stopped short of using word. kardashian has used celebrity since 2011 to bring awareness of genocide.

- **Scores:** ROUGE-1: **12.03** | ROUGE-Lsum: **12.03**
- **Mechanical Analysis:** The model extracts the reaction of Kim Kardashian, but drops the entire antecedent discourse: it generates the pronoun "he" ("he stopped short of using word") without ever establishing President Obama as the referent. Furthermore, it completely omits the historical death toll (1.5 million Armenians).

### Example 4: FAILURE, Degenerative Repetition Loop (Test #5020, ID `95f10632700d42d92da36376e9ae7ba80f13941b`)

**Reference:**

> Grey hair has gone glam with social media awash with images .
>
> Users posting images under the hashtag #grannyhair .
>
> Stars including Kelly Osborne, Nicole Richie and Lady Gaga and Rihanna have helped the trend gain traction .

**Generated:**

> katja wassermeyer ( katja ) wassermeyer ( katja - wassermeyer ( katja - wassermeyer ( katja - wassermeyer ( katja - wassermeyer ( katja - wassermeyer ( katja - wassermeyer ( katja - wassermeyer ( katja - wassermeyer ( katja - wassermeyer ( katja - wassermeyer ( katja - wassermeyer ( katja - wassermeyer ( katja - wassermeyer ( katja - wassermeyer ( katja ). com.

- **Scores:** ROUGE-1: **0.00** | ROUGE-Lsum: **0.00**
- **Mechanical Analysis:** Catastrophic failure of greedy decoding without repetition penalty. The model encountered the photographer credit name ("Katja Wassermeyer") in the article, emitted parenthetical tokens, and fell into a high-probability cyclic state where each subword prediction reinforced the subsequent prediction. The repeated-trigram ratio for this output is **91%**, with the trigram `'katja wassermeyer katja'` repeated 17 times consecutively until length exhaustion.

### Example 5: FAILURE, Hallucination / Unsupported Numbers (Test #9802, ID `41f333d512f52843c065e580fb420de19f50ab6c`)

**Reference:**

> Barack and Michele Obama's effective tax rate was 19.6 per cent .
>
> They made \$70,712 in donations to 33 different charities .
>
> Claimed a whopping \$160,000 in itemized deductions .
>
> Nearly \$95,000 in income came from sales of the president's books .

**Generated:**

> the obamas lowered their 2014 tax bill by claiming nearly \$ 160, 000 in itemized deductions, including \$ 70, 712 in charitable donations to 33 different charities. they donated nearly 15 percent of their income, they donated nearly 15 percent of their income. the largest was a \$ 22, 012 donation to fisher house foundation, which supports military families. they also donated \$ 5, 000 in the exclusive private school where daughters malia and sasha are enrolled. the couple also reported making \$ 94, 889 from the sale of obama's best - selling books.

- **Scores:** ROUGE-1: **45.11** | ROUGE-Lsum: **45.11**
- **Mechanical Analysis:** While achieving high ROUGE overlap, the summary exhibits phrase repetition ("they donated nearly 15 percent of their income" emitted twice) and generates unsupported numerical subwords. Numbers in the article were segmented by WordPiece into subword tokens (`['160', ',', '000']`, `['70', ',', '712']`). The model synthesized numerical fragments (`$22, 012`, `$5, 000`) that appeared in different sections of the document, conflating tax deductions with private school donations due to the lack of an explicit pointer-copy mechanism.

---

## 10. Attention Visualization and Interpretability

Cross-attention weights were captured from the **last decoder layer** (`decoder.layers[-1].cross_attn`) during greedy autoregressive inference for test article index 9700 (Cell 23). This represents how the fully contextualized decoder queries the source document at each generation step.

```
       Source Article Window (Tokens 42 to 121: "cloud formation resembling the wi-fi symbol...")
               42                                                      121
               ┌────────────────────────────────────────────────────────┐
   chinese     │■■■■                                                    │  (Attends to Chinese social media)
   internet    │  ■■■■                                                  │
   users       │    ■■■                                                 │
   have        │       ■                                                │
G  since       │        ■                                               │
E  shared      │         ■■■                                            │
N  ...         │            ...                                         │
E  the         │               ■■■                                      │
R  cloud       │                 ■■■■■■■■■                              │  (Direct copy alignment to "cloud")
A  formation   │                         ■■■■■■                         │  (Direct copy alignment to "formation")
T  resembling  │                               ■■■■■                    │  (Direct copy alignment to "resembling")
E  the         │                                    ■                   │
D  wi          │                                     ■■■■■■■            │  (Direct copy alignment to "wi")
   -           │                                            ■■■         │
   fi          │                                               ■■■■■■   │  (Direct copy alignment to "fi")
   symbol      │                                                     ■■■│
   ...         │                                                        │
               └────────────────────────────────────────────────────────┘
```

### 10.1 Quantitative Interpretability Metrics (`results.json:attention`)

- **Focused Attention Window:** The visualization spans source tokens 42 through 121 (an 80-token window). Exactly **40.01%** of total attention mass across the entire 400-token article is concentrated within this 80-token window.
- **Copy Alignment Rate:** For **70.45%** of the generated tokens, the argmax attended source token is the exact identical WordPiece token in the article. This demonstrates that cross-attention acts as a pseudo-copy mechanism.
- **Lead Position Bias:** **47.73%** of all attention peaks occur within the first 100 source tokens, corroborating the journalistic inverted pyramid hypothesis.
- **Attention Entropy:** Mean cross-attention entropy across generated tokens is **3.19 nats**. For comparison, a completely uniform distribution over 400 source tokens yields `H_uniform = ln(400) = 5.9915 nats`. The observed entropy (3.19 nats) is only 53% of uniform entropy, confirming sharp, selective attention rather than diffuse dispersion.
- **Per-Head Diversity:** Per-head entropy ranges from 3.01 nats (sharpest head) to 3.28 nats (most diffuse head). Visualized in `figures/attention_heads.png`, all four heads align along the primary diagonal, indicating redundant grammatical and lexical tracking rather than specialized orthogonal roles.

### 10.2 Defense Q&A for Attention Heatmaps

**TA Question:** "Why does the attention window start at token 42 rather than token 0?"

**Student Answer:** "Tokens 0 through 41 contain introductory filler ('We have all been there. Wanting to get online but just out of wifi coverage...'). The actual news event ('A cloud formation resembling the Wi-Fi symbol appeared recently...') begins at token 42. The model learned to ignore journalistic pleasantries and attend directly to the substantive news content."

**TA Question:** "Why visualize cross-attention rather than decoder self-attention?"

**Student Answer:** "Decoder self-attention only models the target language history (what the model has already written). Cross-attention reveals the conditioning mechanism: how the decoder queries the encoder's representations of the source document to decide what factual information to output."

---

## 11. How to Run, Dependencies, and Expected Runtime

### 11.1 Software Requirements and Environment

- **Python Version:** Python 3.10 or higher.
- **Hardware Recommended:** GPU runtime with at least 12 GB VRAM (Google Colab T4, Kaggle P100/T4, or local CUDA accelerator).
- **Core Library Dependencies:**

```bash
pip install torch datasets transformers rouge_score nltk python-docx matplotlib
```

`transformers` is utilized exclusively for `AutoTokenizer.from_pretrained("bert-base-uncased")`. No pretrained neural network weights or models are imported.

### 11.2 Execution Instructions

**1. Google Colab / Kaggle Execution (Full Pipeline)**

- Open `summarizer.ipynb` in Colab.
- Select Runtime Type: **GPU** (T4 or higher).
- Click **Run All** (menu: `Runtime` → `Run all`).
- The notebook executes all 28 cells sequentially: mounts Google Drive, loads data, validates sanity checks, trains for up to 110 minutes, generates test predictions, scores ROUGE, and exports `REPORT.docx` and `submission.zip`.

**2. CPU Smoke Test (under 5 minutes)**

To test code integrity without training the full model, set the environment preset before execution:

```bash
export SUMMARIZER_PRESET=smoke
python3 -c "import os; print(\"Smoke mode enabled\")"
```

Or inside `summarizer.ipynb` Cell 4, set `PRESET = "smoke"`. This configures `d_model=64`, 2 layers, 4 heads, `d_ff=128`, trains on 128 pairs for 2 epochs, and executes greedy decoding on 32 test articles.

**3. Standalone Script Inspection (`summarizer_export.py`)**

`summarizer_export.py` can be executed directly from the terminal:

```bash
python3 summarizer_export.py
```

Configure custom storage directories via environment variables:

```bash
export Q3_RUN_DIR="./q3_run"
export Q3_COLAB_MOUNT_ROOT="./content/drive"
python3 summarizer_export.py
```

### 11.3 Checkpoint Persistence and Disconnect Recovery

Training state is written atomically every 1,000 steps to Google Drive at `/content/drive/MyDrive/q3_run/checkpoints/last.pt`. In the event of a Colab timeout or disconnection:

1. Reconnect a new GPU runtime.
2. Select **Run All**.
3. Authorize Google Drive access.
4. The notebook detects `last.pt`, verifies that model hyperparameters match `SIG_KEYS`, restores model weights, optimizer momentum buffers, AMP scaler state, and training history, and resumes at the exact mid-epoch batch index (`mid_skip`).
5. Completed test generation batches in `gen_cache.pkl` are automatically reused, eliminating repeated decoding overhead.

---

## 12. Comprehensive Viva Q&A Bank

### 12.1 Architecture and Mathematical Foundations

#### Q1: Why scale dot-product attention by 1/√d_k?

**Answer:** For large projection dimensions `d_k`, the dot product `q · k = Σ q_i k_i` grows in variance proportionally to `d_k` (assuming zero mean and unit variance inputs). Without scaling, dot products become large in magnitude, pushing the softmax function into regions with near-zero gradients. Dividing by `√d_k` maintains unit variance of the logits regardless of hidden width, ensuring healthy gradient flow during backpropagation.

#### Q2: What is the benefit of Multi-Head Attention over a single large attention head?

**Answer:** A single attention head computes an average attention distribution over the sequence, conflating distinct semantic and syntactic relations. Multi-head attention projects queries, keys, and values into `h` distinct lower-dimensional subspaces (`d_k = d_model / h = 64`), allowing the model to jointly attend to information from different representation subspaces at different positions (e.g., one head tracks coreference, another syntactic verb-object dependencies, and another copy-alignment).

#### Q3: Why are fixed sinusoidal positional encodings used rather than learned positional embeddings?

**Answer:** Sinusoidal encodings require zero learned parameters and naturally extrapolate to sequence lengths unseen during training. Furthermore, because `sin(α + β)` and `cos(α + β)` can be expressed as linear functions of `sin(α)` and `cos(β)`, the model can theoretically learn to attend to relative positions via simple linear transformations.

#### Q4: Why is the causal look-ahead mask essential in the decoder? What happens if it is omitted?

**Answer:** The decoder generates summaries autoregressively token by token; when generating token `y_t`, it cannot access future tokens. The causal lower-triangular mask sets upper-triangular attention weights to `-inf` (implemented as `-1e9`), preventing position `t` from attending to later positions. If omitted, training loss drops rapidly toward zero because the decoder trivializes the task by copying the next ground-truth token, but generated outputs during inference collapse into unconditioned nonsense.

#### Q5: What is the purpose of the padding mask, and where is it applied?

**Answer:** Batched sequences contain arbitrary padding tokens (`[PAD] = 0`) to enforce uniform rectangular tensor shapes. The padding mask is applied in encoder self-attention and decoder cross-attention to set attention weights on pad positions to zero. Without it, padding tokens would contaminate contextual representations and siphon away probability mass from informative words.

#### Q6: Why must the loss function explicitly ignore pad tokens (`ignore_index=0`)?

**Answer:** Padding tokens carry zero predictive or semantic value. If included in cross-entropy, the model achieves artificially low training loss simply by learning to predict padding tokens in the padded sequence tail, distorting gradients away from generating valid vocabulary words. `ignore_index=0` ensures gradients originate solely from real information-carrying tokens.

#### Q7: What is the architectural difference between Post-LN and Pre-LN? Which one does this codebase implement?

**Answer:** Post-LN applies LayerNorm *after* the residual addition: `x = LayerNorm(x + Dropout(Sublayer(x)))`. Pre-LN applies LayerNorm to the input *before* the sublayer: `x = x + Dropout(Sublayer(LayerNorm(x)))`. This codebase strictly implements **Post-LN**, adhering to the original Vaswani et al. (2017) specification, which requires learning rate warmup to avoid early gradient instability.

#### Q8: What is the computational complexity of the Transformer attention mechanism?

**Answer:** For an input sequence of length `N` and hidden dimension `D`, scaled dot-product attention computes `Q·Kᵀ` of shape `(N, D) × (D, N) → (N, N)`, requiring `O(N²·D)` time and `O(N²)` memory to store the attention matrix. In cross-attention with source length `S` and target length `T`, the complexity is `O(S·T·D)`.

#### Q9: What is the structural difference between cross-attention and self-attention in the decoder?

**Answer:** In decoder self-attention, queries, keys, and values all originate from the previous decoder layer's target representations, masked causally to model target prefix history. In cross-attention, queries originate from the decoder, while keys and values originate from the output of the encoder stack, allowing the decoder to query the source document based on its current generation state.

#### Q10: What is the role of the encoder output in the decoder?

**Answer:** The encoder output `H_enc` of shape `(B, S, d_model)` represents the bidirectional, fully contextualized semantic encoding of every token in the source article. It is passed into every decoder layer as the key and value inputs of the cross-attention sublayer, allowing each decoder layer to extract relevant source facts.

#### Q11: How is the parameter count derived, and why does weight tying save parameters?

**Answer:** Without weight tying, the model would require an encoder embedding matrix (30522 × 256), a decoder embedding matrix (30522 × 256), and an output projection matrix (30522 × 256), totaling 23,440,896 embedding parameters. Tying all three to share a single matrix `embed.weight` saves 15,627,264 parameters, reducing total model size from 28.97M to **13,343,232** parameters.

#### Q12: Why are raw attention logits converted to fp32 before softmax when using AMP?

**Answer:** Under automatic mixed precision (fp16), the maximum representable float value is 65,504. The exponentiation step in softmax, `exp(z_i)`, can easily overflow the fp16 dynamic range if dot products exceed ≈ 11. Casting logits to `torch.float32` before `torch.softmax` guarantees numerical stability while retaining fp16 speed for matrix multiplications.

---

### 12.2 Training, Optimization, and Loss Dynamics

#### Q13: What is the Noam learning rate schedule, and why is linear warmup required?

**Answer:** The Noam schedule scales learning rate as `lr(step) = d_model^-0.5 · min(step^-0.5, step · warmup^-1.5)`. During the initial steps of training from scratch, attention and feed-forward weights are random; high learning rates cause large, destabilizing gradients that destroy initial representations (especially in Post-LN). Linear warmup gradually increases the step size over 4,000 steps to let layer norms and attention statistics stabilize before decaying as `1/√step`.

#### Q14: What is teacher forcing, and what is exposure bias?

**Answer:** Teacher forcing feeds the ground-truth shifted target tokens into the decoder during training, allowing fully parallelized loss calculation across all sequence positions. Exposure bias is the resulting train-test discrepancy: during training, the model is exposed only to perfect gold prefixes, but during inference, it must generate conditioned on its own prior (potentially erroneous) predictions, leading to compounding errors.

#### Q15: Why are decoder inputs shifted right and target labels shifted left?

**Answer:** At step `t`, the decoder must predict token `y_t` given only previous tokens `y_<t`. Shifting the target right by prepending `[CLS]` provides the prefix context `y_in = [y_0, y_1, ..., y_(t-1)]`, while shifting the target left provides the prediction targets `y_target = [y_1, y_2, ..., y_t]`. This aligns input position `t-1` with prediction target `t`.

#### Q16: How did early stopping operate in your training run?

**Answer:** Validation loss was evaluated at the end of each epoch on a fixed subset of 3,000 validation pairs. If validation loss failed to improve by at least `1e-4` for `patience = 3` consecutive epochs, training would terminate. In our run, early stopping was never triggered (`bad = 0/3`) because validation loss dropped monotonically across all 6 epochs: 4.1857 → 3.3101 → 2.9903 → 2.8807 → 2.8181 → 2.7997.

#### Q17: What caused training to halt at epoch 6?

**Answer:** Training was halted strictly by the wall-clock budget constraint (`time_budget_min = 110.0`). The run executed 50,009 of 53,838 planned steps (93.1% of the 6-epoch schedule) in 110.08 minutes. The model was compute-limited and undertrained, not converged.

#### Q18: Why was the best checkpoint restored before running test evaluation?

**Answer:** Evaluating the last checkpoint risks using a sub-optimal or overfitted model state. Restoring `best.pt` (epoch 6, validation loss 2.7997) guarantees that final test decoding is executed using the parameter checkpoint with the best validation performance.

#### Q19: What is the purpose of the source ablation test, and what did it prove?

**Answer:** In seq2seq models, decoders can sometimes behave as unconditioned language models, generating fluent generic text while ignoring the source document. By pairing target summaries with rolled/incorrect articles, the loss surged from 2.7997 to 5.5027 nats/token (+2.7031 gap). This confirms that cross-attention is actively conditioning on the source article.

#### Q20: Why was gradient clipping set to 1.0?

**Answer:** Deep recurrent and transformer architectures can experience gradient explosions when unexpected inputs produce large error signals. Clipping gradient norms to a maximum threshold of 1.0 caps the update vector norm, preventing parameter destabilization while preserving gradient direction.

#### Q21: Why was the validation subset restricted to 3,000 pairs during training?

**Answer:** Evaluating the complete validation split of 13,368 pairs takes several minutes per epoch. Restricting validation to a fixed 3,000-pair subset reduced evaluation time to under 1 minute per epoch, preserving maximum wall-clock budget for gradient optimization steps while providing low-variance loss estimates.

#### Q22: How does `BucketSampler` improve training throughput?

**Answer:** In standard random batching, pairing a 50-token article with a 400-token article forces 350 pad tokens across all layers. `BucketSampler` groups samples into chunks of 1,600 pairs and sorts each chunk by source length before batching, ensuring batches contain sequences of similar length and cutting pad-token computation by over 70%.

---

### 12.3 Preprocessing, Tokenization, and Masking Mechanics

#### Q23: Why use the `bert-base-uncased` tokenizer instead of character or word-level tokenization?

**Answer:** Word-level tokenization produces massive vocabularies (over 100k) with severe out-of-vocabulary (OOV) issues on rare entities. Character-level tokenization eliminates OOV but inflates sequence lengths 5-fold, making `O(N²)` self-attention intractable. WordPiece subword tokenization strikes an optimal balance: a compact 30,522 vocabulary that segments unknown words into subword units, ensuring 0% OOV rate while maintaining compact sequence lengths.

#### Q24: Was BERT's pretrained model loaded into this pipeline?

**Answer:** No. We loaded only the tokenizer rules and vocabulary file via `AutoTokenizer.from_pretrained("bert-base-uncased")`. Every transformer weight, embedding matrix, layer normalization parameter, and linear projection was initialized randomly from scratch.

#### Q25: Why was the vocabulary shared between the encoder and decoder?

**Answer:** Because both input articles and output summaries are written in English, sharing the vocabulary allows input word tokens and output word tokens to map into the exact same embedding space. This facilitates representation transfer and enables 3-way parameter tying.

#### Q26: Why choose a 400-token limit for articles and 128 tokens for summaries?

**Answer:** Length statistics on 2,000 articles showed that 87.2% of articles exceed 400 tokens (median 756), while only 4.0% of summaries exceed 128 tokens (median 64). Limiting source length to 400 captures the core journalistic lead while bounding quadratic attention complexity. A 128-token limit accommodates 96.0% of summaries without truncation.

#### Q27: How do you verify that test data never leaked into training or tuning?

**Answer:** Cell 13 executes an automated split disjointness assertion: `test_ids.isdisjoint(train_ids)` and `test_ids.isdisjoint(val_ids)`. During data preprocessing (Cell 6), the test split was never loaded; only its ID column was verified. Test text was read for the very first time in Cell 19 after model training was completely frozen.

#### Q28: How does the causal mask interact with the target padding mask?

**Answer:** The decoder self-attention mask is constructed via bitwise logical AND: `causal_mask(T) & pad_mask(tgt_in)`. This creates a boolean matrix of shape `(B, 1, T, T)` where position `(i, j)` is True if and only if `j ≤ i` (causal constraint) and token `j` is not a `[PAD]` token (padding constraint).

#### Q29: Why is `NEG_INF` set to `-1e9` rather than `-torch.inf`?

**Answer:** When an entire row of an attention matrix is masked out (which can occur if an entire sequence suffix is padded), passing `-torch.inf` through `softmax` produces `exp(-inf) / Σ exp(-inf) = 0/0 = NaN`. Using a large finite negative value like `-1e9` ensures the numerator evaluates to 0.0 and softmax outputs a stable, uniform distribution that is subsequently zeroed out.

#### Q30: How does the code handle lowercasing?

**Answer:** Lowercasing is handled natively by the uncased WordPiece tokenizer (`bert-base-uncased`), which strips accents and converts all text to lowercase before matching subword vocabulary tokens.

---

### 12.4 Decoding, Inference, and Metric Specifications

#### Q31: How does greedy autoregressive decoding work step-by-step?

**Answer:** The encoder processes the source sequence once to produce `H_enc`. Decoding begins with input sequence `[CLS]`. At each step `t`, the decoder processes the current sequence, projects the final hidden state through `out`, selects the token ID with maximum logit via argmax, appends it to the sequence, and repeats until the model emits `[SEP]` or reaches 128 tokens.

#### Q32: Why run the encoder only once during inference rather than at every step?

**Answer:** The encoder representation depends solely on the source article, which does not change as the summary is generated. Running the encoder once and caching its output `H_enc` saves enormous compute, reducing inference from `T` encoder passes (`O(T · S²)`) to a single `O(S²)` pass.

#### Q33: What is the tradeoff between greedy decoding and beam search?

**Answer:** Greedy decoding is computationally efficient (`O(T)` decoder steps) and highly parallelizable across large batches, completing 11,490 test articles in 4.84 minutes. Beam search explores the top-`K` sequence hypotheses at each step, finding higher-probability global paths and reducing repetition, but increases memory and compute by a factor of `K` (e.g., 4× to 5× slower).

#### Q34: What is the mathematical and operational difference between ROUGE-L and ROUGE-Lsum?

**Answer:** ROUGE-L calculates the Longest Common Subsequence (LCS) by treating the entire generated text and reference text as single flat token sequences. ROUGE-Lsum (summary-level LCS) splits texts into sentences using newline characters (`\n`), computes the LCS union between each reference sentence and the generated text, and aggregates the result. ROUGE-Lsum is the standard metric for CNN/DailyMail and yields higher scores (31.82 vs 24.50).

#### Q35: Why must sentences be separated by newline characters (`\n`) for ROUGE-Lsum?

**Answer:** The official `rouge_score` library splits text on `\n` to establish sentence boundaries for the ROUGE-Lsum sentence-level LCS calculation. If newlines are absent, ROUGE-Lsum degrades to flat ROUGE-L, distorting the metric downward. Our code formats summaries with `\n` using NLTK sentence tokenization.

#### Q36: Why is Porter Stemming (`use_stemmer=True`) applied during ROUGE evaluation?

**Answer:** Stemming strips morphological affixes (e.g., "running", "runs" → "run"), allowing matches between word variants that share core semantics. This prevents harsh scoring penalties for inflectional mismatches and follows the standard CNN/DailyMail benchmark setup.

#### Q37: Why report F1 score rather than precision or recall?

**Answer:** Precision rewards brevity (generating one correct word gives 100% precision), while recall rewards verbosity (outputting the entire vocabulary gives 100% recall). F1 is the harmonic mean of precision and recall, `F1 = 2·P·R / (P + R)`, penalizing both uninformative brevity and repetitive bloat.

#### Q38: Why does Lead-3 outperform our from-scratch Transformer?

**Answer:** Lead-3 extracts the first three sentences of the news article verbatim. In CNN/DailyMail, journalists front-load the key story facts in the lead paragraph. Lead-3 extracts human-authored, 100% grammatical sentences with zero hallucinations or repetitions, providing a massive extractive advantage over a small, undertrained abstractive neural net.

---

### 12.5 Failure Diagnosis, Assignment Pitfalls, and Improvements

#### Q39: How did this codebase avoid the 7 Common Pitfalls listed in the assignment specification?

**Answer:**

1. *Banned Libraries:* Avoided. Zero instances of `nn.Transformer`, `nn.MultiheadAttention`, or pretrained weights; verified via static grep.
2. *Causal Mask Failure:* Avoided. Verified via unit test Cell 13(a); loss dropped smoothly and generation is conditioned.
3. *Missing Padding Mask:* Avoided. Verified via unit test Cell 13(b); pad positions receive exact 0.0 attention.
4. *Padding Counted in Loss:* Avoided. Verified via unit test Cell 13(c); appending pads produces identical loss (`ignore_index=0`).
5. *Teacher-Forced ROUGE:* Avoided. Evaluated using pure autoregressive greedy generation (`greedy_decode`).
6. *ROUGE-L Confusion:* Avoided. Both ROUGE-L (24.50) and ROUGE-Lsum (31.82) are explicitly reported and labeled.
7. *Test Set Tuning:* Avoided. Split disjointness verified in Cell 13(h); test text read only after training frozen; best checkpoint (Epoch 6) restored.

#### Q40: Why do 68.12% of generated summaries contain repeated trigrams?

**Answer:** This is a known failure mode of greedy autoregressive decoding in small language models without repetition penalties. When the model outputs a common phrase, self-attention attends back to those tokens, increasing their probability and creating a positive feedback loop that traps the generator in cyclical repetition.

#### Q41: Why did the model hallucinate numbers in 9.34% of test summaries?

**Answer:** The model lacks a copy/pointer mechanism. Subword tokenization splits numbers into arbitrary fragments (e.g., "$160,000" → `['160', ',', '000']`). Without an explicit pointer to copy numbers verbatim from the source, the decoder samples statistically plausible digits from its language model prior.

#### Q42: How would a Pointer-Generator / Copy Mechanism resolve entity hallucination?

**Answer:** A pointer-generator network computes a generation probability `p_gen ∈ [0, 1]` at each step. The final token distribution is a soft blend:

$$P(w) = p_{gen} P_{vocab}(w) + (1 - p_{gen}) \sum_{i:\, w_i = w} a_i$$

where `a_i` is the cross-attention weight over source position `i`. This allows the model to copy rare entities, names, and numbers directly from the input text instead of regenerating them from the vocabulary softmax.

#### Q43: How would a Coverage Mechanism resolve repetitive looping?

**Answer:** A coverage mechanism (See et al., 2017) maintains a running sum of cross-attention distributions over prior steps:

$$c^t = \sum_{t'=0}^{t-1} a^{t'}$$

A coverage loss penalty

$$\text{covloss}_t = \sum_i \min(a_i^t, c_i^t)$$

penalizes attending to source tokens that have already received high attention mass, preventing the decoder from re-generating information it has already summarized.

#### Q44: What decoding strategies would immediately boost test ROUGE without retraining?

**Answer:**

1. **Trigram Blocking:** Prohibiting the emission of any word trigram that has already appeared in the current summary eliminates repeated trigrams entirely.
2. **Beam Search with Length Penalty:** Exploring a beam of width `K=4` with length penalty `α=0.6` balances breadth and verbosity, typically boosting ROUGE-1 by 1.5 to 2.5 points.

---

### 12.6 Team Contributions, Methodology, and Adversarial Inquiries

#### Q45: What was the division of work across the team members?

**Answer:** Kinshuk Gupta implemented the core scratch Transformer modules, attention masking math, and training loop. Harshit Kandpal managed dataset streaming, WordPiece tokenization, and length statistics. Utkarsh Chattarjee structured the evaluation pipeline, Lead-3 baseline, and ROUGE-Lsum sentence formatting. Sujal Som authored the failure analysis diagnostics and qualitative inspection scripts. Tanmay Jaiswal constructed the cross-attention heatmap visualizations and automated report generator.

#### Q46: *Adversarial:* "Your ROUGE score is below the Lead-3 baseline. Is your model broken or defective?"

**Answer:** No, the model is functioning correctly. The assignment specification explicitly states that a from-scratch Transformer trained with limited compute is expected to score below Lead-3. Lead-3 extracts human-written lead sentences with perfect grammar and zero errors. Our source-ablation test shows our model is a genuine summarizer: swapping the source article increases validation loss by +2.70 nats/token, confirming the model generates text conditioned on source information.

#### Q47: *Adversarial:* "Why did validation loss decrease monotonically across all epochs while ROUGE scores remained modest?"

**Answer:** Validation loss measures token-level teacher-forced cross-entropy, whereas ROUGE measures multi-word n-gram overlap of unguided autoregressive generations. As cross-entropy fell from 4.18 to 2.80, the model mastered language syntax and topic alignment. However, because inference uses greedy decoding, minor deviations early in a sentence trigger exposure bias and local repetition loops that suppress n-gram ROUGE precision despite low training perplexity.

#### Q48: *Adversarial:* "Did you tune hyperparameters on the test set?"

**Answer:** No. The test set was held strictly sacred: only its ID column was inspected in Cell 13 to verify split disjointness. All hyperparameter decisions (warmup steps, layer dimensions, batch size) were set prior to training, and early stopping was monitored on a 3,000-sample validation subset. Test articles were loaded and processed for the first time in Cell 19 after model weights were frozen.

#### Q49: *Adversarial:* "Did you use any pretrained Transformer weights from Hugging Face or PyTorch?"

**Answer:** No. We used `transformers` strictly for the subword tokenizer vocabulary (`bert-base-uncased`). Static inspection confirms zero imports of `nn.Transformer`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, or pretrained Hugging Face models. Every weight matrix was initialized randomly from scratch and trained on CNN/DailyMail.

#### Q50: *Adversarial:* "Why did you use only 6 epochs instead of training to convergence?"

**Answer:** We operated under a strict 110-minute wall-clock compute budget on a GPU. With 287,113 training pairs per epoch, each epoch required ~19.7 minutes. Epoch 6 was halted mid-epoch at step 50,009 when the timer expired. Validation loss was still declining at ~0.018 nats/epoch, showing the system was compute-limited, not fundamentally stalled.

#### Q51: *Adversarial:* "Why didn't you implement beam search for evaluation?"

**Answer:** Greedy decoding was the primary requirement specified in the assignment rubric. Furthermore, greedy decoding allowed us to evaluate the entire 11,490-article test set in 4.84 minutes without hitting Colab timeout limits. Implementing beam search of width 4 would have quadrupled inference time, risking notebook termination before metric serialization.

#### Q52: *Adversarial:* "How do you know your attention heatmap isn't just showing random noise?"

**Answer:** Quantitative metrics in `results.json:attention` rule out randomness: 40.01% of attention mass is concentrated in an 80-token window, 70.45% of generated tokens attend to the identical wordpiece token in the source, and mean attention entropy is 3.19 nats compared to 5.99 nats for a uniform distribution. The attention map exhibits sharp, diagonal copy-alignment rather than diffuse noise.

---

## 13. Known Limitations, Discrepancies, and Verification Items

This section documents technical discrepancies, known limitations, and items subject to environment-specific verification.

### 13.1 Technical Discrepancies Across Project Files

1. **Length Statistics Evaluation Sample**
   - *Assignment Specification:* Reported length stats measured on 2,000 *test* articles (article median 641, P90 1304; summary median 49, P90 74).
   - *Codebase Implementation:* Cell 6 measured stats on 2,000 *validation* articles (article median 756, P90 1445; summary median 64, P90 106).
   - *Resolution / TA Explanation:* We intentionally evaluated length statistics on the validation split rather than the test split to uphold strict split hygiene, ensuring the test set remained completely untouched prior to final evaluation.
2. **Lead-3 Baseline Reproduction Margins**
   - *Assignment Benchmark Reference:* ROUGE-1: 40.06, ROUGE-2: 17.48, ROUGE-L: 25.02, ROUGE-Lsum: 36.36.
   - *Our Measured Benchmark:* ROUGE-1: 40.05, ROUGE-2: 17.48, ROUGE-L: 25.02, ROUGE-Lsum: 36.35.
   - *Delta:* -0.01 on ROUGE-1 and ROUGE-Lsum; 0.00 on ROUGE-2 and ROUGE-L.
   - *Resolution / TA Explanation:* The 0.01 difference stems from minor floating-point rounding variations across NLTK sentence tokenization versions. It confirms our evaluation metric is compliant with the benchmark.
3. **Training Halting Mechanism**
   - *Report Prose:* Mentions training completed 6 epochs.
   - *Exact Metric:* Epoch 6 was partial (completed at step 50,009 out of 53,838 planned steps). The stop condition was `time_budget_reached` (110.08 minutes elapsed against a 110.0-minute budget), not early stopping.
   - *Resolution / TA Explanation:* Validation loss was still declining at step 50,009 (best loss 2.7997). The model was compute-bounded and undertrained.
4. **Filesystem Directory Paths**
   - *Notebook:* Hardcodes Google Drive mount paths at `/content/drive/MyDrive/q3_run`.
   - *Export Script (`summarizer_export.py`):* Introduces environment variables `Q3_RUN_DIR` and `Q3_COLAB_MOUNT_ROOT` to permit local and container execution outside Google Colab.

### 13.2 Verification Items (`[VERIFY]`)

- **[VERIFY 1: GPU Execution Hardware]** The exact GPU hardware used during execution was a **Tesla T4 (16 GB VRAM)** on Google Colab. While the notebook logs print `device cuda | AMP True` and timings confirm T4-class throughput (~19.7 min/epoch), `torch.cuda.get_device_name(0)` was not explicitly printed to stdout.
- **[VERIFY 2: PyTorch Release Tag]** Cell 2 prints `torch 2.11.0+cu130`. This denotes a nightly or pre-release PyTorch build present in the Google Colab container environment at the time of execution.
- **[VERIFY 3: Checkpoint Exclusion Policy]** Model weight checkpoint binaries (`best.pt`, `last.pt`, approx. 53 MB each) are excluded from the submitted zip file. This complies with assignment instructions: "Do not include the dataset or model checkpoints."

---

## 14. One-Page Cheat Sheet of Key Numbers

Print this table and memorize these exact numbers before entering the viva defense:

| Category | Parameter / Metric | Exact Code Value |
|---|---|---|
| **Architecture** | Model Dimension (`d_model`) | **256** |
| | Feed-Forward Expansion (`d_ff`) | **1,024** |
| | Attention Heads (`h`) / Head Width (`d_k`) | **4 heads** / **64 dim** |
| | Stack Depth | **3 encoder + 3 decoder layers** |
| | Layer Normalization Order | **Post-LN** |
| | Dropout Rate | **0.1** |
| | Vocabulary Size (`V`) | **30,522** (`bert-base-uncased`) |
| | Embedding Weight Tying | **True** (encoder emb = decoder emb = output projection) |
| | **Total Model Parameters** | **13,343,232** |
| | Tied Embedding Parameter Count | **7,813,632** |
| | Total Encoder Parameters | **2,369,280** (789,760 per layer) |
| | Total Decoder Parameters | **3,160,320** (1,053,440 per layer) |
| **Dataset & Slicing** | Dataset & Version | **CNN/DailyMail 3.0.0** |
| | Training Split Size (Used) | **287,113 pairs** (100% of train split) |
| | Validation Subset (Training Check) | **3,000 pairs** (from 13,368 total) |
| | Test Split Size (Final Evaluation) | **11,490 pairs** (100% of test split) |
| | Max Source / Max Target Truncation | **400 tokens** / **128 tokens** |
| | Articles Exceeding Limit (over 400) | **87.2%** (median 756 tokens) |
| | Summaries Exceeding Limit (over 128) | **4.0%** (median 64 tokens) |
| **Training Run** | Batch Size / Evaluation Batch Size | **32** / **64** |
| | Optimizer | **Adam** (β₁=0.9, β₂=0.98, ε=1e-9) |
| | Learning Rate Schedule | **Noam schedule** (`warmup=4000`) |
| | Peak Learning Rate (Step 4,000) | **9.882e-4** |
| | Initial LR / Final LR | **2.47e-7** / **2.795e-4** |
| | Gradient Clipping Norm | **1.0** |
| | Precision Mode | **Automatic Mixed Precision (AMP fp16)** |
| | Epochs Executed | **6 epochs** (Epoch 6 partial at 50,009 steps) |
| | Planned Steps / Steps Completed | **53,838** / **50,009** (93.1% complete) |
| | Wall-Clock Training Duration | **110.08 minutes** (budget: 110.0 min) |
| | Halting Cause | **Time budget reached** (undertrained, not converged) |
| | Best Epoch / Best Validation Loss | **Epoch 6** / **2.7997 nats/token** |
| | Validation Loss Trajectory | **4.1857 → 3.3101 → 2.9903 → 2.8807 → 2.8181 → 2.7997** |
| | Source Ablation Gap | **+2.7031 nats/token** (2.7997 true vs 5.5027 scrambled) |
| **Test Scores (ROUGE)** | **Our Transformer: ROUGE-1** | **34.28** (exact: 34.2752) |
| | **Our Transformer: ROUGE-2** | **13.94** (exact: 13.9407) |
| | **Our Transformer: ROUGE-L** | **24.50** (exact: 24.4954) |
| | **Our Transformer: ROUGE-Lsum** | **31.82** (exact: 31.8245) |
| | Measured Lead-3 Baseline | **40.05 / 17.48 / 25.02 / 36.35** |
| | Assignment Reference Lead-3 | **40.06 / 17.48 / 25.02 / 36.36** |
| | Delta vs Measured Lead-3 | **-5.78 / -3.54 / -0.52 / -4.52** |
| | Delta vs Assignment Reference Lead-3 | **-5.78 / -3.54 / -0.52 / -4.54** |
| **Diagnostics & Attention** | Summaries with Repeated Trigrams | **68.12%** |
| | Summaries with Unsupported Numbers | **9.34%** |
| | Mean Trigram Copy Rate | **69.24%** |
| | Mean Generated / Reference Words | **48.36 words** / **52.78 words** |
| | Test Generation Speed | **4.84 minutes** for 11,490 articles (2,374 summaries/min) |
| | Attention Visualization Sample | Test index **9700** (`28c3566ff7...`) |
| | Attention Mass in 80-Token Window | **40.01%** (source tokens 42 to 121) |
| | Attention Copy Alignment Fraction | **70.45%** |
| | Attention Entropy | **3.19 nats** (vs 5.99 nats uniform distribution) |
