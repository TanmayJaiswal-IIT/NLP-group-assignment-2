# Amazon Review Rating Prediction: BiGRU and RoBERTa

**NLP Group Project 2 · Question 2**

## Task, data, and methods

The task is five-class Amazon review rating prediction, where dataset labels 0–4 correspond to one through five stars. Review text is formed from the title and body when those fields are present. The experiment uses 200,000 training, 5,000 validation, and 5,000 test reviews. The validation set controls checkpoint selection and the pooling ablation; the test set is reserved for final evaluation.

The recurrent baseline lowercases and cleans text, builds a training-only vocabulary with minimum frequency 3, initializes a 100-dimensional embedding matrix with GloVe, and caps sequences at 200 tokens. The model is a two-layer bidirectional GRU with additive attention, dropout 0.3, and a five-class output. GloVe embeddings are frozen for epoch one and unfrozen from epoch two. A padding mask is applied in attention, but the GRU processes all fixed-length positions; padded tokens may affect backward hidden states.

RoBERTa uses `RobertaTokenizerFast`, truncation at 256 tokens, and dynamic batch padding. The pretrained `roberta-base` backbone is fine-tuned end to end. Its last hidden states are pooled with an attention-mask mean, then passed through dropout 0.2 and a five-class linear layer. Both models use cross-entropy and seeds 42 and 123. The baseline uses Adam and gradient clipping; RoBERTa uses AdamW (2e-5, weight decay 0.01), 8% linear warmup followed by decay, clipping, and CUDA mixed precision. Checkpoints are selected by validation macro F1.

## Validation pooling ablation

| RoBERTa pooling | Accuracy | Macro F1 | MAE (stars) |
|---|---:|---:|---:|
| Masked mean | 0.5316 | 0.5246 | 0.6082 |
| First token (`<s>`) | 0.5302 | 0.5220 | 0.6102 |

Masked mean pooling has a small validation advantage: +0.0014 accuracy, +0.0026 macro F1, and 0.0020 lower MAE. The difference is modest, but all three metrics favor mean pooling. This ablation is based on validation data only.

## Held-out test results

| Model | Accuracy | Macro F1 | MAE (stars) |
|---|---:|---:|---:|
| BiGRU + additive attention | 0.5684 ± 0.0000 | 0.5699 ± 0.0008 | 0.5191 ± 0.0038 |
| RoBERTa + masked mean | 0.5380 ± 0.0037 | 0.5307 ± 0.0031 | 0.6123 ± 0.0021 |

Per-class F1 (BiGRU): 1★ 0.6423 ± 0.0000; 2★ 0.4733 ± 0.0081; 3★ 0.4751 ± 0.0130; 4★ 0.5222 ± 0.0167; 5★ 0.7368 ± 0.0006. Per-class F1 (RoBERTa): 1★ 0.6421 ± 0.0027; 2★ 0.4344 ± 0.0178; 3★ 0.4007 ± 0.0087; 4★ 0.4749 ± 0.0186; 5★ 0.7014 ± 0.0089. The BiGRU is stronger on overall accuracy, macro F1, MAE, and classes 2–5. RoBERTa is marginally lower on one-star F1.

For seed 42, normalized confusion-matrix diagonal values for ratings 1–5 are 0.60, 0.47, 0.49, 0.53, and 0.76 for the BiGRU, and 0.67, 0.46, 0.36, 0.42, and 0.76 for RoBERTa. Both systems most often confuse adjacent ratings. Exact, adjacent, and distant outcomes are 56.84%, 35.88%, and 7.28% for the BiGRU, versus 53.54%, 35.52%, and 10.94% for RoBERTa. The BiGRU has fewer distant errors in this seed-42 breakdown. The matrices and ordinal percentages use seed 42; headline metrics aggregate both seeds.

## Qualitative analysis and conclusion

The five cases where RoBERTa is correct and the BiGRU is wrong are all one-star reviews predicted as two stars by the baseline. They cover a mixed-positive opening followed by a failed sunflower piercing, an empty package and broken replacement cord, unstitched belt loops, glasses breaking after minimal use, and a battery cover failing on the first day. RoBERTa may be better at weighing a decisive negative outcome after a positive opening or at integrating multiple concrete defects. In the two reverse cases, the baseline correctly identifies a strongly negative review of poor fabric/fit despite a customer-service compliment, and the short complaint “Ran entirely too small.” These examples are illustrative; they do not prove a general model behavior. All listed disagreements are adjacent one-star errors.

The saved interpretability example shows relatively high BiGRU attention on “this,” “product,” “terrible,” and “waste.” RoBERTa gradient saliency varies across subword positions, but the saved token labels are crowded, so precise token-level claims are limited. These visualizations show model sensitivity, not causal explanations.

The BiGRU outperforms RoBERTa in the current run. However, RoBERTa validation macro F1 rises through epoch three for both seeds (0.4666 to 0.5246 and 0.4667 to 0.5253), so its training has not demonstrated convergence. Additional training could improve its score, but the held-out test set should not be reused for tuning. Other limitations include the baseline's un-packed padded sequences, confusion and ordinal summaries from only seed 42, and uncertainty estimated from just two seeds.
