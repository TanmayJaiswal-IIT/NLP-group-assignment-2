# Question 1 — Attention-Based Neural Machine Translation (English to Hindi)

**DSL 504 Natural Language Processing · Group Assignment 2 · Group 14**

Kinshuk Gupta (12341190), Harshit Kandpal (B24CS505), Utkarsh Chattarjee (B24CS050), Sujal Som (B24DS032), Tanmay Jaiswal (B24DS033)

Numbers in this report are the ones produced by `Question1/question_1.ipynb` and saved in `Question1/scratch/results.json`. Both models were trained from random initialization. Test scores use the epoch with the lowest validation loss. Checkpoints were written after that run; they were not loaded to produce the scores.

## Objective

Build an English-to-Hindi neural machine translation system with an LSTM encoder–decoder on the IIT Bombay parallel corpus. Train a baseline that passes only the final encoder state to the decoder, then train a second model that adds Bahdanau attention over every encoder state. Compare the two on the official test split, and use the attention weights to see which English tokens the decoder uses while it writes Hindi.

The question the experiment answers is whether attention improves translation, and whether the attention weights explain what the model is doing.

## Data

Corpus: the official IIT Bombay English–Hindi Parallel Corpus, loaded as `cfilt/iitb-english-hindi`.

Cleaning, applied to every split:

1. Unicode NFKC normalization, strip whitespace, and lowercase English only.
2. Separate punctuation, including the Hindi danda (`।`), so it is its own token.
3. Drop empty pairs.
4. Keep a pair only when the English side contains Latin letters and the Hindi side contains Devanagari.
5. Keep pairs whose tokenized length is between 1 and 25 on both sides.
6. Drop pairs whose longer side is more than 2.5 times the shorter side.

The raw training set has 1,659,083 pairs. After cleaning, 1,216,387 pairs remain. A uniform sample of 40,000 pairs is drawn with seed 42 by reservoir sampling. The official validation split (377 pairs after the same filters) selects the best epoch. The official test split (1,424 pairs after the same filters) is used only for translation scores, heatmaps, and error analysis. Of the 2,507 official test sentences, 1,082 were removed by the length filter.

Vocabularies are built from the training sample only. A word that appears once is mapped to `<unk>`. Special tokens are `<pad>` 0, `<sos>` 1, `<eos>` 2, and `<unk>` 3. Both sides are wrapped as `<sos> ... <eos>` and batches are padded to the longest sentence in that batch.

| | English | Hindi |
|---|---:|---:|
| Vocabulary size | 14,018 | 14,953 |
| Test token OOV rate | 9.29% | 10.50% |
| Longest training sequence, including `<sos>` and `<eos>` | 27 | 27 |

## Models

Both models use a 256-dimensional embedding, a one-layer unidirectional LSTM with hidden size 512, dropout 0.3, and `padding_idx` on the embedding. The encoder packs the batch with `pack_padded_sequence`, so the state passed forward is the state after the last real token.

The baseline compresses the English sentence into the final LSTM state `(h_T, c_T)` and generates Hindi from that state alone. That state has dimension 512 whether the sentence has 4 tokens or 25, so information that does not fit is dropped before decoding starts. Because the LSTM state is dominated by recent tokens, subjects and early modifiers are the ones most likely to be lost. English is subject–verb–object and Hindi is subject–object–verb, and a single final state has no way to look back at an English verb that sits in the middle of the source.

Bahdanau attention keeps every encoder state `h_1, …, h_{T_x}` and builds a new context vector at each Hindi step:

```
e_{t,i} = v_a^T tanh(W_a s_{t-1} + U_a h_i)
α_{t,i} = exp(e_{t,i}) / Σ_j exp(e_{t,j})
c_t = Σ_i α_{t,i} h_i
```

Padding positions are set to a large negative value before the softmax, so they receive zero weight. The decoder LSTM reads the previous Hindi embedding concatenated with `c_t`. The output layer sees the decoder state, the context vector, and the previous embedding together. The initial decoder state is still the encoder’s final state, so the encoder is the same unidirectional LSTM and attention is the added path. The attention dimension is 256. Nothing is taken from a pretrained translation package.

| | Baseline | Attention |
|---|---:|---:|
| Parameters | 18,241,385 | 31,036,265 |

The extra 12.8 million parameters are the attention scorer and the wider decoder input and output layers.

## Training

Training uses full teacher forcing: the decoder input at each step is the gold previous Hindi token. The loss is mean token cross-entropy with padding ignored. Token accuracy is the fraction of non-padding target tokens predicted correctly under teacher forcing. It is a training diagnostic. Translation quality is measured later with greedy decoding and corpus BLEU.

Shared settings: Adam, learning rate 0.001, gradient clip 1.0, batch size 128, 10 epochs, teacher forcing 1.0, seed 42. The checkpoint with the lowest validation loss is restored before test decoding.

| | Baseline | Attention |
|---|---:|---:|
| Best epoch | 5 | 3 |
| Train loss / token accuracy at that epoch | 3.83 / 34.05% | 3.59 / 35.37% |
| Validation loss / token accuracy | 4.86 / 29.20% | 4.71 / 28.59% |
| Wall time | 3.2 min | 9.5 min |

Training loss keeps falling after the best epoch. By epoch 10 the attention training accuracy is 66.17% and the validation loss has risen from 4.71 to 5.09. The baseline shows the same split after epoch 5. Both models overfit the 40,000-pair sample. Restoring the best validation epoch is what keeps the later epochs from being the ones that are scored. Learning curves are in `Question1/figures/model_comparison_curves.png`.

## Test results

Decoding is greedy. The model’s own previous prediction is fed back in. The reference is not used at decode time. Corpus BLEU uses `sacrebleu` with `tokenize="none"`, because both the hypothesis and the reference are already split into the tokens from preprocessing. chrF is reported because Hindi morphology makes exact word match strict. BLEU references are the original Hindi tokens, not the tokens after `<unk>` replacement, so unknown words are not credited.

| | Baseline | Attention | Gain |
|---|---:|---:|---:|
| Test BLEU | 0.72 | 1.29 | +0.57 |
| Test chrF | 5.90 | 7.45 | +1.56 |

Attention is the better translator on this split, by a modest margin. Validation token accuracy does not tell the same story: at the restored checkpoints the baseline is slightly higher (29.20% vs 28.59%). That accuracy is measured while the gold previous word is still being fed. BLEU is measured with greedy decoding. Attention can be the better translator here even though its teacher-forced validation accuracy is not higher.

The absolute scores stay low. Under the frequency-2 vocabulary, about one test token in ten is `<unk>`. A rare one-word pair such as *adaption* / *अनुकूलन* is `<unk>` from both models. An ordinary sentence gets a short, partly right Hindi prefix.

## Performance by sentence length

Test sentences that survived the length filter are grouped by English token count. Each score is corpus BLEU on that subset.

| English tokens | Count | Baseline BLEU | Attention BLEU | Gain |
|---|---:|---:|---:|---:|
| 1–5 | 10 | 1.24 | 2.58 | +1.34 |
| 6–10 | 266 | 0.49 | 1.56 | +1.07 |
| 11–15 | 462 | 1.05 | 1.42 | +0.36 |
| 16–25 | 686 | 0.57 | 1.17 | +0.61 |

Attention BLEU is higher in every bin. The 1–5 bin contains only 10 sentences, so +1.34 is unstable. The reliable gains are larger on the shorter bins (+1.07 on 6–10 tokens, n=266) than on the longer ones (+0.61 on 16–25 tokens, n=686). The bar chart is `Question1/figures/sentence_length_analysis.png`.

Grouped another way, short sources (≤ 10 tokens, n=276) score 0.48 BLEU for the baseline and 1.54 for attention. Long sources (> 10 tokens, n=1,148) score 0.75 and 1.26. The baseline does not get worse as the sentence gets longer inside this cap. Its BLEU rises by 0.27 from the short group to the long group.

That result does not match the simple fixed-vector prediction. Sentences longer than 25 tokens were removed before training, so the “16+” bar is 16–25 tokens, not a long news sentence. Inside that range, unknown words and greedy-decoding errors dominate the score. The length plot shows a consistent attention advantage. It does not show that advantage widening with length.

## Attention heatmaps

Four test sentences were plotted. Every row of a heatmap sums to 1, so the masked softmax is a real distribution over the English tokens. The dark cells are not uniformly on the matching content word. Figures: `Question1/figures/attention_heatmap_1.png` through `attention_heatmap_4.png`.

1. **Short.** *it has always taken place .* → `यह हमेशा <unk> है ।` (token F1 0.67). *यह* puts weight on both *it* (0.33) and *always* (0.42). *हमेशा* puts 0.31 on *always*, and its largest weight is on the period (0.57). From the unknown verb onward, the row collapses onto `.` (0.83 to 0.92). The model stops looking at *taken* and *place*, which is why the Hindi verb is `<unk>` instead of *होती आई*.

2. **Medium.** *a dance program has also been organised after that .* → `एक प्रोग्राम जिसमें <unk> भी हो गया है ।` (token F1 0.59). *प्रोग्राम* is a sharp peak on *program*. That is a real lexical alignment. *एक* is split between *dance* and *program*. Later rows move to *that* and then to the period, and *organised* is almost ignored. The Hindi verb becomes the generic *हो गया है* rather than *रखा गया है*.

3. **Longer.** *i could not believe that she was up there in the first place .* → `मुझे विश्वास नहीं था कि वह पहले ही पहली बात थी ।` (token F1 0.56). *विश्वास* and *नहीं* peak on *that*, not on *believe*. The words are roughly the right meaning, and the alignment is on the complementizer. The second clause is rendered as *पहली बात* (“first matter”), and those rows sit on *place* and `.`. *In the first place* is an idiom. The model treats *place* as a noun. The weights are peaked, and the sense is still wrong.

4. **High overlap.** *and i think about my father .* → `और मैं अपने निर्जन हूँ ।` (token F1 0.57). *और* has a mild peak on *and*. *मैं*, *अपने*, *निर्जन*, *हूँ*, and the danda all peak on the period, with only a side glance at *father*. *think* and *about* receive almost nothing. *निर्जन* (“deserted”) is not licensed by any English word the model is looking at. A high weight on the period is not an alignment.

The repeated pattern is that the first one or two Hindi tokens sometimes land near the right English word, and later tokens fall onto the final punctuation. Attention is moving. It is not a clean monotonic or SVO-to-SOV trace on these sentences.

## Error analysis

Five test sentences were selected for different failure patterns: an unknown English token, a prediction much shorter than the reference, a repeated token, a shared-word order inversion, and a long source with low overlap. Each case is plotted in `Question1/figures/error_analysis_1.png` through `error_analysis_5.png`.

1. **Unknown names.** *meanwhile , former chief reporter neville thurlbeck and former reporter james weatherup oversaw surveillance of ms pawlby ' s movements .* The names are outside the 14,018-word English vocabulary (8 source tokens map to `<unk>`). Both models emit only `<unk>` and a danda. Token F1 is 0.00 for both. The weights walk from left to right and then lock onto the period. Attention cannot recover a name that was replaced before the encoder ran.

2. **Short unknown headline.** *un hails new goals to tackle poverty*. *hails* is `<unk>`. Both outputs are a short string of `<unk>`. Token F1 is 0.00 for both. The first row peaks on *un*, the next rows touch *new* and *goals*, and the rest lock onto *poverty*. The decoder is looking at the content words and still has no Hindi type it will emit. The reference words *यू*, *एन*, *निर्धनता*, and *लक्ष्य* never appear. The short length is the missing-word pattern. The cause is the same unknown-word bottleneck.

3. **Named-entity sentence scored as repetition.** *" the gas tax is just not sustainable , " said lee munnich , a transportation policy expert at the university of minnesota .* The attention output is `<unk>` on every step, which is why the repetition ratio is 1. Token F1 is 0.00 for both. The weights slide across the sentence and then sit on *university* and the period, with no cell above about 0.4 until the end. The baseline emits a stray quote and a `% 1` fragment, then unknowns. That fragment looks like a formatting leftover from the corpus, not a translation of *gas tax*.

4. **Collapsed function-word loop.** *he thwarted the police ' s investigation as much as possible to enable them to continue in their criminal activities .* Both models produce a generic frame (*उन्होंने … के लिए …*) and miss *thwarted*, the police investigation, and the criminal activity. Attention token F1 is 0.09 and baseline token F1 is 0.10. Early cells touch *police* and *investigation*. Later rows put more than half of their mass on the period while *के लिए* repeats. This is omission plus a collapsed attention column. It is not a clean English SVO to Hindi SOV reorder.

5. **Long sentence, frequent-pattern hallucination.** *he got engaged about 8 months ago to the 28 year old woman , francesca pascale , who is 50 years younger than him .* The reference is about an engagement, an age gap, and the name Francesca Pascale. The attention model opens with *उन्होंने कहा कि* (“he said that”), which is not in the source. Token F1 is 0.00 for both. Those rows sit on *woman* for several unknown tokens and finish on *him* and the period. The numbers and the name are dropped. The decoder has fallen back to a high-frequency news frame and parked the attention on one noun.

Across the five cases, attention does not repair unknown names or rare verbs. Where it changes the string at all, it changes it into a short, frequent Hindi frame. The BLEU gain on the full test set comes from partial overlap on ordinary words (*यह*, *है*, *और*, *मुझे*, *विश्वास*), not from solving these sentences.

## Comparative answers

**Did attention improve translation?** Yes, modestly. Test BLEU rises from 0.72 to 1.29 and chrF from 5.90 to 7.45. The gain is carried by partial overlap on common words. The five selected failures stay unsolved.

**Did attention help more on short or long sentences?** It helped on every length bin, and the larger reliable gain is on the shorter sentences. From 6–10 tokens the gain is +1.07 BLEU. From 16–25 tokens it is +0.61 BLEU. The fixed-vector bottleneck is why attention was added. Inside a 25-token cap it is not the main reason the test scores are low. Unknown words and exposure bias hit the sentences that remain.

**What do the heatmaps reveal?** A weak, unstable alignment.

- One clear lexical peak: *प्रोग्राम* attends to *program*.
- A few early rows split their mass across a nearby content word and a function word (*यह* on *it* and *always*, *और* on *and*).
- Later rows in all four examples collapse onto the final period, including rows that generate content words (*निर्जन*, `<unk>`, *हूँ*).
- On the idiom *in the first place*, the peak on *place* lines up with the wrong Hindi sense *बात*.
- On names, the weights still move left to right, and the output is `<unk>`.

There is no consistent off-diagonal pattern of “translate the object, then jump back to the verb.” The row sums are 1, so the mechanism is working. The place it looks is often the end of the sentence.

**Are high attention weights always associated with a correct translation?** No. In example 4 the largest weights are on the period, and the generated word *निर्जन* is unrelated to *father* or *think*. In example 3 the peak on *that* is followed by a reasonable *विश्वास नहीं*, so a high weight on the wrong source word can still sit next to a partly right output. In the named-entity errors the sharpest cells are on `<unk>` source positions or on the period, and the Hindi token is `<unk>`. A high weight means the context vector was mostly that encoder state. It does not mean the output layer chose the reference word.

**Can the visualization explain the model’s behavior?** It explains the alignment. The plots make three behaviors visible that the BLEU number hides: an early token sometimes tracks the source, generation then sticks to the final punctuation, and unknown names produce a left-to-right scan whose outputs are all `<unk>`. They do not explain the output layer. *विश्वास* given a peak on *that*, and *निर्जन* given a peak on the period, are decisions made after the context vector is built. The heatmap is evidence about where the decoder looked. The Hindi token still has to be checked against the reference.

## Limitations

- Word-level vocabularies with a minimum count of 2 leave about one test token in ten unknown. Subword segmentation would remove that ceiling. This run does not use it.
- The training sample is 40,000 pairs drawn from 1,216,387 pairs that passed the filters. Ten epochs overfit that sample. The best validation epochs are early (5 for the baseline, 3 for attention).
- Both models only saw sentences of at most 25 tokens. The length study cannot speak for the 1,082 test sentences that were filtered out.
- Decoding is greedy. There is no beam search, and teacher forcing during training does not match the tokens the decoder sees at test time.
- The attention model has a larger decoder. The comparison holds the encoder, data, optimizer, and seed fixed. It does not hold the parameter count fixed.

## Artifacts

| Artifact | Path |
|---|---|
| Notebook | `Question1/question_1.ipynb` |
| Saved scores | `Question1/scratch/results.json` |
| Baseline checkpoint | `Question1/checkpoints/baseline_model.pt` |
| Attention checkpoint | `Question1/checkpoints/attention_model.pt` |
| Curves, heatmaps, length plot | `Question1/figures/` |
