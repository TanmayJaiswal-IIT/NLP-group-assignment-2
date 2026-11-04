# Q3 - Abstractive summarization with a from-scratch Transformer (CNN/DailyMail 3.0.0)

## Requirements
Python 3.10+, GPU recommended (Colab/Kaggle T4). `pip install torch datasets transformers rouge_score nltk python-docx matplotlib`
(`transformers` is used for the pretrained `bert-base-uncased` *tokenizer only*; no pretrained weights are loaded).

## Run
1. Open `summarizer.ipynb` on Colab/Kaggle with a GPU runtime and use Run All (about 2-2.5 h on a T4: training budget 110 min).
2. CPU smoke test: set `PRESET = "smoke"` (or env `SUMMARIZER_PRESET=smoke`).
3. Config (model size, TRAIN_SUBSET=`train_subset`, MAX_EPOCHS=`max_epochs`, TIME_BUDGET_MIN=`time_budget_min`, RESUME=`resume`, CKPT_DIR=`ckpt_dir`) lives in the config dataclass of section 2 and `USER_OVERRIDES`.

## Resume after a disconnect
On Colab the notebook mounts Drive and keeps checkpoints, token caches, outputs and the decoding cache in `/content/drive/MyDrive/q3_run`. After a disconnect: reconnect a GPU runtime, Run All, authorise Drive; training continues from the newest checkpoint (written every `ckpt_every` steps, even mid-epoch) and finished decoding batches are reused. Resume needs the same model/data config (checked automatically).

## Files
- `summarizer.ipynb` - full pipeline (data, model, sanity checks, training, evaluation, qualitative, attention, report, zip)
- `summarizer_export.py` - code of the executed cells (if exported)
- `REPORT.docx` - report generated from `results.json`
- `results.json`, `predictions.json` - all numbers and generated test summaries
- `figures/` - loss curves, LR schedule, ROUGE bar chart, attention heatmaps

## Not included
Dataset, Hugging Face cache, tokenised caches and model checkpoints.
