# mini-rag-quant

This package now trains a real byte-level BPE vocabulary on symptom-to-diagnosis text and uses that tokenizer in a small transformer classifier plus a retrieval layer over known training cases.

The previous movie-plot language-model demo was a poor fit for diagnosis-style prompts, so the main entrypoint has been repointed at a medical-domain dataset pipeline that stays grounded in known labels instead of open-ended text generation.

## What Changed

- Builds a reusable byte-level BPE tokenizer with reserved special tokens and saved metadata.
- Downloads and caches the public `gretelai/symptom_to_diagnosis` dataset locally as JSONL.
- Trains a compact transformer classifier on symptom descriptions.
- Uses BPE-token lexical retrieval over training examples to ground predictions.
- Combines classifier probabilities with retriever scores to reduce unstable outputs.

## Requirements

- Python 3.10+ recommended
- See `requirements.txt` (`torch`, `numpy`, `tokenizers`, `datasets`)

## Setup

```bash
cd mini-rag-quant
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Train And Run

From the `src` directory:

```bash
cd src
python main.py --retrain
```

That command will:

1. Download/cache the symptom-diagnosis dataset into `data/diagnosis_dataset/`
2. Train a BPE tokenizer and save it to `data/diagnosis_bpe_tokenizer.json`
3. Train the diagnosis classifier and save it to `model_diagnosis.pth`
4. Evaluate on the held-out test split
5. Run a prediction for the provided query

You can pass your own symptom query:

```bash
python main.py --query "I have a fever, chills, vomiting, and body pain."
```

If artifacts already exist, the script will load them instead of retraining.

## Project Layout

| Path | Role |
|------|------|
| `src/main.py` | End-to-end training, evaluation, and prediction entrypoint |
| `src/bpe.py` | BPE tokenizer training/loading with metadata and special tokens |
| `src/disease_data.py` | Dataset download, local caching, and label-space generation |
| `src/model.py` | Original `MiniGPT` plus the new `DiagnosisClassifier` |
| `src/retrieve.py` | Retrieval over training cases using BPE-token TF-IDF |
| `data/diagnosis_dataset/` | Cached local copy of the dataset after first run |
| `data/diagnosis_bpe_tokenizer.json` | Saved domain tokenizer |
| `data/diagnosis_bpe_tokenizer.meta.json` | Tokenizer training metadata |
| `data/diagnosis_labels.json` | Ordered diagnosis label list |
| `data/diagnosis_metrics.json` | Saved test metrics |
| `model_diagnosis.pth` | Trained diagnosis classifier checkpoint |

## Notes

- This is still a tiny CPU-friendly educational model, not a production medical system.
- The prediction path is deliberately label-constrained and retrieval-backed to avoid free-form hallucinated answers.
- Outputs should be treated as experimental signals only, not as clinical advice.

## License

Add a license if you distribute this project.
