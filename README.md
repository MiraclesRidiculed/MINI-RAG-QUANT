#  mini-rag-quant

**Lightweight Retrieval-Augmented Diagnosis AI with Tokenizer Training + Quantization Concepts**

---

##  Overview

mini-rag-quant is a compact, end-to-end demonstration of a **grounded AI diagnosis system** built from scratch.

Instead of relying on large pre-trained models, this project:

* Trains a **byte-level BPE tokenizer**
* Builds a **small transformer classifier**
* Adds a **retrieval layer over known cases**
* Combines both to produce **stable, label-constrained predictions**

 The goal:
**Show how intelligent systems can be built efficiently with limited compute while reducing hallucinations.**

---

##  Core Idea

> “Don’t generate blindly — retrieve + constrain + predict.”

This system avoids free-form generation and instead:

* Grounds predictions in **known medical labels**
* Uses **retrieval to stabilize outputs**
* Keeps everything **CPU-friendly and explainable**

---

##  Features

* Byte-level BPE tokenizer (trained from scratch)
* Local dataset caching (symptom → diagnosis)
* Compact transformer classifier
* Retrieval using BPE-token TF-IDF
* Hybrid scoring (classifier + retriever)
* Model + tokenizer persistence
* Evaluation + metrics tracking

---

## Architecture

User Query
→ Tokenization (BPE)
→ Transformer Classifier (probabilities)
→ Retrieval (similar cases)
→ Score Fusion
→ Final Diagnosis Prediction

---

##  Project Structure

| Path                  | Description                        |
| --------------------- | ---------------------------------- |
| `src/main.py`         | Main training + inference pipeline |
| `src/bpe.py`          | Tokenizer training + loading       |
| `src/disease_data.py` | Dataset handling + caching         |
| `src/model.py`        | Transformer + classifier           |
| `src/retrieve.py`     | Retrieval system                   |
| `data/`               | Tokenizer, dataset, metadata       |
| `model_diagnosis.pth` | Trained model                      |

---

##  Setup

```bash
git clone <your-repo-url>
cd mini-rag-quant

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

##  Train & Run

```bash
cd src
python main.py --retrain
```

This will:

* Download dataset
* Train tokenizer
* Train model
* Evaluate performance
* Run prediction

---

##  Example

```bash
python main.py --query "I have fever, chills and body pain"
```

---

## Why This Matters

Most AI systems:
❌ hallucinate
❌ require huge models

This system:
✅ grounded in real data
✅ efficient
✅ explainable
✅ reproducible

---

##  Disclaimer

This project is for **educational and research purposes only**.
It is **not a medical system** and should **not be used for diagnosis or treatment**.
Always consult a qualified healthcare professional.

---

## Hackathon Value

* Demonstrates **LLM fundamentals from scratch**
* Shows **RAG without heavy frameworks**
* Highlights **efficiency + quantization mindset**
* Fully reproducible pipeline

---

## License

This project is licensed under the MIT License.
