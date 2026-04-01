# mini-rag-quant

A small educational demo that stitches together retrieval-augmented generation (RAG) pieces: deterministic fake embeddings, int8 quantization of those vectors, cosine-style similarity retrieval, and a tiny attention-based `MiniGPT` over the combined query and retrieved text.

## Requirements

- Python 3.10+ recommended
- See `requirements.txt` (PyTorch, NumPy)

## Setup

```bash
cd mini-rag-quant
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Run

From the `src` directory (paths assume `data/docs.txt` is one level up):

```bash
cd src
python main.py
```

This will train the MiniGPT model on the movie plots data (takes some time), then perform retrieval and generation.

If you have a pre-trained model, place `model.pth` in the root directory and modify the code to load it.

## Project layout

| Path | Role |
|------|------|
| `src/main.py` | End-to-end script: load docs, embed, quantize, retrieve, train and run `MiniGPT` |
| `src/embed.py` | Deterministic pseudo-embedding: `torch.randn(8)` seeded by `len(text)` |
| `src/quantize.py` | Per-tensor int8 quantization with scale; `dequantize` for round-trip |
| `src/retrieve.py` | Dot-product scores over doc vectors; returns top document string |
| `src/model.py` | Small `nn.Module` with Q/K/V linear layers and softmax attention |
| `data/docs.txt` | One document per line for retrieval |
| `data/plots.txt` | Training data: movie plots text for the model |

## Behavior notes

- **Embeddings** are not semantic; they are reproducible random 8-D vectors so the pipeline runs without an external model.
- **Quantization** is applied to document embeddings in `main.py`; retrieval still uses the original float vectors (`retrieve` is called with `doc_vecs`).
- **Retrieval** uses dot product (same as cosine for normalized vectors; these vectors are not normalized).

## License

Add a license if you distribute this project.
