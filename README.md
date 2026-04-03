# MINI-RAG-QUANT

**Hackathon demo for showing how much model memory drops after quantization and KV-cache optimization.**

## What This Project Is

MINI-RAG-QUANT is a compact end-to-end demo built to answer one main question clearly:

**How much smaller can a useful model become after quantization and inference-memory optimization?**

The diagnosis workflow is the demo surface.
The primary goal is the **before-vs-after memory story**:

- Train a small transformer classifier
- Measure floating-point weight size
- Estimate the post-quantization INT8 weight size
- Estimate autoregressive KV-cache memory and optimized cache variants
- Show that the same workflow can still run with a much smaller model payload

## Hackathon Pitch

Most AI demos focus only on accuracy.
This one focuses on **deployability**:

- Smaller memory footprint
- Lower serving cost
- Easier CPU-friendly deployment
- Lower inference-time memory pressure from KV-cache optimization
- A clear, measurable quantization story judges can understand quickly

## Core Demo Story

This project shows:

1. A compact BPE-based symptom classifier can be trained locally.
2. The model can be evaluated in standard floating-point form.
3. The weight payload can be compared against an INT8 quantized estimate.
4. Decoder-style KV-cache memory can be compared against quantized and sliding-window cache variants.
5. The output workflow still remains usable after we optimize for size.

## Why Diagnosis?

Diagnosis is not the primary product claim here.
It is the **example workload** used to make the quantization story concrete.

It gives us:

- Real text input
- A constrained label space
- Retrieval-assisted grounding
- A visible output that is easy to demo live

## Main Features

- Byte-level BPE tokenizer trained from scratch
- Compact transformer classifier
- Retrieval over similar training cases
- Expanded diagnosis dataset for broader coverage
- Open-set abstention for uncertain queries
- Safety-rule escalation for must-not-miss conditions
- Natural-language assessment output
- Browser UI for demos
- Floating-point vs INT8 memory comparison utilities
- KV-cache memory estimation utilities

## Quantization Focus

The most important part of this repo is in `src/quantize.py`.

It includes:

- `floating_point_weight_bytes(...)`
- `estimated_int8_symmetric_per_tensor_bytes(...)`

These are used to show:

- Approximate model size before quantization
- Approximate model size after symmetric INT8 quantization
- The size reduction ratio

This is the key metric to present during the hackathon.

## KV Cache Optimization Focus

The second optimization story in this repo is **KV-cache memory**.

Why it matters:

- Quantization reduces model weight memory
- KV-cache optimization reduces inference-time memory during autoregressive decoding

This repo includes `src/kv_cache.py`, which estimates:

- Full KV cache in FP32
- Full KV cache in FP16
- Quantized INT8 KV cache
- Sliding-window KV cache in FP16

It also includes a real cached-generation path in `src/model.py`:

- `MiniGPT.generate_with_kv_cache(...)`

This gives you a second hackathon metric:

- **How much smaller does inference-time memory become when the cache is optimized?**

## Project Structure

| Path | Purpose |
| --- | --- |
| `src/main.py` | Training, evaluation, and CLI inference |
| `src/webapp.py` | Browser demo UI |
| `src/quantize.py` | Memory comparison and quantization helpers |
| `src/kv_cache.py` | KV-cache memory estimation helpers |
| `src/model.py` | Compact transformer classifier |
| `src/retrieve.py` | Retrieval over known training cases |
| `src/disease_data.py` | Dataset loading and expansion |
| `src/clinical_support.py` | Safety rules and evidence support |
| `data/` | Cached datasets, tokenizer, metrics |
| `model_diagnosis.pth` | Trained checkpoint |

## Setup

```bash
git clone <your-repo-url>
cd MINI-RAG-QUANT

python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Train The Demo

```bash
python src/main.py --retrain
```

This will:

- Build or refresh the dataset
- Train the tokenizer
- Train the classifier
- Save the checkpoint
- Save evaluation metrics

## Show The Memory Comparison

```bash
python src/main.py --show-memory --show-kv-cache
```

This is the simplest CLI version of the hackathon story.

It prints:

- Floating-point weight size
- Estimated INT8 weight size
- Approximate reduction factor
- Reference decoder KV-cache size in FP32 / FP16
- Quantized KV-cache estimate
- Sliding-window KV-cache estimate

## Run The Browser Demo

```bash
python src/webapp.py
```

Then open:

[http://127.0.0.1:8123](http://127.0.0.1:8123)

The browser demo is designed for presentation.
It highlights:

- FP32 size
- INT8 estimated size
- Reduction ratio
- KV-cache baseline memory
- Quantized KV-cache memory
- Sliding-window KV-cache memory
- The diagnosis workflow as a live example
- Retrieved cases and readable output formatting

## Suggested Live Demo Flow

1. Open the browser UI.
2. Point to the `FP32 Size`, `INT8 Estimate`, and weight `Reduction` first.
3. Then point to the KV-cache section and explain that inference memory can also be reduced.
4. Run one symptom query to show that the optimized system still produces interpretable output.
5. Close by restating both stories: smaller weights and smaller inference cache.

## Example Pitch Line

> We built a compact retrieval-assisted classifier and used it to show two practical deployment wins: model weights shrink after quantization, and inference memory can shrink further with KV-cache optimization.

## Important Note

This repo is for **educational and hackathon demonstration purposes only**.

- It is **not** a medical device
- It is **not** for diagnosis or treatment
- The diagnosis workflow exists to support the quantization and KV-cache demo narrative

## License

This project is licensed under the MIT License.
