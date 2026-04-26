# PlainGPT 1.1

A from-scratch implementation of a decoder-only GPT-style Transformer, designed to study how structured reasoning emerges in small models under autoregressive decoding.

Unlike typical LLM projects, this work focuses on a controlled setting:
multi-digit arithmetic with explicit handling of exposure bias.

No HuggingFace, no pretrained weights — all components are implemented directly in PyTorch.

---

## Key Results

4-digit arithmetic (0–9999), greedy decoding:

| Task                      | Accuracy       |
|---------------------------|---------------|
| Addition (carry)          | 99.5–100%     |
| Subtraction (borrow)      | 99.3–99.6%    |

- Evaluated with pure autoregressive greedy decoding
- No teacher forcing during evaluation
- Stable across multiple random seeds

Example aggregated results:

{
  "add": { "mean": 0.997, "min": 0.995, "max": 1.0 },
  "sub": { "mean": 0.993, "min": 0.990, "max": 0.996 }
}

Full evaluation outputs are saved to:
outputs/eval_summary.json

---

## Problem: Exposure Bias

Arithmetic is a sequence prediction task.

During training:
- model sees correct digits (teacher forcing)

During inference:
- model conditions on its own predictions

This mismatch causes:
- low training loss
- but degraded greedy decoding accuracy (~85–90%)

---

## Solution: Autoregressive Scheduled Sampling (SS2)

PlainGPT 1.1 implements a true autoregressive training loop:

- With probability p_ss:
  - model generates answer digits step-by-step
  - writes predictions back into input
  - computes loss under real inference conditions

This aligns training with inference and improves stability.

---

## Observations

- Addition performs slightly better than subtraction
- Performance drops in long carry / borrow chains
- Indicates model learns structured rules but struggles with deep digit dependencies

---

## Project Structure
```text
PlainGPT/
├── plain_gpt/
│   ├── attention.py
│   ├── model.py
│   ├── lora.py
│   └── data_core.py
│
├── data/
│   ├── gen_rules_data.py
│   └── gen_rules_carry.py
│
├── scripts/
│   ├── train_rules.py
│   ├── eval_rules.py
│   ├── eval_rules_carry.py
│   └── test_rules.py
│
├── outputs/
├── checkpoints/
└── README.md
```
---

## Setup

pip install -r requirements.txt

---

## Data Generation

python data/gen_rules_carry.py

---

## Training

python scripts/train_rules.py --config configs/rules_4digit_ss2.json

---

## Evaluation

python -m scripts.eval_rules_carry

---

## Reproducibility

- No pretrained models used
- All data is synthetic and regenerable
- Results reproducible via scripts

---

## Motivation

This project explores:

- how reasoning emerges in small Transformers
- why low loss does not guarantee correct inference
- how exposure bias affects sequence models
- how training-time rollout improves inference stability

---

## License

MIT
