# PlainGPT 1.1

A from-scratch implementation of a decoder-only GPT-style Transformer, built to study **how algorithmic reasoning emerges under constrained settings**, with a concrete focus on **multi-digit arithmetic and exposure bias correction**.

This project does **not** rely on HuggingFace, Axolotl, or pretrained checkpoints. All components—including attention, training loop, and scheduled sampling—are implemented directly in PyTorch for maximal transparency and experimental control.

---

##  Key Results

**4-digit arithmetic (0–9999), greedy decoding**

| Task                      | Accuracy       |
| ------------------------- | -------------- |
| Addition (with carry)     | **99.5–100%**  |
| Subtraction (with borrow) | **99.3–99.6%** |

* Evaluation uses **pure greedy autoregressive decoding**
* No teacher forcing during evaluation
* Stable across multiple random seeds

These results indicate that the model learns **generalizable arithmetic rules**, rather than memorizing lookup tables.

---

##  Why This Works

### The Core Problem: Exposure Bias

Multi-digit arithmetic is fundamentally a **sequence generation** problem:

* During training (teacher forcing), the model always sees correct previous digits
* During inference, the model must condition on *its own predictions*

A naïve setup leads to:

* Very low training/validation loss
* Poor greedy decoding accuracy (~85–90%)

This gap is caused by **exposure bias**—a mismatch between training and inference distributions.

---

### The Key Fix: Autoregressive Scheduled Sampling (SS2)

PlainGPT 1.1 implements a **true autoregressive scheduled sampling strategy**:

* During training, with probability `p_ss`:

  * The model generates answer digits **step-by-step**
  * Each predicted digit is written back into the input sequence
  * Loss is computed under the *same conditions as greedy inference*

This directly aligns the training distribution with the inference distribution, effectively correcting exposure bias.

Additionally, the dataset includes **carry / borrow auxiliary supervision**, allowing the model to learn structured digit interactions instead of relying on surface statistics.

---

##  Repository Structure

```
PlainGPT/
├── attention.py          # Multi-head self-attention (from scratch)
├── model.py              # Decoder-only Transformer
├── lora.py               # Optional LoRA utilities
├── data_core.py          # Batch loading utilities
│
├── data/
│   ├── gen_rules_data.py         # Baseline arithmetic (small range)
│   └── gen_rules_carry.py        # 4-digit arithmetic with carry/borrow
│
├── train_rules.py        # Training (autoregressive scheduled sampling)
├── eval_rules.py         # Evaluation (baseline)
├── eval_rules_carry.py   # Evaluation (4-digit carry/borrow)
├── test_rules.py         # Manual sanity tests
│
├── pth/                  # Checkpoints (ignored by git)
└── README.md
```

---

##  Setup

```bash
pip install torch numpy
```

Python ≥ 3.9 is recommended.

---

##  Data Generation

Generate training data for 4-digit arithmetic with carry / borrow:

```bash
python data/gen_rules_carry.py
```

This produces tokenized `.npy` datasets used by `train_rules.py`.

---

##  Training

Train the model using **autoregressive scheduled sampling (SS2)**:

```bash
python train_rules.py
```

Key characteristics:

* Decoder-only Transformer
* Autoregressive rollout during training
* Gradual scheduled sampling ramp-up
* No pretrained weights

Training typically converges to >99% greedy accuracy within ~30k steps on a single GPU.

---

## 📊 Evaluation

Evaluate greedy decoding accuracy:

```bash
python eval_rules_carry.py
```

This script:

* Samples random arithmetic problems
* Uses **pure greedy decoding**
* Reports accuracy across multiple random seeds

---

##  Reproducibility Notes

* No pretrained models are used
* All arithmetic behavior emerges from data and architecture
* Checkpoints are not committed by default
* Results can be reproduced by rerunning data generation and training

---

##  Project Motivation

PlainGPT 1.1 was built to explore:

* How symbolic reasoning emerges in small Transformers
* Why low loss does not guarantee correct autoregressive behavior
* How exposure bias affects algorithmic tasks
* How training-time rollout can dramatically improve inference stability

The project prioritizes **clarity, correctness, and experimental insight** over scale.

---

##  License

MIT License
