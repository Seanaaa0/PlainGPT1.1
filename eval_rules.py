import torch
import numpy as np

from model import DecoderOnlyLM
from data.gen_rules_data import (
    VOCAB_SIZE, SEQ_LEN,
    TASK_ADD, TASK_SUB, PAD,
    encode_number, decode_number,
    MAX_NUMBER,
)

CKPT_PATH = "pth/rules_small.pth"


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg = ckpt["config"]

    model = DecoderOnlyLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        max_seq_len=cfg["seq_len"],
        dropout=0.0,
        n_kv_head=cfg["n_head"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def predict_answer_token(model, task_id, a, b):
    """
    x = [TASK, a, b, PAD, PAD, PAD]
    使用 logits 在 index=3 預測 index=4 的答案
    """
    x = torch.tensor([
        task_id,
        encode_number(a),
        encode_number(b),
        PAD,
        PAD,
        PAD,
    ], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(x)
        logits_pos = logits[0, 3, :]
        tok_id = int(logits_pos.argmax(dim=-1).item())

    return decode_number(tok_id)


def eval_add(model, n_samples=1000, seed=0):
    rng = np.random.default_rng(seed)
    correct = 0
    used = 0

    for _ in range(n_samples):
        a = int(rng.integers(0, 50))
        b = int(rng.integers(0, 50))
        gt = a + b
        if gt > MAX_NUMBER:
            continue
        pred = predict_answer_token(model, TASK_ADD, a, b)
        used += 1
        if pred == gt:
            correct += 1

    acc = correct / used
    return acc, correct, used


def eval_sub(model, n_samples=1000, seed=0):
    rng = np.random.default_rng(seed)
    correct = 0

    for _ in range(n_samples):
        a = int(rng.integers(0, 100))
        b = int(rng.integers(0, 100))
        if a < b:
            a, b = b, a
        gt = a - b
        pred = predict_answer_token(model, TASK_SUB, a, b)
        if pred == gt:
            correct += 1

    acc = correct / n_samples
    return acc, correct, n_samples


if __name__ == "__main__":
    print("Loading model...")
    model = load_model()
    print("Model loaded!\n")

    seeds = [0, 1, 2, 3, 7, 13, 42, 99, 123, 999]

    print("=== ADD (different seeds) ===")
    for s in seeds:
        acc, c, n = eval_add(model, seed=s)
        print(f"[ADD][seed={s:>3}] accuracy = {acc*100:6.2f}% ({c}/{n})")

    print("\n=== SUB (different seeds) ===")
    for s in seeds:
        acc, c, n = eval_sub(model, seed=s)
        print(f"[SUB][seed={s:>3}] accuracy = {acc*100:6.2f}% ({c}/{n})")
