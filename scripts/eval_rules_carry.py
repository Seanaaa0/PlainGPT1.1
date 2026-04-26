import torch
import numpy as np
import json

from pathlib import Path
from plain_gpt.model import DecoderOnlyLM
from data.gen_rules_carry import (
    TASK_ADD, TASK_SUB, SEP,
    encode_number_4, decode_number_4,
    MAX_NUMBER,
)

CKPT_PATH = "checkpoints/rules_4digit_carry_ss2.pth"


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


def greedy_decode_ans4(model, task_id: int, a: int, b: int):
    """
    prefix decode + constrained decoding:
    只允許輸出 DIGIT_0..DIGIT_9 (token 10..19)，避免答案位跑去選 SEP/TASK。
    """
    # prefix: [TASK, a4, b4, SEP]
    x = [task_id] + encode_number_4(a) + encode_number_4(b) + [SEP]

    # allowed digit token range
    DIGIT_LO = 10
    DIGIT_HI = 19

    for _ in range(4):
        xt = torch.tensor(x, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(xt)                 # (1, T, V)
        last = logits[0, -1, :]                   # (V,)

        # mask: only keep 10..19
        masked = last.clone()
        masked[:DIGIT_LO] = -1e9
        masked[DIGIT_HI + 1:] = -1e9

        next_tok = int(masked.argmax(dim=-1).item())
        x.append(next_tok)

    ans = decode_number_4(x[-4:])
    return ans


def eval_add(model, n_samples=10000, seed=0):
    rng = np.random.default_rng(seed)
    correct = 0
    used = 0

    for _ in range(n_samples):
        # mix easy/hard but ensure a+b<=9999
        if rng.random() < 0.5:
            a = int(rng.integers(0, 1000))
            b = int(rng.integers(0, 1000))
            if a + b > MAX_NUMBER:
                b = MAX_NUMBER - a
        else:
            a = int(rng.integers(0, 10000))
            b = int(rng.integers(0, 10000 - a))
        gt = a + b
        pred = greedy_decode_ans4(model, TASK_ADD, a, b)
        used += 1
        if pred == gt:
            correct += 1

    return correct / used, correct, used


def eval_sub(model, n_samples=10000, seed=1):
    rng = np.random.default_rng(seed)
    correct = 0

    for _ in range(n_samples):
        a = int(rng.integers(0, 10000))
        b = int(rng.integers(0, 10000))
        if a < b:
            a, b = b, a
        gt = a - b
        pred = greedy_decode_ans4(model, TASK_SUB, a, b)
        if pred == gt:
            correct += 1

    return correct / n_samples, correct, n_samples


if __name__ == "__main__":
    print("Loading model...")
    model = load_model()
    print("Model loaded!\n")

    N = 1000
    seeds = [67, 69, 28, 29, 136, 179, 167, 143]

    results = {
        "config": {
            "checkpoint": CKPT_PATH,
            "n_samples_per_seed": N,
            "seeds": seeds
        },
        "add": [],
        "sub": []
    }

    print("=== ADD ===")
    for s in seeds:
        acc, c, n = eval_add(model, n_samples=N, seed=s)
        print(f"[ADD][seed={s:>3}] accuracy = {acc*100:6.2f}% ({c}/{n})")

        results["add"].append({
            "seed": s,
            "accuracy": acc,
            "correct": int(c),
            "total": int(n)
        })

    print("\n=== SUB ===")
    for s in seeds:
        acc, c, n = eval_sub(model, n_samples=N, seed=s)
        print(f"[SUB][seed={s:>3}] accuracy = {acc*100:6.2f}% ({c}/{n})")

        results["sub"].append({
            "seed": s,
            "accuracy": acc,
            "correct": int(c),
            "total": int(n)
        })

    # 🔥 aggregate summary（超重要）
    def summarize(lst):
        accs = [x["accuracy"] for x in lst]
        return {
            "mean": float(np.mean(accs)),
            "min": float(np.min(accs)),
            "max": float(np.max(accs))
        }

    results["summary"] = {
        "add": summarize(results["add"]),
        "sub": summarize(results["sub"])
    }

    # 🔥 輸出
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "eval_summary_1.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {output_path}")
