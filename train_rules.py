import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import DecoderOnlyLM
from data_core import get_batch
from data.gen_rules_carry import VOCAB_SIZE, SEQ_LEN


def loss_ans_and_aux(model, x: torch.Tensor) -> torch.Tensor:
    """
    Sequence index map (T=20):
      0: TASK
      1..4: a digits
      5..8: b digits
      9: SEP
      10..13: ans digits     (targets)
      14: SEP
      15..18: aux digits     (carry/borrow bits, targets)
      19: SEP

    next-token LM: token at t is predicted by logits at t-1.
    So we train targets t in {10..13, 15..18}.
    """
    logits, _ = model(x)  # (B, T, V)
    targets = list(range(10, 14)) + list(range(15, 19))
    losses = []
    for t in targets:
        losses.append(F.cross_entropy(logits[:, t - 1, :], x[:, t]))
    return sum(losses) / len(losses)


@torch.no_grad()
def rollout_ans4_autoreg(model, x: torch.Tensor) -> torch.Tensor:
    """
    真正修 exposure bias 的關鍵：
    用自回歸方式逐位產生 ans4，每次預測一位就寫回 x，讓下一位受到前一位的影響。

    x: (B, T=20) teacher-forcing 的完整序列（包含真實 ans/aux）
    return:
      x_roll: (B, T=20) 其中 ans4(10..13) 已被模型自回歸產生的 token 覆寫
    """
    x_roll = x.clone()
    # 逐位生成 ans digits：t=10..13，分別由 logits[t-1] 預測
    for t in range(10, 14):
        logits, _ = model(x_roll)  # (B, T, V)
        tok = logits[:, t - 1, :].argmax(dim=-1)  # (B,)
        x_roll[:, t] = tok
    return x_roll


def eval_split(model, data_ids, cfg, device, tag="val"):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(cfg["eval_batches"]):
            x, _ = get_batch(data_ids, cfg["batch_size"], cfg["seq_len"], device)
            losses.append(loss_ans_and_aux(model, x).item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    print(f"[{tag}] loss={mean_loss:.4f}, ppl≈{math.exp(mean_loss):.1f}")
    return mean_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")

    cfg = {
        "train_ids_path": "data/rules_train_ids.npy",
        "val_ids_path": "data/rules_val_ids.npy",
        "vocab_size": VOCAB_SIZE,
        "d_model": 384,
        "n_layer": 6,
        "n_head": 6,
        "seq_len": SEQ_LEN,  # 20
        "batch_size": 128,
        "lr": 3e-4,

        # 建議給 SS2 多一點步數
        "max_steps": 35000,
        "log_interval": 100,
        "eval_interval": 500,
        "eval_batches": 50,

        "ckpt_dir": "pth",
        "ckpt_name": "rules_4digit_carry_ss2.pth",

        # Scheduled sampling (SS2) 設定：穩定為主
        "ss_warmup_steps": 5000,
        "ss_ramp_steps": 20000,
        "ss_max_p": 0.20,
    }

    Path(cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

    train_ids = np.load(cfg["train_ids_path"], mmap_mode="r")
    val_ids = np.load(cfg["val_ids_path"], mmap_mode="r")
    print(f"[info] train_tokens={len(train_ids)} val_tokens={len(val_ids)}")

    model = DecoderOnlyLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        max_seq_len=cfg["seq_len"],
        dropout=0.0,
        n_kv_head=cfg["n_head"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    def current_p_ss(step: int) -> float:
        if step <= cfg["ss_warmup_steps"]:
            return 0.0
        s = step - cfg["ss_warmup_steps"]
        if s >= cfg["ss_ramp_steps"]:
            return cfg["ss_max_p"]
        return cfg["ss_max_p"] * (s / cfg["ss_ramp_steps"])

    best_val = float("inf")
    t0 = time.time()

    print("[info] start training (SS2 autoreg rollout)")
    for step in range(1, cfg["max_steps"] + 1):
        x, _ = get_batch(train_ids, cfg["batch_size"], cfg["seq_len"], device)

        p_ss = current_p_ss(step)

        # 以機率 p_ss 使用 rollout 版本（ans4 用模型逐位產生）來算 loss
        if p_ss > 0.0 and torch.rand((), device=device).item() < p_ss:
            with torch.no_grad():
                x_roll = rollout_ans4_autoreg(model, x)
            loss = loss_ans_and_aux(model, x_roll)
        else:
            loss = loss_ans_and_aux(model, x)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg["log_interval"] == 0:
            dt = time.time() - t0
            print(f"[step {step}] loss={loss.item():.4f}  p_ss={p_ss:.3f}  ({dt:.1f}s)")
            t0 = time.time()

        if step % cfg["eval_interval"] == 0:
            val_loss = eval_split(model, val_ids, cfg, device, tag="val")
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = Path(cfg["ckpt_dir"]) / cfg["ckpt_name"]
                torch.save(
                    {"model": model.state_dict(), "config": cfg, "step": step, "best_val": best_val},
                    ckpt_path,
                )
                print(f"[ckpt] saved best to {ckpt_path} (val_loss={best_val:.4f})")

    print("[done] training finished")


if __name__ == "__main__":
    main()
