"""
data/gen_rules_data.py  (DIGIT tokenizer, 0~9999 add/sub, and optional mul)

Token space:
  0: PAD
  1: TASK_ADD
  2: TASK_SUB
  3: TASK_MUL  (先預留，想做乘法就開)
  10~19: DIGIT_0 ~ DIGIT_9

Number representation: 4 digits with leading zeros.
Example 87 -> 0 0 8 7 (as digit tokens)

Sequence format (fixed length):
  [TASK, a(d4), b(d4), PAD, ans(d4), PAD]
  length = 1 + 4 + 4 + 1 + 4 + 1 = 15
"""

from pathlib import Path
import numpy as np

PAD = 0
TASK_ADD = 1
TASK_SUB = 2
TASK_MUL = 3

DIGIT_OFFSET = 10  # DIGIT_0 = 10 ... DIGIT_9 = 19
VOCAB_SIZE = 20

N_DIGITS = 4
MAX_NUMBER = 9999

SEQ_LEN = 1 + N_DIGITS + N_DIGITS + 1 + N_DIGITS + 1  # 15


def encode_digit(d: int) -> int:
    if not (0 <= d <= 9):
        raise ValueError(d)
    return DIGIT_OFFSET + d


def decode_digit(tid: int) -> int:
    return tid - DIGIT_OFFSET


def encode_number_4(n: int):
    """Encode 0..9999 -> 4 digit tokens (thousands..ones)"""
    if not (0 <= n <= MAX_NUMBER):
        raise ValueError(f"n out of range: {n}")
    s = f"{n:04d}"
    return [encode_digit(int(ch)) for ch in s]


def decode_number_4(toks):
    """Decode 4 digit tokens -> int"""
    ds = [decode_digit(t) for t in toks]
    return ds[0]*1000 + ds[1]*100 + ds[2]*10 + ds[3]


def pack_example(task_id: int, a: int, b: int, ans: int):
    return [task_id] + encode_number_4(a) + encode_number_4(b) + [PAD] + encode_number_4(ans) + [PAD]


def gen_rule_sequences(
    n_add: int = 200000,
    n_sub: int = 200000,
    n_mul: int = 0,          # 想做乘法再開 >0
    seed: int = 1,
):
    rng = np.random.default_rng(seed)
    seqs = []

    # ADD: a,b in 0..9999 but ensure ans <= 9999 (always true for add only if we constrain)
    # 這裡用 curriculum：一半簡單（0~999），一半困難（0~9999 且確保不爆）
    for _ in range(n_add):
        if rng.random() < 0.5:
            a = int(rng.integers(0, 1000))
            b = int(rng.integers(0, 1000))
        else:
            a = int(rng.integers(0, 10000))
            b = int(rng.integers(0, 10000 - a))  # 確保 a+b <= 9999
        ans = a + b
        seqs.append(pack_example(TASK_ADD, a, b, ans))

    # SUB: a,b in 0..9999, ensure non-negative
    for _ in range(n_sub):
        a = int(rng.integers(0, 10000))
        b = int(rng.integers(0, 10000))
        if a < b:
            a, b = b, a
        ans = a - b
        seqs.append(pack_example(TASK_SUB, a, b, ans))

    # MUL: 建議先做 0..99 x 0..99 -> <= 9801（安全）
    for _ in range(n_mul):
        a = int(rng.integers(0, 100))
        b = int(rng.integers(0, 100))
        ans = a * b
        seqs.append(pack_example(TASK_MUL, a, b, ans))

    arr = np.array(seqs, dtype=np.int32)  # (N, SEQ_LEN)
    return arr


def flatten(arr_2d: np.ndarray):
    return arr_2d.reshape(-1)


def build_and_save_datasets(
    out_dir: str = "data",
    n_train_add: int = 200000,
    n_train_sub: int = 200000,
    n_train_mul: int = 0,
    n_val_add: int = 20000,
    n_val_sub: int = 20000,
    n_val_mul: int = 0,
    seed: int = 1,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_2d = gen_rule_sequences(n_add=n_train_add, n_sub=n_train_sub, n_mul=n_train_mul, seed=seed)
    val_2d   = gen_rule_sequences(n_add=n_val_add,   n_sub=n_val_sub,   n_mul=n_val_mul,   seed=seed+1)

    train_ids = flatten(train_2d)
    val_ids = flatten(val_2d)

    np.save(out_path / "rules_train_ids.npy", train_ids)
    np.save(out_path / "rules_val_ids.npy", val_ids)

    print(f"[OK] Saved train: {out_path / 'rules_train_ids.npy'} tokens={len(train_ids)}")
    print(f"[OK] Saved val  : {out_path / 'rules_val_ids.npy'} tokens={len(val_ids)}")
    print(f"[INFO] VOCAB_SIZE = {VOCAB_SIZE}")
    print(f"[INFO] SEQ_LEN    = {SEQ_LEN} (TASK + a4 + b4 + PAD + ans4 + PAD)")
    print(f"[INFO] MAX_NUMBER = {MAX_NUMBER}")


if __name__ == "__main__":
    build_and_save_datasets()
