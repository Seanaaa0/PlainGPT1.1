"""
data/gen_rules_carry.py

4-digit (0000~9999) add/sub with auxiliary carry/borrow supervision.

Token space:
  0: PAD (unused)
  1: TASK_ADD
  2: TASK_SUB
  9: SEP
  10~19: DIGIT_0..DIGIT_9

Sequence (fixed length):
  [TASK, a4, b4, SEP, ans4, SEP, aux4, SEP]
  len = 1 + 4 + 4 + 1 + 4 + 1 + 4 + 1 = 20
"""

from pathlib import Path
import numpy as np

PAD = 0
TASK_ADD = 1
TASK_SUB = 2
SEP = 9

DIGIT_OFFSET = 10
VOCAB_SIZE = 20

N_DIGITS = 4
MAX_NUMBER = 9999
SEQ_LEN = 1 + 4 + 4 + 1 + 4 + 1 + 4 + 1  # 20


def encode_digit(d: int) -> int:
    return DIGIT_OFFSET + d


def decode_digit(tid: int) -> int:
    return tid - DIGIT_OFFSET


def encode_number_4(n: int):
    s = f"{n:04d}"
    return [encode_digit(int(ch)) for ch in s]


def decode_number_4(toks):
    ds = [decode_digit(t) for t in toks]
    return ds[0]*1000 + ds[1]*100 + ds[2]*10 + ds[3]


def digits_of(n: int):
    """Return [thousands, hundreds, tens, ones] as ints"""
    s = f"{n:04d}"
    return [int(ch) for ch in s]


def add_with_carry(a: int, b: int):
    """
    Return (ans, carry4)
    carry4[i] is carry-out from position i to i+1, aligned with digits:
      digits index: 0 1 2 3  (th, hu, te, on)
      carry index : 0 1 2 3  (carry from th->overflow, hu->th, te->hu, on->te)
    We store carry bits as [c0,c1,c2,c3] (th..on order) to align with digits.
    Note: If a+b<=9999, overflow carry (c0) is always 0.
    """
    A = digits_of(a)
    B = digits_of(b)

    c = [0, 0, 0, 0]
    out = [0, 0, 0, 0]

    carry_in = 0
    # go from ones->thousands
    for i in range(3, -1, -1):
        s = A[i] + B[i] + carry_in
        out[i] = s % 10
        carry_out = 1 if s >= 10 else 0
        c[i] = carry_out
        carry_in = carry_out

    ans = out[0]*1000 + out[1]*100 + out[2]*10 + out[3]
    # c is aligned with digit index; for thousands carry-out is c[0] (overflow)
    return ans, c


def sub_with_borrow(a: int, b: int):
    """
    Assume a>=b.
    Return (ans, borrow4)
    borrow4[i] is borrow-out from position i to i-1 (to higher digit),
    aligned in th..on order.
      borrow index 0: borrow from thousands (overflow borrow, always 0)
      borrow index 3: borrow from ones -> tens
    """
    A = digits_of(a)
    B = digits_of(b)

    bor = [0, 0, 0, 0]
    out = [0, 0, 0, 0]

    borrow_in = 0
    for i in range(3, -1, -1):
        x = A[i] - borrow_in
        if x < B[i]:
            x += 10
            borrow_out = 1
        else:
            borrow_out = 0
        out[i] = x - B[i]
        bor[i] = borrow_out
        borrow_in = borrow_out

    ans = out[0]*1000 + out[1]*100 + out[2]*10 + out[3]
    return ans, bor


def pack_example(task_id: int, a: int, b: int, ans: int, aux_bits):
    """
    aux_bits: 4 ints in {0,1} aligned th..on
    store aux as DIGIT_0 / DIGIT_1 tokens (still within DIGIT vocab)
    """
    aux_toks = [encode_digit(int(x)) for x in aux_bits]
    return (
        [task_id]
        + encode_number_4(a)
        + encode_number_4(b)
        + [SEP]
        + encode_number_4(ans)
        + [SEP]
        + aux_toks
        + [SEP]
    )


def gen_sequences(n_add: int, n_sub: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    seqs = []

    # ADD: ensure a+b<=9999 to keep 4 digits valid
    for _ in range(n_add):
        # curriculum: mix small and hard
        if rng.random() < 0.5:
            a = int(rng.integers(0, 1000))
            b = int(rng.integers(0, 1000))
            if a + b > MAX_NUMBER:
                b = MAX_NUMBER - a
        else:
            a = int(rng.integers(0, 10000))
            b = int(rng.integers(0, 10000 - a))
        ans, carry = add_with_carry(a, b)
        seqs.append(pack_example(TASK_ADD, a, b, ans, carry))

    # SUB: any a,b, enforce a>=b
    for _ in range(n_sub):
        a = int(rng.integers(0, 10000))
        b = int(rng.integers(0, 10000))
        if a < b:
            a, b = b, a
        ans, bor = sub_with_borrow(a, b)
        seqs.append(pack_example(TASK_SUB, a, b, ans, bor))

    return np.array(seqs, dtype=np.int32)


def flatten(arr2d: np.ndarray):
    return arr2d.reshape(-1)


def build_and_save(
    out_dir="data",
    n_train_add=300000,
    n_train_sub=300000,
    n_val_add=30000,
    n_val_sub=30000,
    seed=1,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train2d = gen_sequences(n_train_add, n_train_sub, seed=seed)
    val2d = gen_sequences(n_val_add, n_val_sub, seed=seed + 1)

    train_ids = flatten(train2d)
    val_ids = flatten(val2d)

    np.save(out / "rules_train_ids.npy", train_ids)
    np.save(out / "rules_val_ids.npy", val_ids)

    print(f"[OK] Saved train: {out/'rules_train_ids.npy'} tokens={len(train_ids)}")
    print(f"[OK] Saved val  : {out/'rules_val_ids.npy'} tokens={len(val_ids)}")
    print(f"[INFO] VOCAB_SIZE={VOCAB_SIZE}  SEQ_LEN={SEQ_LEN}  MAX={MAX_NUMBER}")


if __name__ == "__main__":
    build_and_save()
