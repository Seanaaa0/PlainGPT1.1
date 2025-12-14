import torch

from model import DecoderOnlyLM
from data.gen_rules_carry import (
    TASK_ADD, TASK_SUB, SEP,
    encode_number_4, decode_number_4,
)

CKPT_PATH = "pth/rules_4digit_carry.pth"


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg = ckpt["config"]

    model = DecoderOnlyLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        max_seq_len=cfg["seq_len"],  # 20
        dropout=0.0,
        n_kv_head=cfg["n_head"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def greedy_decode_ans4(model, task_id: int, a: int, b: int) -> int:
    """
    Carry 資料的序列格式 (T=20):
      [TASK, a4, b4, SEP, ans4, SEP, aux4, SEP]

    ans 位於 index 10..13，必須用 greedy 一位一位填回去，
    才不會 teacher-forcing / inference mismatch。
    """
    # 先用 DIGIT_0 當 placeholder（token=10），避免出現非 digit token 影響 context
    DIGIT0 = 10
    x = [task_id] + encode_number_4(a) + encode_number_4(b) + [SEP] + [DIGIT0]*4 + [SEP] + [DIGIT0]*4 + [SEP]
    x = torch.tensor(x, dtype=torch.long).unsqueeze(0)  # (1,20)

    # 逐位生成 ans digits
    for t in range(10, 14):
        with torch.no_grad():
            logits, _ = model(x)
        tok = int(logits[0, t - 1, :].argmax(dim=-1).item())
        x[0, t] = tok

    ans = decode_number_4([int(x[0, i].item()) for i in range(10, 14)])
    return ans


def main():
    print("Loading model...")
    model = load_model()
    print("Model loaded!\n")

    tests = [
        (TASK_ADD, 14, 9, 23),
        (TASK_ADD, 15, 78, 93),
        (TASK_SUB, 16, 13, 3),
        (TASK_SUB, 9999, 1, 9998),
    ]

    for task, a, b, gt in tests:
        pred = greedy_decode_ans4(model, task, a, b)
        op = "+" if task == TASK_ADD else "-"
        print(f"Test: {a} {op} {b} = ?  pred={pred}  gt={gt}")


if __name__ == "__main__":
    main()
