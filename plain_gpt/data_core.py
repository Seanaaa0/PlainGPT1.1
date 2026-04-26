import torch
import numpy as np


def make_splits(data_ids, split=0.9):
    """
    如果你之後還想用文字版，可以用這個把一條長序列切 train/val。
    在現在這個 rules 專案裡，我們直接用 gen_rules_data 產生好的
    train/val，不一定會用到這個函式。
    """
    n = len(data_ids)
    n_train = int(n * split)
    train_ids = torch.tensor(data_ids[:n_train], dtype=torch.long)
    val_ids = torch.tensor(data_ids[n_train:], dtype=torch.long)
    return train_ids, val_ids


def _to_tensor_slice(arr, start, length):
    """
    舊版滑動視窗用的工具，現在新的 get_batch 不再使用。
    先保留在這裡，以免你之後要拿去做別的實驗。
    """
    if isinstance(arr, np.ndarray):
        view = arr[start:start + length]
        t = torch.from_numpy(view.astype(np.int64, copy=False))
    elif torch.is_tensor(arr):
        t = arr[start:start + length].to(dtype=torch.long)
    else:
        t = torch.tensor(arr[start:start + length], dtype=torch.long)
    return t


def get_batch(data_ids, batch_size, seq_len, device):
    """
    新版本：假設 data_ids 是「flatten 後的一條長序列」，
    但裡面其實是 N 個樣本串起來，每個樣本長度 = seq_len。

    我們的目標是：
    - 每次 batch 抽「整筆樣本」，不跨題目邊界
    - x: [TASK, a, b, PAD, ans, PAD]
    - y: shift 一格：
        y[:, 0:-1] = x[:, 1:]
        y[:, -1]   = x[:, -1]  (最後一個 token 的 target 就讓它學成自己，例如 PAD→PAD)
    """

    # 先把 data_ids 變成 numpy array，方便統一處理
    if torch.is_tensor(data_ids):
        arr = data_ids.detach().cpu().numpy()
    else:
        arr = np.asarray(data_ids)

    total_tokens = arr.shape[0]
    assert total_tokens % seq_len == 0, \
        f"total_tokens={total_tokens} 不能被 seq_len={seq_len} 整除，請確認資料產生邏輯。"

    n_seq = total_tokens // seq_len  # 有幾筆完整樣本
    arr_2d = arr.reshape(n_seq, seq_len)  # (N, T)

    # 隨機抽 batch_size 筆樣本
    idx = np.random.randint(0, n_seq, size=batch_size)
    x_np = arr_2d[idx]  # (B, T)

    # 建立 y：向右 shift 一格，最後一格就讓它學成自己
    y_np = x_np.copy()
    y_np[:, :-1] = x_np[:, 1:]
    y_np[:, -1] = x_np[:, -1]

    # 轉成 tensor 丟進 GPU
    x = torch.from_numpy(x_np).long().to(device, non_blocking=True)
    y = torch.from_numpy(y_np).long().to(device, non_blocking=True)
    return x, y
