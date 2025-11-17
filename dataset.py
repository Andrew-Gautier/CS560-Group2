import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any, Tuple


class WeeklySentimentDataset(Dataset):
    """
    Weekly (or N-day) time-series dataset for stock movement prediction.

    Each sample represents one ticker-week window with:
      - x: Tensor [T_i, D] daily features (e.g., FinBERT embedding + numeric features)
      - length: int actual number of days (<= max_seq_len)
      - Optional labels per window: direction (0/1), magnitude (float), confidence (0..1)

    Notes:
      - This class assumes inputs are already preprocessed/aggregated per day.
      - Use the provided collate_weekly function for variable-length batching.
    """

    def __init__(
        self,
        sequences: List[torch.Tensor],  # list of [T_i, D]
        lengths: List[int],
        direction: Optional[torch.Tensor] = None,  # [N] or [N, 1]
        magnitude: Optional[torch.Tensor] = None,  # [N] or [N, 1]
        confidence: Optional[torch.Tensor] = None,  # [N] or [N, 1]
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        assert len(sequences) == len(lengths), "sequences and lengths must align"
        n = len(sequences)
        self.sequences = sequences
        self.lengths = [int(l) for l in lengths]

        def _norm_label(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            t = t.float().view(n, 1) if t.ndim == 1 else t.float()
            assert t.shape[0] == n, "label must match number of sequences"
            return t

        self.direction = _norm_label(direction)
        self.magnitude = _norm_label(magnitude)
        self.confidence = _norm_label(confidence)
        self.metadata = metadata or [{} for _ in range(n)]
        assert len(self.metadata) == n, "metadata length must equal number of samples"

        # Validate feature dims are consistent
        feat_dims = {seq.shape[1] for seq in self.sequences}
        if len(feat_dims) != 1:
            raise ValueError(f"Inconsistent feature dims detected: {feat_dims}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = {
            "x": self.sequences[idx],  # [T_i, D]
            "length": int(self.lengths[idx]),
        }
        if self.direction is not None:
            sample["direction"] = self.direction[idx]
        if self.magnitude is not None:
            sample["magnitude"] = self.magnitude[idx]
        if self.confidence is not None:
            sample["confidence"] = self.confidence[idx]
        sample["meta"] = self.metadata[idx]
        return sample


def collate_weekly(batch: List[Dict[str, Any]], pad_to: Optional[int] = None, pad_value: float = 0.0) -> Dict[str, Any]:
    """
    Collate function to batch variable-length weekly sequences.

    Inputs (per item):
      - x: [T_i, D]
      - length: int
      - optional labels: direction, magnitude, confidence (each [1])

    Output dict:
      - x: [B, T_max, D]
      - lengths: [B]
      - direction/magnitude/confidence: [B, 1] if present else None
      - meta: list of metadata dicts
    """
    # lengths and feature dim
    lengths = torch.tensor([int(b["length"]) for b in batch], dtype=torch.long)
    B = len(batch)
    D = batch[0]["x"].shape[1]
    T_max = int(pad_to or max(int(b["length"]) for b in batch))

    x_padded = torch.full((B, T_max, D), pad_value, dtype=batch[0]["x"].dtype)
    for i, b in enumerate(batch):
        t = int(b["length"])
        x_padded[i, :t] = b["x"][:t]

    def _stack_label(key: str) -> Optional[torch.Tensor]:
        if key not in batch[0]:
            return None
        vals = []
        for b in batch:
            if key in b and b[key] is not None:
                v = b[key]
                v = v.view(1, 1) if v.ndim == 1 else v.view(1, -1)
                vals.append(v)
            else:
                vals.append(torch.full((1, 1), float("nan")))
        return torch.cat(vals, dim=0)

    out = {
        "x": x_padded,
        "lengths": lengths,
        "direction": _stack_label("direction"),
        "magnitude": _stack_label("magnitude"),
        "confidence": _stack_label("confidence"),
        "meta": [b.get("meta", {}) for b in batch],
    }
    return out


def demo_random_dataset(n: int = 8, seq_len: int = 5, dim: int = 768) -> Tuple[WeeklySentimentDataset, Dict[str, Any]]:
    """Utility to create a random dataset for quick tests."""
    import random

    seqs: List[torch.Tensor] = []
    lens: List[int] = []
    dirs: List[float] = []
    mags: List[float] = []
    confs: List[float] = []
    metas: List[Dict[str, Any]] = []

    for i in range(n):
        L = random.choice(list(range(3, seq_len + 1)))
        lens.append(L)
        seqs.append(torch.randn(L, dim))
        dirs.append(float(random.randint(0, 1)))
        mags.append(float(torch.randn(()).item()))
        confs.append(float(torch.sigmoid(torch.randn(())).item()))
        metas.append({"idx": i})

    ds = WeeklySentimentDataset(
        sequences=seqs,
        lengths=lens,
        direction=torch.tensor(dirs),
        magnitude=torch.tensor(mags),
        confidence=torch.tensor(confs),
        metadata=metas,
    )
    return ds, {"n": n, "seq_len": seq_len, "dim": dim}
