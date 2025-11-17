import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

"""
Time Series Classifier for Reddit comment sentiment to predict next-week stock movement.

Architecture:
- Input: sequence of daily features per ticker (e.g., 5 trading days).
  Include FinBERT embeddings (e.g., 768-d) and optional numeric features
  (e.g., close, volume, sentiment aggregates, engagement features).
- Encoder: Bidirectional LSTM.
- Self-Attention: Additive attention over time steps to obtain a context vector.
- Heads:
  * direction_logit (binary, BCEWithLogits) -> sigmoid gives direction_prob
  * magnitude (regression, MSE)
  * confidence_logit (optional binary-like confidence; if unlabeled, it's predicted only)

Shapes:
- x: FloatTensor [batch, seq_len, input_dim]
- lengths: LongTensor [batch] with actual sequence lengths (optional)
"""


def _sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Create a boolean mask from sequence lengths.
    Returns mask of shape [batch, max_len], True for valid positions.
    """
    batch_size = lengths.size(0)
    max_len = int(max_len or lengths.max().item())
    range_row = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    return range_row < lengths.unsqueeze(1)


class AdditiveSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H: [batch, seq_len, hidden_dim]
        mask: [batch, seq_len] boolean mask (True for valid timesteps)
        Returns:
          context: [batch, hidden_dim]
          attn_weights: [batch, seq_len]
        """
        # score_t = v^T tanh(W h_t)
        scores = self.v(torch.tanh(self.W(self.dropout(H)))).squeeze(-1)  # [B, T]
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=1)  # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), H).squeeze(1)  # [B, H]
        return context, attn_weights


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        attn_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        enc_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attn = AdditiveSelfAttention(enc_out_dim, attn_dim=attn_dim, dropout=dropout)

        # Prediction heads
        self.direction_head = nn.Sequential(
            nn.LayerNorm(enc_out_dim),
            nn.Dropout(dropout),
            nn.Linear(enc_out_dim, 1),  # binary logit
        )
        self.magnitude_head = nn.Sequential(
            nn.LayerNorm(enc_out_dim),
            nn.Dropout(dropout),
            nn.Linear(enc_out_dim, 1),  # regression
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(enc_out_dim),
            nn.Dropout(dropout),
            nn.Linear(enc_out_dim, 1),  # optional binary-like confidence logit
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        x: [B, T, D]
        lengths: [B] (optional). If None, assumes full length.

        Returns dict with:
          - direction_logit: [B, 1]
          - direction_prob: [B, 1]
          - magnitude: [B, 1]
          - confidence_logit: [B, 1]
          - confidence: [B, 1]
          - attn_weights: [B, T]
        """
        B, T, D = x.shape
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        mask = _sequence_mask(lengths, T)

        # Pack for variable-length sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)

        # Self-attention pooling
        context, attn_weights = self.attn(H, mask=mask)

        direction_logit = self.direction_head(context)  # [B,1]
        direction_prob = torch.sigmoid(direction_logit)
        magnitude = self.magnitude_head(context)  # [B,1]
        confidence_logit = self.confidence_head(context)  # [B,1]
        confidence = torch.sigmoid(confidence_logit)

        return {
            "direction_logit": direction_logit,
            "direction_prob": direction_prob,
            "magnitude": magnitude,
            "confidence_logit": confidence_logit,
            "confidence": confidence,
            "attn_weights": attn_weights,
        }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    mae_sum = 0.0
    batches = 0

    for batch in dataloader:
        # Expect batch dict or tuple
        if isinstance(batch, dict):
            x = batch["x"].to(device)
            lengths = batch.get("lengths")
            lengths = lengths.to(device) if lengths is not None else None
            y_dir = batch.get("direction")  # float/bool 0/1
            y_mag = batch.get("magnitude")
        else:
            x, lengths, y_dir, y_mag = batch
            x = x.to(device)
            lengths = lengths.to(device) if lengths is not None else None
            y_dir = y_dir
            y_mag = y_mag

        outputs = model(x, lengths)

        # Direction accuracy if labels present
        if y_dir is not None:
            y_dir = y_dir.to(device).float().view(-1, 1)
            preds = (outputs["direction_prob"] >= 0.5).float()
            correct += (preds == y_dir).sum().item()
            total += y_dir.numel()

        # Magnitude MAE if labels present
        if y_mag is not None:
            y_mag = y_mag.to(device).float().view(-1, 1)
            mae_sum += torch.abs(outputs["magnitude"] - y_mag).mean().item()
            batches += 1

    acc = (correct / total) if total > 0 else float("nan")
    mae = (mae_sum / max(batches, 1)) if batches > 0 else float("nan")
    return {"direction_acc": acc, "magnitude_mae": mae}


def train_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
    loss_weights: Optional[Dict[str, float]] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generic training loop. Expects the dataloader to yield either a dict with keys
    {x, lengths, direction, magnitude} or a tuple (x, lengths, direction, magnitude).
    Computes a weighted sum of losses for the available targets.
    """
    model.to(device)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    loss_weights = loss_weights or {"direction": 1.0, "magnitude": 0.2}

    history = {"loss": []}
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        for batch in dataloader:
            optimizer.zero_grad()

            if isinstance(batch, dict):
                x = batch["x"].to(device)
                lengths = batch.get("lengths")
                lengths = lengths.to(device) if lengths is not None else None
                y_dir = batch.get("direction")
                y_mag = batch.get("magnitude")
                y_conf = batch.get("confidence")
            else:
                # Tuple ordering fallback
                # (x, lengths, direction, magnitude[, confidence])
                x = batch[0].to(device)
                lengths = batch[1] if len(batch) > 1 else None
                lengths = lengths.to(device) if lengths is not None else None
                y_dir = batch[2] if len(batch) > 2 else None
                y_mag = batch[3] if len(batch) > 3 else None
                y_conf = batch[4] if len(batch) > 4 else None

            outputs = model(x, lengths)

            loss = 0.0
            if y_dir is not None:
                y_dir = y_dir.to(device).float().view(-1, 1)
                loss_dir = bce(outputs["direction_logit"], y_dir)
                loss = loss + loss_weights.get("direction", 1.0) * loss_dir
            if y_mag is not None:
                y_mag = y_mag.to(device).float().view(-1, 1)
                loss_mag = mse(outputs["magnitude"], y_mag)
                loss = loss + loss_weights.get("magnitude", 0.2) * loss_mag
            if y_conf is not None:
                y_conf = y_conf.to(device).float().view(-1, 1)
                loss_conf = bce(outputs["confidence_logit"], y_conf)
                loss = loss + loss_weights.get("confidence", 0.1) * loss_conf

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += float(loss.item())
            steps += 1

        mean_loss = epoch_loss / max(steps, 1)
        history["loss"].append(mean_loss)

    return history


def test_model():
    """Quick smoke test with random tensors to validate shapes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, seq_len, input_dim = 4, 5, 768  # FinBERT default hidden size
    x = torch.randn(batch, seq_len, input_dim, device=device)
    lengths = torch.tensor([5, 4, 5, 3], device=device)

    model = LSTMClassifier(input_dim=input_dim)
    model.to(device)
    out = model(x, lengths)
    assert out["direction_logit"].shape == (batch, 1)
    assert out["direction_prob"].shape == (batch, 1)
    assert out["magnitude"].shape == (batch, 1)
    assert out["confidence_logit"].shape == (batch, 1)
    assert out["confidence"].shape == (batch, 1)
    assert out["attn_weights"].shape == (batch, seq_len)
    return {k: v.shape for k, v in out.items()}