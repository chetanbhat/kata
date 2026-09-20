"""Modular training-loop utility.

Supports gradient clipping by global norm, gradient accumulation over
micro-batches, mixed-precision (AMP) scaling hooks, and optional scheduler
stepping. On CPU-only hosts AMP degrades gracefully to full precision.
"""

import torch
import torch.nn as nn


def _grad_scaler(use_amp: bool, device: str):
    if use_amp and str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.amp.GradScaler("cuda")
    return None  # CPU: AMP hooks become no-ops


def train_loop(model: nn.Module, loss_fn, optimizer, X: torch.Tensor, y: torch.Tensor,
               epochs: int = 10, batch_size: int | None = None, max_grad_norm: float | None = None,
               accumulation_steps: int = 1, use_amp: bool = False, scheduler=None,
               device: str = "cpu") -> dict:
    """Full training loop over (X, y); returns {'loss': [per-epoch means]}."""
    model.to(device)
    scaler = _grad_scaler(use_amp, device)
    amp_dtype = torch.float16 if str(device).startswith("cuda") else torch.bfloat16
    n = X.shape[0]
    bs = batch_size or n
    history = {"loss": []}

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss, seen = 0.0, 0
        for start in range(0, n, bs * accumulation_steps):
            micro_losses = []
            for m in range(start, min(start + bs * accumulation_steps, n), bs):
                xb, yb = X[m:m + bs].to(device), y[m:m + bs].to(device)
                with torch.amp.autocast(device_type="cpu" if device == "cpu" else "cuda",
                                        enabled=scaler is not None, dtype=amp_dtype):
                    loss = loss_fn(model(xb), yb) / accumulation_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                micro_losses.append(float(loss.detach()) * accumulation_steps)
            if max_grad_norm is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer) if hasattr(scaler, "unscale_") else None
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            epoch_loss += sum(micro_losses)
            seen += len(micro_losses)
        if scheduler is not None:
            scheduler.step()
        history["loss"].append(epoch_loss / max(1, seen))
    return history
