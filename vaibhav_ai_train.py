# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 Vaibhav — final power over this project: Vaibhav.
# License default: GPL-3.0-only, NEVER MIT. Ask Vaibhav before public/GitHub.
"""Train Vaibhav 5M market-explainer GPT on CPU. venv-only, AVX2/MKL.

Arch (~4.8M, char-level, block 128):
  vocab(~90-130) x embd256 + pos128x256 + 6x[attn+mlp] + head
  per-block ~0.79M -> total ~4.8M. Verify printed at start.
Opts: AdamW + cosine + clip1.0, threads=12, tqdm bar + file log + --resume.
Measured on YOUR i5-1335U: ~35 tok/s (~30s/step at batch 8).
  --steps 200 --batch 16  ≈ 2-3h (good first model)
  --steps 1000 --batch 16 ≈ overnight
  --steps 3000 --batch 64 ≈ 2+ days — not recommended on CPU
Run (long — run it yourself, my tool times out):
  source ~/ml_env/bin/activate
  nohup python -u vaibhav_ai_train.py --steps 500 --batch 16
    --eval-every 25 > train.out 2>&1 &
  tail -f train.out            # progress bar
  tail -f models/train.log     # survives close
Resume: python -u vaibhav_ai_train.py --steps 1000 --batch 16 --resume
Checker (300k fluency scorer, minutes on CPU):
  nohup python -u vaibhav_ai_train.py --size 300k > check.out 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS = HERE / "models"


class GPTConfig:
    """GPT configuration."""

    def __init__(
        self,
        vocab: int = 128,
        block: int = 128,
        n_layer: int = 6,
        n_head: int = 8,
        n_embd: int = 256,
        drop: float = 0.0,
    ) -> None:
        """Create configuration."""
        self.vocab = vocab
        self.block = block
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.drop = drop


SIZES: dict[str, dict[str, Any]] = {
    # 5m explainer (~4.84M). 300k checker (~0.33M, minutes).
    "5m": {
        "block": 128,
        "n_layer": 6,
        "n_head": 8,
        "n_embd": 256,
        "out": "vaibhav-5m.pt",
        "steps": 3000,
        "batch": 64,
    },
    "300k": {
        "block": 96,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 80,
        "out": "vaibhav-check-300k.pt",
        "steps": 1500,
        "batch": 64,
    },
}


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, cfg: GPTConfig) -> None:
        """Create block."""
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = nn.MultiheadAttention(
            cfg.n_embd, cfg.n_head, dropout=cfg.drop, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.drop),
        )
        size = cfg.block
        self.register_buffer(
            "mask", torch.tril(torch.ones(size, size)).view(1, 1, size, size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        seq_len = inputs.size(1)
        normed = self.ln1(inputs)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=inputs.device), 1,
        ).bool()
        attn_out, _ = self.attn(
            normed, normed, normed, need_weights=False, attn_mask=mask,
        )
        hidden = inputs + attn_out
        return hidden + self.mlp(self.ln2(hidden))


class TinyGPT(nn.Module):
    """Tiny GPT model."""

    def __init__(self, cfg: GPTConfig) -> None:
        """Create model."""
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block, cfg.n_embd)
        self.drop = nn.Dropout(cfg.drop)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run forward pass."""
        _batch, seq_len = idx.shape
        pos = torch.arange(seq_len, device=idx.device)
        hidden = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.head(self.ln(hidden))
        loss = None
        if targets is not None:
            loss = functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new: int = 120,
        temp: float = 0.7,
        top_k: int = 40,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        for _ in range(max_new):
            clipped = idx[:, -self.cfg.block :]
            logits, _ = self(clipped)
            scaled = logits[:, -1, :] / max(0.05, temp)
            if top_k:
                vals, _ = torch.topk(
                    scaled, min(top_k, scaled.size(-1)),
                )
                scaled[scaled < vals[:, [-1]]] = -float("inf")
            probs = functional.softmax(scaled, -1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], 1)
        return idx


def count_params(model: nn.Module) -> int:
    """Count parameters."""
    return sum(p.numel() for p in model.parameters())


def load_corpus() -> tuple[torch.Tensor, torch.Tensor, list[str], dict[str, int]]:
    """Load vocab and encoded corpus."""
    with (DATA / "vocab.json").open(encoding="utf-8") as handle:
        vocab_data = json.load(handle)
    itos: list[str] = vocab_data["itos"]
    stoi: dict[str, int] = vocab_data["stoi"]

    def enc(text: str) -> list[int]:
        return [stoi.get(char, 0) for char in text]

    with (DATA / "train.txt").open(encoding="utf-8") as handle:
        train_text = handle.read()
    with (DATA / "val.txt").open(encoding="utf-8") as handle:
        val_text = handle.read()
    train = torch.tensor(enc(train_text), dtype=torch.long)
    val = torch.tensor(enc(val_text), dtype=torch.long)
    return train, val, itos, stoi


def batch(
    data: torch.Tensor, batch_size: int, seq_len: int, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch."""
    idx = torch.randint(len(data) - seq_len - 1, (batch_size,))
    inputs = torch.stack(
        [data[int(i) : int(i) + seq_len] for i in idx],
    ).to(device)
    targets = torch.stack(
        [data[int(i) + 1 : int(i) + seq_len + 1] for i in idx],
    ).to(device)
    return inputs, targets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--block", type=int, default=None)
    parser.add_argument(
        "--size",
        choices=["5m", "300k"],
        default="5m",
        help="5m explainer or 300k checker (fluency scorer)",
    )
    parser.add_argument(
        "--out", default=None, help="override output filename",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="validate + log every N steps "
        "(progress bar shows train loss every step)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="opt-in torch.compile "
        "(slow first compile on CPU, can look hung)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from models/<out> if present",
    )
    parser.add_argument(
        "--log",
        default="models/train.log",
        help="append run log here "
        "(so long runs survive terminal close)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=12,
        help="CPU threads "
        "(12=max speed, 6=leaves room to use the desk live)",
    )
    return parser.parse_args(argv)


def resolve_log_path(log_arg: str) -> Path:
    """Resolve log path."""
    path = Path(log_arg) if Path(log_arg).is_absolute() else HERE / log_arg
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def make_logger(log_path: Path) -> Callable[[str], None]:
    """Create a logger."""

    def _log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass

    return _log


def get_state_dict(model: nn.Module) -> dict[str, Any]:
    """Get model state dict."""
    inner = getattr(model, "_orig_mod", model)
    return inner.state_dict()


def save_checkpoint(
    model: nn.Module,
    path: Path,
    vocab_size: int,
    block_size: int,
    size_cfg: dict[str, Any],
    step: int,
    val: float,
) -> None:
    """Save checkpoint."""
    torch.save(
        {
            "model": get_state_dict(model),
            "cfg": {
                "vocab": vocab_size,
                "block": block_size,
                "n_layer": size_cfg["n_layer"],
                "n_head": size_cfg["n_head"],
                "n_embd": size_cfg["n_embd"],
            },
            "step": step,
            "val": val,
        },
        path,
    )


def build_model(
    vocab_size: int, block_size: int, size_cfg: dict[str, Any], device: str,
) -> Any:
    """Build model."""
    cfg = GPTConfig(
        vocab=vocab_size,
        block=block_size,
        n_layer=int(size_cfg["n_layer"]),
        n_head=int(size_cfg["n_head"]),
        n_embd=int(size_cfg["n_embd"]),
    )
    return TinyGPT(cfg).to(device)


def compute_lr(step: int, total: int, base: float) -> float:
    """Compute learning rate."""
    warm = max(1, total // 10)
    decay = 0.5 * (1 + math.cos(math.pi * step / total)) + 1e-2
    return base * (min(1.0, step / warm) * decay)


def should_eval(step: int, eval_every: int, total: int) -> bool:
    """Decide evaluation."""
    return step % eval_every == 0 or step in (1, total)


def maybe_compile(
    model: Any, log: Callable[[str], None], *, want: bool,
) -> Any:
    """Compile model optionally."""
    if not want:
        log(
            "torch.compile OFF "
            "(default on CPU for fast start; pass --compile to try it)",
        )
        return model
    try:
        log("torch.compile requested — compiling (first step slow)…")
        compiled = torch.compile(model)
    except Exception as err:
        log(f"compile off ({err})")
        return model
    else:
        log("torch.compile ON")
        return compiled


def maybe_resume(
    model: Any,
    out_name: str,
    log: Callable[[str], None],
) -> tuple[int, float]:
    """Resume checkpoint optionally."""
    start = 1
    best = 1e9
    path = MODELS / out_name
    if not path.exists():
        return start, best
    try:
        ckpt: Any = torch.load(path, map_location="cpu")
        inner = getattr(model, "_orig_mod", model)
        inner.load_state_dict(ckpt["model"])
        start = int(ckpt.get("step", 0)) + 1
        best = float(ckpt.get("val", 1e9))
    except Exception as err:
        log(f"resume failed ({err}), starting fresh")
        return 1, 1e9
    else:
        log(f"resumed from step {start - 1} best val {best:.3f}")
        return start, best


def run_training(
    model: Any,
    opt: Any,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    log: Callable[[str], None],
    start: int,
    initial_best: float,
    total: int,
    batch_size: int,
    block_size: int,
    eval_every: int,
    base_lr: float,
    out_name: str,
    size_label: str,
    vocab_size: int,
    size_cfg: dict[str, Any],
) -> float:
    """Run training loop."""
    device = "cpu"
    toks_per_step = batch_size * block_size
    start_time = time.time()
    last_time = start_time
    model.train()
    cur_train = float("nan")
    cur_val = float("nan")
    best = initial_best
    step = start
    pbar = tqdm(
        total=total,
        initial=start - 1,
        desc=f"train {size_label}",
        unit="step",
        dynamic_ncols=True,
        file=sys.stdout,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    try:
        for step in range(start, total + 1):
            lr = compute_lr(step, total, base_lr)
            for group in opt.param_groups:
                group["lr"] = lr
            inputs, targets = batch(
                train_data, batch_size, block_size, device,
            )
            _, loss = model(inputs, targets)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            cur_train = loss.item()
            now = time.time()
            step_dt = max(1e-6, now - last_time)
            last_time = now
            tok_s = toks_per_step / step_dt
            if should_eval(step, eval_every, total):
                cur_val = eval_once(
                    model,
                    val_data,
                    batch_size,
                    block_size,
                    device,
                )
                elapsed = now - start_time
                log(
                    f"step {step}/{total} train {cur_train:.3f} "
                    f"val {cur_val:.3f} lr {lr:.1e} "
                    f"{tok_s:,.0f} tok/s elapsed {elapsed / 60:.1f}min",
                )
                if cur_val < best:
                    best = cur_val
                    save_checkpoint(
                        model,
                        MODELS / out_name,
                        vocab_size,
                        block_size,
                        size_cfg,
                        step,
                        best,
                    )
                    log(f"  saved BEST val {best:.3f} -> models/{out_name}")
                model.train()
            val_text = (
                f"{cur_val:.3f}" if not math.isnan(cur_val) else "?"
            )
            pbar.set_postfix(
                train=f"{cur_train:.3f}",
                val=val_text,
                lr=f"{lr:.1e}",
                tok_s=f"{tok_s:,.0f}",
            )
            pbar.update(1)
    except KeyboardInterrupt:
        log("interrupted by user — saving last checkpoint…")
        save_last(
            model,
            out_name,
            size_cfg,
            vocab_size,
            block_size,
            step,
            best,
            size_label,
            log,
        )
    finally:
        pbar.close()
    return best


def eval_once(
    model: Any,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
) -> float:
    """Evaluate once."""
    model.eval()
    with torch.no_grad():
        inputs, targets = batch(
            val_data, min(batch_size, 32), block_size, device,
        )
        _, loss = model(inputs, targets)
    return loss.item()


def save_last(
    model: Any,
    out_name: str,
    size_cfg: dict[str, Any],
    vocab_size: int,
    block_size: int,
    step: int,
    best: float,
    size_label: str,
    log: Callable[[str], None],
) -> None:
    """Save last checkpoint."""
    try:
        last_name = out_name.replace(".pt", "-last.pt")
        save_checkpoint(
            model,
            MODELS / last_name,
            vocab_size,
            block_size,
            size_cfg,
            step,
            best,
        )
    except Exception as err:
        log(f"save on interrupt failed: {err}")
    else:
        log(f"saved models/{last_name} (resume with --resume --size {size_label})")


def main() -> None:
    """Train model."""
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)
    device = "cpu"
    MODELS.mkdir(parents=True, exist_ok=True)
    log_path = resolve_log_path(args.log)
    log = make_logger(log_path)
    train_data, val_data, itos, _stoi = load_corpus()
    vocab_size = len(itos)
    size_cfg = SIZES[args.size]
    block_size = args.block or int(size_cfg["block"])
    total = args.steps or int(size_cfg["steps"])
    batch_size = args.batch or int(size_cfg["batch"])
    out_name = str(args.out or size_cfg["out"])
    model = build_model(vocab_size, block_size, size_cfg, device)
    log(
        f"size={args.size} params: {count_params(model) / 1e6:.2f}M "
        f"vocab={vocab_size} block={block_size} "
        f"threads={max(1, args.threads)} device=cpu",
    )
    log(
        f"corpus: train={len(train_data):,} toks "
        f"val={len(val_data):,} toks batch={batch_size} steps={total}",
    )
    model = maybe_compile(model, log, want=bool(args.compile))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    start, best = (1, 1e9)
    if bool(args.resume):
        start, best = maybe_resume(model, out_name, log)
    best = run_training(
        model,
        opt,
        train_data,
        val_data,
        log,
        start,
        best,
        total,
        batch_size,
        block_size,
        int(args.eval_every),
        float(args.lr),
        out_name,
        str(args.size),
        vocab_size,
        size_cfg,
    )
    log(f"done best val {best:.3f} -> {MODELS}/{out_name}")


if __name__ == "__main__":
    main()
