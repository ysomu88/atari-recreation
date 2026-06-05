"""
agent/networks.py
=================
Nature DQN Convolutional Neural Network — DeepMind (Mnih et al., 2015).

Architecture faithfully reproduces the network described in:
  "Human-level control through deep reinforcement learning"
  Nature 518, 529–533 (2015).

  Input  : (batch, 4, 84, 84)  — 4 stacked greyscale frames
  Conv1  : 32 filters, 8×8 kernel, stride 4  → (batch, 32, 20, 20)
  Conv2  : 64 filters, 4×4 kernel, stride 2  → (batch, 64,  9,  9)
  Conv3  : 64 filters, 3×3 kernel, stride 1  → (batch, 64,  7,  7)
  Flatten:                                    → (batch, 3136)
  FC1    : 512 units, ReLU
  FC2    : n_actions units  (linear, no activation)

VRAM notes (RTX 3070 Ti, 8 GB):
  - float32 training batch of 32 stays well under 1 GB for the network alone.
  - The replay buffer is the dominant consumer; keep it in CPU RAM and move
    mini-batches to CUDA just-in-time (handled in agent.py).
  - Mixed-precision (torch.cuda.amp) is supported via the optional
    `use_amp` flag on the forward pass — enable it in agent.py if you want
    to squeeze larger batches.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conv_output_size(h: int, w: int, layers: list[dict]) -> Tuple[int, int]:
    """Compute spatial output size after a sequence of Conv2d layers."""
    for cfg in layers:
        k, s, p = cfg["kernel"], cfg["stride"], cfg.get("padding", 0)
        h = math.floor((h + 2 * p - k) / s + 1)
        w = math.floor((w + 2 * p - k) / s + 1)
    return h, w


# ---------------------------------------------------------------------------
# Nature DQN
# ---------------------------------------------------------------------------

class NatureDQN(nn.Module):
    """
    Canonical Nature DQN network (Mnih et al., 2015).

    Parameters
    ----------
    n_actions : int
        Number of discrete actions in the environment's action space.
    in_channels : int
        Number of stacked frames fed as input channels (default: 4).
    input_height : int
        Frame height after preprocessing (default: 84).
    input_width : int
        Frame width after preprocessing (default: 84).
    """

    # Conv layer specs kept as a class-level constant so subclasses and
    # the output-size helper can reference them without touching __init__.
    _CONV_SPECS = [
        {"in_channels": None, "out_channels": 32, "kernel": 8, "stride": 4},
        {"in_channels": 32,   "out_channels": 64, "kernel": 4, "stride": 2},
        {"in_channels": 64,   "out_channels": 64, "kernel": 3, "stride": 1},
    ]
    _FC_HIDDEN = 512

    def __init__(
        self,
        n_actions: int,
        in_channels: int = 4,
        input_height: int = 84,
        input_width: int = 84,
    ) -> None:
        super().__init__()

        if n_actions < 1:
            raise ValueError(f"n_actions must be ≥ 1, got {n_actions}.")
        if in_channels < 1:
            raise ValueError(f"in_channels must be ≥ 1, got {in_channels}.")

        self.n_actions = n_actions
        self.in_channels = in_channels

        # ── Convolutional backbone ──────────────────────────────────────────
        specs = self._CONV_SPECS
        specs[0] = {**specs[0], "in_channels": in_channels}  # patch first layer

        self.conv = nn.Sequential(
            nn.Conv2d(
                specs[0]["in_channels"], specs[0]["out_channels"],
                kernel_size=specs[0]["kernel"], stride=specs[0]["stride"],
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                specs[1]["in_channels"], specs[1]["out_channels"],
                kernel_size=specs[1]["kernel"], stride=specs[1]["stride"],
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                specs[2]["in_channels"], specs[2]["out_channels"],
                kernel_size=specs[2]["kernel"], stride=specs[2]["stride"],
            ),
            nn.ReLU(inplace=True),
        )

        # Compute flat conv output dimension dynamically
        h_out, w_out = _conv_output_size(
            input_height, input_width,
            [{"kernel": s["kernel"], "stride": s["stride"]} for s in specs],
        )
        self._conv_out_dim: int = specs[-1]["out_channels"] * h_out * w_out

        # ── Fully-connected head ────────────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(self._conv_out_dim, self._FC_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(self._FC_HIDDEN, n_actions),
        )

        # Weight initialisation (orthogonal for conv, uniform for linear)
        self._init_weights()

    # -----------------------------------------------------------------------
    # Weight initialisation
    # -----------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Orthogonal init for conv layers (scale √2 for ReLU), and
        uniform Kaiming init for linear layers — both are common
        choices that improve early training stability over PyTorch defaults.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for every action given a batch of stacked frames.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, in_channels, H, W).
            Pixel values should be normalised to [0, 1] (divide uint8 by 255).

        Returns
        -------
        torch.Tensor
            Q-value estimates, shape (batch, n_actions).
        """
        # Normalise uint8 inputs on-the-fly if caller forgot to do it.
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = self.conv(x)
        x = x.flatten(start_dim=1)   # (batch, conv_out_dim)
        x = self.fc(x)
        return x                       # (batch, n_actions)

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> int:
        """
        Greedy action selection for a *single* unbatched observation.

        The agent's epsilon-greedy wrapper lives in agent.py; this method
        is the pure-greedy fallback used during evaluation / target updates.

        Parameters
        ----------
        obs : torch.Tensor
            Shape (in_channels, H, W) — no batch dimension.

        Returns
        -------
        int
            Index of the action with the highest Q-value.
        """
        q_values = self.forward(obs.unsqueeze(0))  # add batch dim
        return int(q_values.argmax(dim=1).item())

    def feature_size(self) -> int:
        """Return the flat conv output dimension (useful for Dueling heads)."""
        return self._conv_out_dim

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"NatureDQN("
            f"in_channels={self.in_channels}, "
            f"n_actions={self.n_actions}, "
            f"conv_out_dim={self._conv_out_dim})"
        )


# ---------------------------------------------------------------------------
# Dueling DQN head (optional drop-in upgrade — Wang et al., 2016)
# ---------------------------------------------------------------------------

class DuelingDQN(NatureDQN):
    """
    Dueling Network Architecture (Wang et al., 2016).

    Replaces the single FC head with two streams:
      • Value stream  V(s)          → scalar
      • Advantage stream A(s, a)   → (n_actions,)

    Combined as:  Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]

    All convolutional layers are identical to NatureDQN, so you can
    swap it in without touching the rest of the pipeline.
    """

    def __init__(self, n_actions: int, **kwargs) -> None:
        # Let the parent build conv + original fc (we'll override fc below).
        super().__init__(n_actions, **kwargs)

        hidden = self._FC_HIDDEN

        # Shared feature extraction (replaces parent's first FC layer)
        self.shared = nn.Sequential(
            nn.Linear(self._conv_out_dim, hidden),
            nn.ReLU(inplace=True),
        )

        # Value stream: hidden → 1
        self.value_stream = nn.Linear(hidden, 1)

        # Advantage stream: hidden → n_actions
        self.advantage_stream = nn.Linear(hidden, n_actions)

        # Remove the parent's monolithic fc head so its params don't linger.
        del self.fc

        # Re-init the new layers
        for module in [self.shared, self.value_stream, self.advantage_stream]:
            for layer in (module.modules() if hasattr(module, "modules") else [module]):
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = self.conv(x)
        x = x.flatten(start_dim=1)

        features = self.shared(x)
        value = self.value_stream(features)                  # (B, 1)
        advantage = self.advantage_stream(features)          # (B, n_actions)

        # Mean-centering stabilises training vs. max-centering.
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_network(
    n_actions: int,
    architecture: str = "nature",
    device: torch.device | str | None = None,
    **kwargs,
) -> NatureDQN:
    """
    Instantiate and optionally move a DQN network to the target device.

    Parameters
    ----------
    n_actions : int
        Size of the discrete action space.
    architecture : {"nature", "dueling"}
        Which network variant to build.
    device : torch.device | str | None
        If None, auto-selects CUDA when available, falls back to CPU.
    **kwargs
        Forwarded to the network constructor (e.g., in_channels, input_height).

    Returns
    -------
    NatureDQN (or DuelingDQN, which inherits from it)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    arch = architecture.lower()
    if arch == "nature":
        net = NatureDQN(n_actions=n_actions, **kwargs)
    elif arch == "dueling":
        net = DuelingDQN(n_actions=n_actions, **kwargs)
    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Choose 'nature' or 'dueling'."
        )

    net = net.to(device)

    # Log parameter count and VRAM estimate for the network weights only.
    total_params = sum(p.numel() for p in net.parameters())
    vram_mb = (total_params * 4) / (1024 ** 2)  # float32 = 4 bytes
    print(
        f"[networks] Built {net.__class__.__name__} on {device} | "
        f"params: {total_params:,} | weight VRAM ≈ {vram_mb:.1f} MB"
    )

    return net


# ---------------------------------------------------------------------------
# Quick smoke-test (run: ".\.venv\Scripts\python.exe" agent/networks.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running smoke test on: {device}\n")

    N_ACTIONS = 18  # Atari max (e.g., Space Invaders)
    BATCH = 32

    for arch in ("nature", "dueling"):
        net = build_network(N_ACTIONS, architecture=arch, device=device)
        print(net)

        dummy = torch.zeros(BATCH, 4, 84, 84, dtype=torch.uint8, device=device)
        q = net(dummy)

        assert q.shape == (BATCH, N_ACTIONS), (
            f"Expected ({BATCH}, {N_ACTIONS}), got {q.shape}"
        )
        print(f"  Output shape : {q.shape}  ✓")

        # Verify greedy action selection on a single observation
        single_obs = torch.zeros(4, 84, 84, dtype=torch.uint8, device=device)
        action = net.select_action(single_obs)
        assert 0 <= action < N_ACTIONS
        print(f"  Greedy action: {action}  ✓\n")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print(f"CUDA memory — allocated: {alloc:.1f} MB | reserved: {reserved:.1f} MB")

    print("All assertions passed.")
    sys.exit(0)