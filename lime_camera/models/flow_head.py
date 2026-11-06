"""Flow-matching pose head.

Output: 9-D pose (3 translation + 6-D rotation, Zhou et al. 2019), standardized.
Conditioning: the backbone's last decoder layer output (B, L, H), plus the noisy
pose x_t (B, 9) and time t (B,).

Architecture (small action expert, ~20M params):
  - SinusoidalTimeEmbed -> MLP -> embed_dim
  - Linear(9 -> embed_dim) on x_t
  - Linear(H -> embed_dim) projection of the hidden states
  - N stacked CrossAttnBlocks (action token cross-attends to the hidden states)
  - LayerNorm -> Linear(embed_dim -> 9)

The flow runs from noise at t=0 to the pose at t=1:
    x_t = (1 - t) * eps + t * pose,  eps ~ N(0, I)

The output means one of two things, set by the `predict_x` flag:
  - predict_x=False: the velocity v, trained with MSE against (pose - eps);
  - predict_x=True:  the clean pose x (Li & He, 2025, "Back to Basics: Let Denoising
    Generative Models Denoise"), trained with MSE against pose. The ODE drift is
    derived from it as v = (x - x_t) / (1 - t).
"""
import math
import torch
import torch.nn as nn


# Output dimensionality: 3 translation + 6-D rotation (first two columns of R).
POSE_DIM = 9


class SinusoidalTimeEmbed(nn.Module):
    """Map scalar t in [0, 1] to (B, dim) sinusoidal features.

    Identical to the standard NeRF/diffusion positional embedding except
    we treat t as a normalized scalar rather than an integer position.
    Each output channel is sin or cos at a logarithmically-spaced frequency
    so a downstream linear can extract whichever time-band it needs.
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, f"SinusoidalTimeEmbed dim must be even, got {dim}"
        self.half = dim // 2
        self._log10000 = math.log(10000.0)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B, 1) -> (B, dim)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)              # (B, 1)
        freqs = torch.exp(
            -self._log10000 * torch.arange(self.half, device=t.device, dtype=t.dtype)
            / self.half
        )                                     # (half,)
        args = t * freqs.unsqueeze(0)         # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention + FFN.

    q   : (B, 1, D)  — the single action token
    kv  : (B, L, D)  — projected VLM hidden states
    Returns: (B, 1, D) updated action token.
    """

    def __init__(self, dim: int, n_heads: int = 8, ffn_mult: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        # batch_first=True so shapes are (B, L, D) everywhere
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim),
            nn.SiLU(),
            nn.Linear(ffn_mult * dim, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                kv_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # cross-attn
        q_norm  = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        # PyTorch MultiheadAttention key_padding_mask: True = position is PADDED.
        attn_out, _ = self.attn(
            q_norm, kv_norm, kv_norm,
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )
        q = q + attn_out
        # FFN
        q = q + self.ffn(self.norm_ffn(q))
        return q


class FlowMatchingPoseHead(nn.Module):
    """Single-pose flow-matching head.

    Args:
        vlm_hidden_size: H of the backbone (e.g. 2560 for Qwen3-VL-4B).
        embed_dim:       internal working dim.
        n_layers:        cross-attention blocks.
        n_heads:         attention heads per block.
        time_embed_dim:  size of the raw sinusoidal embedding before the MLP.
        detach_vlm:      stop the flow loss's gradient at the head boundary, so only
                         the text loss shapes the backbone. No effect at inference.
        predict_x:       the output is the clean pose rather than the velocity; see
                         the module docstring. This changes how the output is read,
                         not the forward graph.
    """

    def __init__(
        self,
        vlm_hidden_size: int,
        embed_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        time_embed_dim: int = 256,
        detach_vlm: bool = False,
        predict_x: bool = False,
    ):
        super().__init__()
        self.pose_dim = POSE_DIM
        self.embed_dim = embed_dim
        self.detach_vlm = bool(detach_vlm)
        self.predict_x  = bool(predict_x)

        # Inputs to the action token.
        self.action_in = nn.Linear(POSE_DIM, embed_dim)
        self.time_in = nn.Sequential(
            SinusoidalTimeEmbed(time_embed_dim),
            nn.Linear(time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Project VLM hidden states down to working dim.
        self.vlm_proj = nn.Linear(vlm_hidden_size, embed_dim)

        # Cross-attention stack.
        self.blocks = nn.ModuleList(
            [CrossAttnBlock(embed_dim, n_heads) for _ in range(n_layers)]
        )

        # Output head.
        self.norm_out = nn.LayerNorm(embed_dim)
        self.action_out = nn.Linear(embed_dim, POSE_DIM)

        # Zero-init the output (DiT trick), so training starts from a neutral head.
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(
        self,
        vlm_hidden: torch.Tensor,    # (B, L, H)
        x_t: torch.Tensor,           # (B, 9)
        t: torch.Tensor,             # (B,) in [0, 1]
        kv_padding_mask: torch.Tensor = None,  # (B, L) bool, True=PADDED
    ) -> torch.Tensor:
        """Return the (B, 9) output: velocity or clean pose, per `predict_x`."""
        if self.detach_vlm:
            vlm_hidden = vlm_hidden.detach()
        kv = self.vlm_proj(vlm_hidden)                 # (B, L, D)

        a = self.action_in(x_t) + self.time_in(t)      # (B, D)
        q = a.unsqueeze(1)                             # (B, 1, D)

        for block in self.blocks:
            q = block(q, kv, kv_padding_mask=kv_padding_mask)

        q = self.norm_out(q.squeeze(1))                # (B, D)
        return self.action_out(q)                      # (B, 9)


@torch.no_grad()
def euler_integrate(
    flow_head: FlowMatchingPoseHead,
    vlm_hidden: torch.Tensor,
    n_steps: int = 10,
    kv_padding_mask: torch.Tensor = None,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Integrate the flow from random noise to a clean pose. Returns (B, 9)."""
    device, dtype = vlm_hidden.device, vlm_hidden.dtype
    B = vlm_hidden.size(0)
    x = torch.randn(B, POSE_DIM, device=device, dtype=dtype, generator=generator)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t_scalar = k * dt
        t = torch.full((B,), t_scalar, device=device, dtype=dtype)
        out = flow_head(vlm_hidden, x, t, kv_padding_mask=kv_padding_mask)
        if flow_head.predict_x:
            # Clamp (1 - t) away from 0 to avoid the singularity at t -> 1.
            v = (out - x) / max(1.0 - t_scalar, 1e-3)
        else:
            v = out
        x = x + dt * v
    # Under x-prediction the last forward pass is the cleanest read of the pose: it
    # has the highest signal-to-noise, and using it skips the last Euler step.
    return out if flow_head.predict_x else x
