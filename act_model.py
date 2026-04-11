"""
State-Only Action Chunking Transformer (ACT) for LeKiwi Robot.

Architecture (following Zhao et al., 2023 — without vision backbone):
  - CVAE Encoder: Transformer that compresses (obs, action_chunk) → latent z
  - Policy Decoder: Transformer encoder processes (obs_embed, z_token),
    Transformer decoder with learned action queries outputs action chunk
  - At inference: z = 0 (prior mean), temporal ensembling over overlapping chunks

Designed for state-based observations (30D Skill-2 / 29D Skill-3) and
9D actions [arm5, grip1, base3] in v6 ordering.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Positional encoding (sinusoidal)
# ---------------------------------------------------------------------------
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# CVAE Encoder — only used during training
# ---------------------------------------------------------------------------
class CVAEEncoder(nn.Module):
    """
    Compresses (obs, action_sequence) → latent z.
    Architecture: [CLS] + obs_token + action_tokens → Transformer → CLS → (mu, logvar)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        latent_dim: int,
        chunk_size: int,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Project obs and actions to hidden_dim
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # Positional encoding for the sequence: [CLS, obs, action_0, ..., action_{k-1}]
        self.pos_enc = SinusoidalPositionEncoding(hidden_dim, max_len=chunk_size + 2)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Latent projection
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs:     (B, obs_dim)
            actions: (B, chunk_size, action_dim)
        Returns:
            mu:     (B, latent_dim)
            logvar: (B, latent_dim)
        """
        B = obs.shape[0]

        # Project
        obs_emb = self.obs_proj(obs).unsqueeze(1)  # (B, 1, H)
        act_emb = self.action_proj(actions)  # (B, K, H)

        # Build sequence: [CLS, obs, actions]
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, H)
        seq = torch.cat([cls, obs_emb, act_emb], dim=1)  # (B, 2+K, H)
        seq = self.pos_enc(seq)

        # Transformer
        out = self.transformer(seq)  # (B, 2+K, H)

        # CLS token output → latent
        cls_out = out[:, 0]  # (B, H)
        mu = self.fc_mu(cls_out)
        logvar = self.fc_logvar(cls_out)
        return mu, logvar


# ---------------------------------------------------------------------------
# Policy Decoder — the actual policy
# ---------------------------------------------------------------------------
class PolicyDecoder(nn.Module):
    """
    Given obs and latent z, predict action chunk.
    Architecture:
      Encoder: processes [z_token, obs_token] → context
      Decoder: learned action queries + cross-attention with context → action chunk
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        latent_dim: int,
        chunk_size: int,
        n_heads: int = 4,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        # Input projections
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # Positional encoding for encoder inputs
        self.enc_pos = SinusoidalPositionEncoding(hidden_dim, max_len=16)

        # Transformer encoder (processes obs + z context)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        # Learnable action queries (DETR-style)
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))
        nn.init.normal_(self.action_queries, std=0.02)

        # Positional encoding for decoder queries
        self.dec_pos = SinusoidalPositionEncoding(hidden_dim, max_len=chunk_size)

        # Transformer decoder (cross-attends to encoder output)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)

        # Output projection
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_dim)
            z:   (B, latent_dim)
        Returns:
            actions: (B, chunk_size, action_dim)
        """
        B = obs.shape[0]

        # Build encoder input: [z_token, obs_token]
        z_emb = self.latent_proj(z).unsqueeze(1)  # (B, 1, H)
        obs_emb = self.obs_proj(obs).unsqueeze(1)  # (B, 1, H)
        enc_input = torch.cat([z_emb, obs_emb], dim=1)  # (B, 2, H)
        enc_input = self.enc_pos(enc_input)

        # Encode
        memory = self.encoder(enc_input)  # (B, 2, H)

        # Decoder: learned queries → cross-attend to memory
        queries = self.action_queries.expand(B, -1, -1)  # (B, K, H)
        queries = self.dec_pos(queries)
        dec_out = self.decoder(queries, memory)  # (B, K, H)

        # Project to actions
        actions = self.action_head(dec_out)  # (B, K, action_dim)
        return actions


# ---------------------------------------------------------------------------
# Full ACT Policy
# ---------------------------------------------------------------------------
class ACTPolicy(nn.Module):
    """
    State-only Action Chunking Transformer.

    Training: CVAE encoder compresses (obs, actions) → z, decoder reconstructs.
              Loss = L1_reconstruction + beta * KL(q(z|obs,a) || N(0,I))
    Inference: z = 0, temporal ensembling over overlapping chunks.

    Args:
        obs_dim:    observation dimensionality (30 for Skill-2, 29 for Skill-3)
        action_dim: action dimensionality (9: arm5 + grip1 + base3)
        hidden_dim: transformer hidden size
        latent_dim: CVAE latent size
        chunk_size: number of future actions to predict
        n_heads:    number of attention heads
        n_enc_layers: encoder transformer layers
        n_dec_layers: decoder transformer layers
        ff_dim:     feedforward dimension
        dropout:    dropout rate
    """

    def __init__(
        self,
        obs_dim: int = 30,
        action_dim: int = 9,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        chunk_size: int = 20,
        n_heads: int = 4,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size

        # CVAE encoder (training only)
        self.cvae_encoder = CVAEEncoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            chunk_size=chunk_size,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # Policy decoder (used in both training and inference)
        self.decoder = PolicyDecoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            chunk_size=chunk_size,
            n_heads=n_heads,
            n_enc_layers=n_enc_layers,
            n_dec_layers=n_dec_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # ------- Temporal ensembling state -------
        self._temporal_agg = False
        self._all_time_actions: torch.Tensor | None = None
        self._query_freq: int = 1  # re-plan every N steps
        self._step_counter: int = 0
        self._exp_weights: torch.Tensor | None = None

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            obs:     (B, obs_dim)
            actions: (B, chunk_size, action_dim) — only during training

        Returns (training / when actions provided):
            {"pred_actions": (B, K, 9), "mu": (B, Z), "logvar": (B, Z)}
        Returns (inference / when actions=None):
            {"pred_actions": (B, K, 9)}
        """
        if actions is not None:
            # CVAE: encode → sample z → decode
            mu, logvar = self.cvae_encoder(obs, actions)
            z = self.reparameterize(mu, logvar)
            pred_actions = self.decoder(obs, z)
            return {"pred_actions": pred_actions, "mu": mu, "logvar": logvar}
        else:
            # Inference: z = 0 (prior mean)
            B = obs.shape[0]
            z = torch.zeros(B, self.latent_dim, device=obs.device)
            pred_actions = self.decoder(obs, z)
            return {"pred_actions": pred_actions}

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Single-step inference with temporal ensembling.
        Args:
            obs: (obs_dim,) or (1, obs_dim)
        Returns:
            action: (action_dim,) — single action for current timestep
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            out = self.forward(obs)
            chunk = out["pred_actions"]  # (1, K, action_dim)

        if not self._temporal_agg:
            # No ensembling: return first action of the chunk
            return chunk[0, 0]

        # --- Temporal ensembling ---
        K = self.chunk_size
        if self._all_time_actions is None:
            # Lazy init
            max_T = 10000
            self._all_time_actions = torch.zeros(
                max_T, max_T + K, self.action_dim, device=obs.device
            )
            w = torch.exp(-0.01 * torch.arange(K, dtype=torch.float32, device=obs.device))
            self._exp_weights = w / w.sum()

        t = self._step_counter
        self._all_time_actions[t, t : t + K] = chunk[0]
        self._step_counter += 1

        # Weighted average over all chunks that cover current timestep
        actions_for_t = self._all_time_actions[: t + 1, t]  # (t+1, action_dim)
        valid = min(t + 1, K)
        weights = self._exp_weights[:valid]
        action = (actions_for_t[-valid:] * weights.unsqueeze(-1)).sum(0)
        return action

    def reset_temporal(self):
        """Reset temporal ensembling state (call at episode start)."""
        self._all_time_actions = None
        self._step_counter = 0

    def enable_temporal_agg(self, enable: bool = True):
        self._temporal_agg = enable


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------
def act_loss(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 10.0,
) -> dict:
    """
    ACT training loss = L1 reconstruction + KL divergence.

    Args:
        pred_actions:   (B, K, action_dim) predicted action chunk
        target_actions: (B, K, action_dim) ground-truth action chunk
        mu:             (B, latent_dim)
        logvar:         (B, latent_dim)
        kl_weight:      weight for KL term (default 10, following original ACT)

    Returns:
        dict with 'total', 'l1', 'kl' losses
    """
    # L1 reconstruction loss
    l1 = F.l1_loss(pred_actions, target_actions)

    # KL divergence: KL(N(mu, sigma) || N(0, I))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = l1 + kl_weight * kl
    return {"total": total, "l1": l1, "kl": kl}


# ---------------------------------------------------------------------------
# Utility: count parameters
# ---------------------------------------------------------------------------
def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    # Quick sanity check
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ACTPolicy(
        obs_dim=30, action_dim=9, hidden_dim=256, latent_dim=32, chunk_size=20
    ).to(device)
    print(f"Model params: {count_params(model)}")

    # Training forward
    model.train()
    obs = torch.randn(8, 30, device=device)
    actions = torch.randn(8, 20, 9, device=device)
    out = model(obs, actions)
    loss = act_loss(out["pred_actions"], actions, out["mu"], out["logvar"])
    print(f"Training — L1: {loss['l1']:.4f}, KL: {loss['kl']:.4f}, Total: {loss['total']:.4f}")

    # Validation forward (eval mode + actions provided → should still return mu/logvar)
    model.eval()
    out = model(obs, actions)
    assert "mu" in out and "logvar" in out, "eval+actions should return mu/logvar"
    print(f"Validation — pred_actions shape: {out['pred_actions'].shape}, mu/logvar present: OK")

    # Inference forward (no actions → z=0)
    out = model(obs)
    assert "mu" not in out, "inference should not return mu"
    print(f"Inference — pred_actions shape: {out['pred_actions'].shape}")

    # Single-step with temporal ensembling
    model.enable_temporal_agg(True)
    model.reset_temporal()
    for t in range(5):
        a = model.get_action(obs[0])
        print(f"  Step {t}: action shape {a.shape}, values {a[:3].cpu().numpy()}")
