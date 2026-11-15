"""Competitive transformer policy (v2).

Entity tokens + 2-tile patch tokens + one global token -> transformer -> factored
heads: gate, hand-card pointer, wait-for-card pointer, fine 2,304-cell pointer
conditioned on the chosen card, categorical value over the crown difference, and
training-only auxiliary heads for opponent elixir / hand / next card.

The auxiliary labels are never inputs; they exist to force the trunk to model
the opponent for the main policy.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .obs import (
    GLOBAL_FEATURE_DIM, N_CELLS, N_PATCHES, PATCH_X, PATCH_Y, PAST_K,
    UNIT_FEATURE_DIM, cell_patch, cell_xy, patch_xy,
)

N_VALUE_BINS = 7
VALUE_BINS = tuple(range(-3, 4))  # crown difference -3..3


def fourier(xy: torch.Tensor, n_freq: int = 4) -> torch.Tensor:
    """xy [..., 2] -> periodic features [..., 4*n_freq]."""
    out = []
    for k in range(n_freq):
        w = (2.0 ** k) * math.pi
        out.append(torch.sin(w * xy))
        out.append(torch.cos(w * xy))
    return torch.cat(out, dim=-1)


def patch_index(xy_norm: torch.Tensor) -> torch.Tensor:
    """Normalised arena coords [..., 2] -> flat 2-tile patch index [...,]."""
    px = (xy_norm[..., 0] * PATCH_X).floor().long()
    py = (xy_norm[..., 1] * PATCH_Y).floor().long()
    return (py.clamp(0, PATCH_Y - 1) * PATCH_X
            + px.clamp(0, PATCH_X - 1))


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        logits = logits.masked_fill(~mask, -1e4)
    return F.softmax(logits, dim=dim)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor | None, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        logits = logits.masked_fill(~mask, -1e4)
    return F.log_softmax(logits, dim=dim)


class CompetitiveNet(nn.Module):
    def __init__(
        self,
        n_cards: int,
        *,
        d_model: int = 192,
        layers: int = 4,
        heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        d_card: int = 64,
        unit_feat_dim: int = UNIT_FEATURE_DIM,
        global_feat_dim: int = GLOBAL_FEATURE_DIM,
        belief_dim: int | None = None,
        n_hand: int = 4,
        n_deck: int = 8,
        n_past: int = PAST_K,
        n_fourier: int = 4,
        n_value_bins: int = N_VALUE_BINS,
    ):
        super().__init__()
        self.n_cards = int(n_cards)
        self.d_model = int(d_model)
        self.d_card = int(d_card)
        self.unit_feat_dim = int(unit_feat_dim)
        self.global_feat_dim = int(global_feat_dim)
        self.belief_dim = int(belief_dim) if belief_dim is not None else 3 * self.n_cards + 6
        self.n_hand = int(n_hand)
        self.n_deck = int(n_deck)
        self.n_past = int(n_past)
        self.n_fourier = int(n_fourier)
        self.n_value_bins = int(n_value_bins)

        nf = 4 * n_fourier
        self.card_emb = nn.Embedding(self.n_cards, d_card, padding_idx=0)
        self.unit_in = nn.Sequential(
            nn.Linear(unit_feat_dim + nf, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.unit_id_proj = nn.Linear(d_card, d_model)

        self.patch_pos = nn.Parameter(torch.zeros(N_PATCHES, d_model))
        self.patch_in = nn.Sequential(
            nn.Linear(nf, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.type_emb = nn.Parameter(torch.zeros(3, d_model))

        self.glob_fc = nn.Sequential(
            nn.Linear(global_feat_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.hand_fc = nn.Sequential(
            nn.Linear(2 * d_card, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.deck_fc = nn.Sequential(
            nn.Linear(d_card, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.next_fc = nn.Sequential(
            nn.Linear(d_card, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.past_fc = nn.Sequential(
            nn.Linear(d_card + 2 + nf + 1, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.opp_past_fc = nn.Sequential(
            nn.Linear(d_card + 2 + nf + 1, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.belief_fc = nn.Sequential(
            nn.Linear(self.belief_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.global_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)

        self.gate_head = nn.Linear(d_model, 1)
        self.card_q = nn.Linear(d_model, d_card)
        self.card_bias = nn.Embedding(self.n_cards, 1)
        self.wait_q = nn.Linear(d_model, d_card)
        self.wait_bias = nn.Embedding(self.n_cards, 1)
        self.query = nn.Sequential(
            nn.Linear(d_model + d_card, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.cell_key = nn.Linear(d_model, d_model, bias=False)
        self.cell_emb = nn.Parameter(torch.zeros(N_CELLS, d_model))
        self.cell_bias = nn.Parameter(torch.zeros(N_CELLS))

        self.value_logits = nn.Linear(d_model, self.n_value_bins)
        self.value_scalar = nn.Linear(d_model, 1)
        self.aux_elixir = nn.Linear(d_model, 1)
        self.aux_hand = nn.Linear(d_model, self.n_cards)
        self.aux_next = nn.Linear(d_model, self.n_cards)

        self.register_buffer("cell_xy_buf", torch.as_tensor(cell_xy(), dtype=torch.float32), persistent=False)
        self.register_buffer("patch_xy_buf", torch.as_tensor(patch_xy(), dtype=torch.float32), persistent=False)
        self.register_buffer("cell_patch_buf", torch.as_tensor(cell_patch(), dtype=torch.long), persistent=False)
        self.register_buffer("value_bins_buf", torch.tensor(VALUE_BINS, dtype=torch.float32), persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.patch_pos, std=0.02)
        nn.init.normal_(self.type_emb, std=0.02)
        nn.init.normal_(self.cell_emb, std=0.02)
        nn.init.zeros_(self.cell_bias)
        nn.init.orthogonal_(self.gate_head.weight, gain=0.01)
        nn.init.orthogonal_(self.card_q.weight, gain=0.01)
        nn.init.orthogonal_(self.wait_q.weight, gain=0.01)
        nn.init.orthogonal_(self.value_logits.weight, gain=0.01)
        nn.init.orthogonal_(self.value_scalar.weight, gain=1.0)
        for head in (self.aux_hand, self.aux_next):
            nn.init.orthogonal_(head.weight, gain=0.01)

    # ------------------------------------------------------------------ config
    def config(self) -> dict[str, Any]:
        return {
            "n_cards": self.n_cards,
            "d_model": self.d_model,
            "layers": len(self.encoder.layers),
            "heads": self.encoder.layers[0].self_attn.num_heads,
            "dim_feedforward": self.encoder.layers[0].linear1.out_features,
            "dropout": self.encoder.layers[0].dropout.p,
            "d_card": self.d_card,
            "unit_feat_dim": self.unit_feat_dim,
            "global_feat_dim": self.global_feat_dim,
            "belief_dim": self.belief_dim,
            "n_hand": self.n_hand,
            "n_deck": self.n_deck,
            "n_past": self.n_past,
            "n_fourier": self.n_fourier,
            "n_value_bins": self.n_value_bins,
        }

    # ------------------------------------------------------------------ encode
    def _past(self, past: torch.Tensor) -> torch.Tensor:
        """past [B,K,5] -> [B,d_model] summed over valid entries."""
        B, K, _ = past.shape
        ids = past[..., 0].long().clamp(0, self.n_cards - 1)
        emb = self.card_emb(ids)
        xy = past[..., 1:3].clamp(0.0, 1.0)
        dt = past[..., 3:4]
        valid = past[..., 4] > 0.5
        vec = torch.cat([emb, xy, fourier(xy, self.n_fourier), dt], dim=-1)
        proj = self.past_fc(vec) * valid.unsqueeze(-1)
        return proj.sum(dim=1)

    def _opp_past(self, past: torch.Tensor) -> torch.Tensor:
        B, K, _ = past.shape
        ids = past[..., 0].long().clamp(0, self.n_cards - 1)
        emb = self.card_emb(ids)
        xy = past[..., 1:3].clamp(0.0, 1.0)
        dt = past[..., 3:4]
        valid = past[..., 4] > 0.5
        vec = torch.cat([emb, xy, fourier(xy, self.n_fourier), dt], dim=-1)
        proj = self.opp_past_fc(vec) * valid.unsqueeze(-1)
        return proj.sum(dim=1)

    def encode(
        self,
        unit_ids: torch.Tensor,
        unit_feats: torch.Tensor,
        unit_mask: torch.Tensor,
        global_feats: torch.Tensor,
        hand_ids: torch.Tensor,
        hand_mask: torch.Tensor,
        deck_ids: torch.Tensor,
        deck_mask: torch.Tensor,
        next_id: torch.Tensor,
        past_own: torch.Tensor,
        past_opp: torch.Tensor,
        belief: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, U = unit_ids.shape
        d = self.d_model
        nf = self.n_fourier

        ids = unit_ids.long().clamp(0, self.n_cards - 1)
        unit_emb = self.card_emb(ids)
        xy = unit_feats[..., 8:10].clamp(0.0, 1.0)
        u = self.unit_in(torch.cat([unit_feats, fourier(xy, nf)], dim=-1))
        u = u + self.unit_id_proj(unit_emb)
        u = u * unit_mask.unsqueeze(-1).to(u.dtype)

        pxy = self.patch_xy_buf.to(u.device)
        p = (self.patch_pos.to(u.device).unsqueeze(0).expand(B, -1, -1)
             + self.patch_in(fourier(pxy, nf)).unsqueeze(0))
        # Normalised coordinates -> 2-tile patch index.
        pid = patch_index(xy)
        p = p.scatter_add(1, pid.unsqueeze(-1).expand(-1, -1, d), u)

        # Global token parts.
        hand_emb = self.card_emb(hand_ids.long().clamp(0, self.n_cards - 1))
        hm = hand_mask.unsqueeze(-1).to(hand_emb.dtype)
        hand_mean = (hand_emb * hm).sum(1) / hm.sum(1).clamp(min=1.0)
        hand_max = (hand_emb.masked_fill(~hand_mask.unsqueeze(-1), -1e4)).max(dim=1).values
        hand_max = torch.where(hand_mask.any(dim=1, keepdim=True), hand_max, torch.zeros_like(hand_max))
        hand_vec = self.hand_fc(torch.cat([hand_mean, hand_max], dim=-1))

        deck_emb = self.card_emb(deck_ids.long().clamp(0, self.n_cards - 1))
        dm = deck_mask.unsqueeze(-1).to(deck_emb.dtype)
        deck_mean = (deck_emb * dm).sum(1) / dm.sum(1).clamp(min=1.0)
        deck_vec = self.deck_fc(deck_mean)

        next_vec = self.next_fc(self.card_emb(next_id.long().clamp(0, self.n_cards - 1)))

        g = (self.glob_fc(global_feats)
             + hand_vec + deck_vec + next_vec
             + self._past(past_own) + self._opp_past(past_opp)
             + self.belief_fc(belief))
        g = self.global_norm(g).unsqueeze(1)

        # Keep attention finite when a frame has no units.
        safe_mask = unit_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True

        x = torch.cat([
            g + self.type_emb[0].to(g.device),
            p + self.type_emb[1].to(g.device),
            u + self.type_emb[2].to(g.device),
        ], dim=1)
        pad = torch.cat([
            torch.zeros(B, 1 + N_PATCHES, dtype=torch.bool, device=u.device),
            ~safe_mask,
        ], dim=1)
        h = self.encoder(x, src_key_padding_mask=pad)
        return {
            "g": h[:, 0],
            "p": h[:, 1:1 + N_PATCHES],
            "u": h[:, 1 + N_PATCHES:],
        }

    # ------------------------------------------------------------------- heads
    def heads(
        self,
        enc: dict[str, torch.Tensor],
        hand_ids: torch.Tensor,
        hand_mask: torch.Tensor,
        deck_ids: torch.Tensor,
        deck_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        g = enc["g"]
        card_emb = self.card_emb(hand_ids.long().clamp(0, self.n_cards - 1))
        card_logits = torch.einsum("bd,bkd->bk", self.card_q(g), card_emb)
        card_logits = card_logits + self.card_bias(hand_ids.long().clamp(0, self.n_cards - 1)).squeeze(-1)
        card_logits = card_logits.masked_fill(~hand_mask, -1e4)

        deck_emb = self.card_emb(deck_ids.long().clamp(0, self.n_cards - 1))
        wait_logits = torch.einsum("bd,bkd->bk", self.wait_q(g), deck_emb)
        wait_logits = wait_logits + self.wait_bias(deck_ids.long().clamp(0, self.n_cards - 1)).squeeze(-1)
        wait_logits = wait_logits.masked_fill(~deck_mask, -1e4)

        value_logits = self.value_logits(g)
        value = (torch.softmax(value_logits, dim=-1) * self.value_bins_buf.to(g.device)).sum(-1, keepdim=True)
        return {
            "gate": self.gate_head(g).squeeze(-1),
            "card": card_logits,
            "wait": wait_logits,
            "value": value,
            "value_logits": value_logits,
            "aux_elixir": torch.sigmoid(self.aux_elixir(g).squeeze(-1)),
            "aux_hand": self.aux_hand(g),
            "aux_next": self.aux_next(g),
        }

    def cell_logits(self, enc: dict[str, torch.Tensor], card_ids: torch.Tensor) -> torch.Tensor:
        """[B, 2304] logits for placing ``card_ids`` (one card per row)."""
        g = enc["g"]
        card_emb = self.card_emb(card_ids.long().clamp(0, self.n_cards - 1))
        q = self.query(torch.cat([g, card_emb], dim=-1))
        kp = (self.cell_key(enc["p"]) * q.unsqueeze(1)).sum(-1)
        kc = q @ self.cell_key(self.cell_emb).t()
        return (kp[:, self.cell_patch_buf.to(g.device)] + kc) / math.sqrt(self.d_model) + self.cell_bias.to(g.device)

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        unit_ids: torch.Tensor,
        unit_feats: torch.Tensor,
        unit_mask: torch.Tensor,
        global_feats: torch.Tensor,
        hand_ids: torch.Tensor,
        hand_mask: torch.Tensor,
        deck_ids: torch.Tensor,
        deck_mask: torch.Tensor,
        next_id: torch.Tensor,
        past_own: torch.Tensor,
        past_opp: torch.Tensor,
        belief: torch.Tensor,
        cell_card: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        enc = self.encode(
            unit_ids, unit_feats, unit_mask, global_feats,
            hand_ids, hand_mask, deck_ids, deck_mask, next_id,
            past_own, past_opp, belief,
        )
        out = self.heads(enc, hand_ids, hand_mask, deck_ids, deck_mask)
        out["enc"] = enc
        if cell_card is not None:
            out["cell"] = self.cell_logits(enc, cell_card)
        return out
