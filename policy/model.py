"""
Global-vocabulary policy network for CR-Zero.

Card identity comes from the single global CARD_DATABASE vocabulary:
    Embedding(135, 16) + Linear(static card features -> 16)

The static branch guarantees that a card ID never seen during training still
has a valid, non-crashing input representation. Action heads are factored:
card hand slot (0=Wait, 1..4=slots) and one of 576 arena tiles.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.constants import (
    ARENA_COLS,
    ARENA_ROWS,
    NUM_SCALARS,
    NUM_TOTAL_CARDS,
    STACKED_GRID_CHANNELS,
    TOTAL_TILES,
)
from configs.card_features import CARD_FEATURE_DIM, build_card_feature_matrix


class ResidualBlock2D(nn.Module):
    """Residual convolutional block with batchnorm and SiLU activations."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)


class DeckConditionedPolicy(nn.Module):
    """Global-card-conditioned factored policy."""

    def __init__(
        self,
        num_card_classes: int = NUM_TOTAL_CARDS,
        card_emb_dim: int = 16,
        num_scalars: int = NUM_SCALARS,
        in_channels: int = STACKED_GRID_CHANNELS,
        use_residual: bool = True,
    ):
        super().__init__()
        if num_scalars != 12:
            raise ValueError(f"Expected 12 scalars, got {num_scalars}")
        if in_channels != 32:
            raise ValueError(f"Expected 32 spatial input channels, got {in_channels}")
        if num_card_classes <= 0 or num_card_classes > NUM_TOTAL_CARDS:
            raise ValueError(
                f"num_card_classes must be in 1..{NUM_TOTAL_CARDS}, got {num_card_classes}"
            )

        self.num_card_classes = num_card_classes
        self.card_emb_dim = card_emb_dim
        self.num_scalars = num_scalars
        self.in_channels = in_channels
        self.use_residual = use_residual

        # Learned identity embedding + static-feature projection, summed.
        self.card_emb = nn.Embedding(num_card_classes, card_emb_dim, padding_idx=0)
        self.card_feat_mlp = nn.Linear(CARD_FEATURE_DIM, card_emb_dim)
        self.register_buffer(
            "card_features",
            torch.from_numpy(build_card_feature_matrix(num_card_classes)),
        )

        # Spatial arena encoder: (B, 32, 32, 18) -> (B, 64, 8, 5) -> 2560.
        if use_residual:
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                ResidualBlock2D(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                ResidualBlock2D(64),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Flatten(),
            )
        else:
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Flatten(),
            )

        # Scalar + hand embedding encoder.
        self.scalar_mlp = nn.Sequential(
            nn.Linear(num_scalars + (4 * card_emb_dim), 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(2560 + 128, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        # Head 1: Wait + four hand slots.
        self.card_head = nn.Linear(512, 5)

        # Head 2: 576 tiles, conditioned on trunk + chosen card embedding.
        self.tile_head = nn.Sequential(
            nn.Linear(512 + card_emb_dim, 512),
            nn.SiLU(),
            nn.Linear(512, TOTAL_TILES),
        )

    def _validate_hand_ids(self, hand_card_ids: torch.Tensor) -> None:
        if hand_card_ids.numel() == 0:
            raise ValueError("hand_card_ids cannot be empty")
        if int(hand_card_ids.min().item()) < 0 or int(hand_card_ids.max().item()) >= self.num_card_classes:
            raise ValueError(
                f"hand_card_ids must be global IDs in 0..{self.num_card_classes - 1}"
            )

    def _encode(
        self,
        arena_grid: torch.Tensor,
        scalars: torch.Tensor,
        hand_card_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if arena_grid.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels (4 stacked frames), "
                f"got {arena_grid.size(1)}"
            )
        if scalars.size(-1) != self.num_scalars:
            raise ValueError(f"Expected {self.num_scalars} scalars, got {scalars.size(-1)}")
        self._validate_hand_ids(hand_card_ids)

        batch_size = arena_grid.size(0)
        spatial_feat = self.spatial_conv(arena_grid)

        hand_embs = self.card_emb(hand_card_ids.long()) + self.card_feat_mlp(
            self.card_features[hand_card_ids.long()]
        )
        hand_embs_flat = hand_embs.reshape(batch_size, -1)

        scalar_in = torch.cat([scalars, hand_embs_flat], dim=-1)
        scalar_feat = self.scalar_mlp(scalar_in)
        trunk_feat = self.trunk(torch.cat([spatial_feat, scalar_feat], dim=-1))
        return trunk_feat, hand_embs

    @staticmethod
    def _mask_value(dtype: torch.dtype) -> float:
        return -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9

    @staticmethod
    def _sample_or_argmax(logits: torch.Tensor, temperature: float, deterministic: bool) -> Tuple[int, float]:
        probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
        if deterministic:
            index = int(torch.argmax(probs, dim=-1)[0].item())
        else:
            index = int(torch.multinomial(probs, num_samples=1)[0, 0].item())
        return index, float(probs[0, index].item())

    def forward(
        self,
        arena_grid: torch.Tensor,
        scalars: torch.Tensor,
        hand_card_ids: torch.Tensor,
        legal_card_mask: torch.Tensor,
        legal_tile_mask: torch.Tensor,
        teacher_card_action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if arena_grid.dim() != 4:
            raise ValueError(f"arena_grid must be (B, C, H, W), got {tuple(arena_grid.shape)}")
        batch_size = arena_grid.size(0)
        trunk_feat, hand_embs = self._encode(arena_grid, scalars, hand_card_ids)

        raw_card_logits = self.card_head(trunk_feat)
        fill_val = self._mask_value(raw_card_logits.dtype)
        masked_card_logits = raw_card_logits.masked_fill(~legal_card_mask.bool(), fill_val)

        if teacher_card_action is not None:
            chosen_slot = teacher_card_action.long()
        else:
            chosen_slot = torch.argmax(masked_card_logits, dim=-1)

        pad_zero = torch.zeros(
            batch_size, 1, self.card_emb_dim, device=arena_grid.device, dtype=hand_embs.dtype
        )
        all_slot_embs = torch.cat([pad_zero, hand_embs], dim=1)
        batch_idx = torch.arange(batch_size, device=arena_grid.device)
        chosen_card_emb = all_slot_embs[batch_idx, chosen_slot]

        tile_in = torch.cat([trunk_feat, chosen_card_emb], dim=-1)
        raw_tile_logits = self.tile_head(tile_in)
        masked_tile_logits = raw_tile_logits.masked_fill(~legal_tile_mask.bool(), fill_val)
        return masked_card_logits, masked_tile_logits

    def _ensure_batched(
        self,
        arena_grid: torch.Tensor,
        scalars: torch.Tensor,
        hand_card_ids: torch.Tensor,
        legal_card_mask: torch.Tensor,
        legal_tile_mask: Optional[torch.Tensor] = None,
    ):
        if arena_grid.dim() == 3:
            arena_grid = arena_grid.unsqueeze(0)
            scalars = scalars.unsqueeze(0)
            hand_card_ids = hand_card_ids.unsqueeze(0)
            legal_card_mask = legal_card_mask.unsqueeze(0)
            if legal_tile_mask is not None:
                legal_tile_mask = legal_tile_mask.unsqueeze(0)
        return arena_grid, scalars, hand_card_ids, legal_card_mask, legal_tile_mask

    @torch.no_grad()
    def predict_card(
        self,
        arena_grid: torch.Tensor,
        scalars: torch.Tensor,
        hand_card_ids: torch.Tensor,
        legal_card_mask: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """Return (slot 0..4, confidence). Does not compute tile logits."""
        self.eval()
        arena_grid, scalars, hand_card_ids, legal_card_mask, _ = self._ensure_batched(
            arena_grid,
            scalars,
            hand_card_ids,
            legal_card_mask,
            None,
        )
        trunk_feat, _ = self._encode(arena_grid, scalars, hand_card_ids)
        logits = self.card_head(trunk_feat)
        logits = logits.masked_fill(~legal_card_mask.bool(), self._mask_value(logits.dtype))
        return self._sample_or_argmax(logits, temperature, deterministic)

    @torch.no_grad()
    def predict_tile(
        self,
        arena_grid: torch.Tensor,
        scalars: torch.Tensor,
        hand_card_ids: torch.Tensor,
        legal_tile_mask: torch.Tensor,
        card_slot: int,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """Return (tile 0..575, confidence) conditioned on a fixed card slot."""
        self.eval()
        if not 0 <= int(card_slot) <= 4:
            raise ValueError(f"card_slot must be 0..4, got {card_slot}")

        arena_grid, scalars, hand_card_ids, _, legal_tile_mask = self._ensure_batched(
            arena_grid,
            scalars,
            hand_card_ids,
            torch.ones(1, 5, dtype=torch.bool, device=arena_grid.device),
            legal_tile_mask,
        )
        trunk_feat, hand_embs = self._encode(arena_grid, scalars, hand_card_ids)

        pad_zero = torch.zeros(
            1, 1, self.card_emb_dim, device=arena_grid.device, dtype=hand_embs.dtype
        )
        all_slot_embs = torch.cat([pad_zero, hand_embs], dim=1)
        chosen_card_emb = all_slot_embs[:, int(card_slot)]
        logits = self.tile_head(torch.cat([trunk_feat, chosen_card_emb], dim=-1))
        logits = logits.masked_fill(~legal_tile_mask.bool(), self._mask_value(logits.dtype))
        return self._sample_or_argmax(logits, temperature, deterministic)

    @torch.no_grad()
    def predict_action(
        self,
        arena_grid: torch.Tensor,
        scalars: torch.Tensor,
        hand_card_ids: torch.Tensor,
        legal_card_mask: torch.Tensor,
        legal_tile_mask: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Tuple[int, int, float, float]:
        """Convenience wrapper used by tests and simple callers."""
        slot, card_conf = self.predict_card(
            arena_grid,
            scalars,
            hand_card_ids,
            legal_card_mask,
            temperature=temperature,
            deterministic=deterministic,
        )
        tile, tile_conf = self.predict_tile(
            arena_grid,
            scalars,
            hand_card_ids,
            legal_tile_mask,
            card_slot=slot,
            temperature=temperature,
            deterministic=deterministic,
        )
        return slot, tile, card_conf, tile_conf
