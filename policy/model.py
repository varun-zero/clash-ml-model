"""
Deck-Conditioned Factored Policy Network for Clash Royale.
Learns across arbitrary Clash Royale decks from YouTube VoD demonstrations.
Uses Card ID Embeddings + Factored Autoregressive (Card -> Tile) Decision Heads.
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.constants import NUM_TOTAL_CARDS, TOTAL_TILES, ARENA_ROWS, ARENA_COLS


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
    """
    Modular policy network conditioned on arbitrary deck compositions.
    Architecture:
      1. Spatial CNN: Encodes (B, 8, 32, 18) board tensor into 2560-dim vector.
      2. Card Embeddings + Scalar MLP: Encodes hand card IDs and 12 game scalars into 128-dim vector.
      3. Shared Trunk: 512-dim fusion representation.
      4. Head 1 (Card Head): 5 action logits [Wait, Slot0, Slot1, Slot2, Slot3] with Elixir legality mask.
      5. Head 2 (Tile Head): 576 action logits conditioned on trunk + chosen card embedding with tile placement mask.
    """
    def __init__(
        self,
        num_card_classes: int = NUM_TOTAL_CARDS,
        card_emb_dim: int = 16,
        num_scalars: int = 12,
        use_residual: bool = True,
    ):
        super().__init__()
        self.num_card_classes = num_card_classes
        self.card_emb_dim = card_emb_dim
        self.num_scalars = num_scalars
        self.use_residual = use_residual

        # 0. Card ID Embedding Layer (0 = Wait / Empty Card)
        self.card_emb = nn.Embedding(num_card_classes, card_emb_dim, padding_idx=0)

        # 1. Spatial Arena CNN: Input (B, 8, 32, 18)
        # Conv1: (B, 8, 32, 18) -> (B, 32, 32, 18)
        # Conv2: (B, 32, 32, 18) -> (B, 64, 16, 9)
        # Conv3: (B, 64, 16, 9) -> (B, 64, 8, 5)
        # Flatten: 64 * 8 * 5 = 2560
        if use_residual:
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(8, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                ResidualBlock2D(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # (B, 64, 16, 9)
                nn.BatchNorm2d(64),
                nn.SiLU(),
                ResidualBlock2D(64),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # (B, 64, 8, 5)
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Flatten(),
            )
        else:
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(8, 32, kernel_size=3, padding=1),
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

        # 2. Scalar + Hand Embedding Encoder
        # Scalars: [ally_elixir, est_enemy_elixir, time_rem, 6x tower_hp, double_elixir, overtime, king_active] -> 12 floats
        # Plus 4 cards in hand: 4 * card_emb_dim
        self.scalar_mlp = nn.Sequential(
            nn.Linear(num_scalars + (4 * card_emb_dim), 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )

        # 3. Shared Trunk
        self.trunk = nn.Sequential(
            nn.Linear(2560 + 128, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        # 4. Head 1: Card Slot Selection (0 = Wait, 1..4 = Hand Slots 0..3)
        self.card_head = nn.Linear(512, 5)

        # 5. Head 2: Tile Coordinate Selection (576 = 32 * 18 tiles)
        # Conditioned on Trunk features + Chosen Card Embedding
        self.tile_head = nn.Sequential(
            nn.Linear(512 + card_emb_dim, 512),
            nn.SiLU(),
            nn.Linear(512, TOTAL_TILES),
        )

    def forward(
        self,
        arena_grid: torch.Tensor,
        scalars: torch.Tensor,
        hand_card_ids: torch.Tensor,
        legal_card_mask: torch.Tensor,
        legal_tile_mask: torch.Tensor,
        teacher_card_action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            arena_grid: (B, 8, 32, 18) float32
            scalars: (B, num_scalars) float32
            hand_card_ids: (B, 4) int64
            legal_card_mask: (B, 5) bool
            legal_tile_mask: (B, 576) bool
            teacher_card_action: (B,) int64 ground truth slot during training
        Returns:
            masked_card_logits: (B, 5) float32
            masked_tile_logits: (B, 576) float32
        """
        B = arena_grid.size(0)

        # If incoming scalars dimension is 9, pad with zeros up to num_scalars
        if scalars.size(-1) < self.num_scalars:
            pad = torch.zeros(
                B, self.num_scalars - scalars.size(-1),
                device=scalars.device, dtype=scalars.dtype
            )
            scalars = torch.cat([scalars, pad], dim=-1)

        # 1. Spatial representation (B, 2560)
        spatial_feat = self.spatial_conv(arena_grid)

        # 2. Embed the 4 hand cards (B, 4, card_emb_dim)
        hand_embs = self.card_emb(hand_card_ids)
        hand_embs_flat = hand_embs.view(B, -1)

        # 3. Scalar fusion & shared trunk
        scalar_in = torch.cat([scalars, hand_embs_flat], dim=-1)
        scalar_feat = self.scalar_mlp(scalar_in)
        trunk_feat = self.trunk(torch.cat([spatial_feat, scalar_feat], dim=-1))

        # 4. Head 1: Card Slot Prediction
        raw_card_logits = self.card_head(trunk_feat)
        # Apply legal card action mask (-1e4 for fp16/half to avoid overflow, -1e9 for fp32)
        fill_val = -1e4 if raw_card_logits.dtype in (torch.float16, torch.bfloat16) else -1e9
        masked_card_logits = raw_card_logits.masked_fill(~legal_card_mask, fill_val)

        # Determine chosen card embedding for conditioning the tile head
        if teacher_card_action is not None:
            chosen_slot = teacher_card_action
        else:
            chosen_slot = torch.argmax(masked_card_logits, dim=-1)

        # Map chosen_slot (0..4) to its embedding (slot 0 = Wait -> zero vector)
        pad_zero = torch.zeros(B, 1, self.card_emb_dim, device=arena_grid.device, dtype=hand_embs.dtype)
        all_slot_embs = torch.cat([pad_zero, hand_embs], dim=1)  # (B, 5, 16)
        
        batch_idx = torch.arange(B, device=arena_grid.device)
        chosen_card_emb = all_slot_embs[batch_idx, chosen_slot]  # (B, 16)

        # 5. Head 2: Tile Coordinate Prediction
        tile_in = torch.cat([trunk_feat, chosen_card_emb], dim=-1)
        raw_tile_logits = self.tile_head(tile_in)
        masked_tile_logits = raw_tile_logits.masked_fill(~legal_tile_mask, fill_val)

        return masked_card_logits, masked_tile_logits

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
        """
        Inference helper for live execution loop.
        Accepts unbatched or batched tensors and returns:
            chosen_slot (0..4), chosen_tile (0..575), card_conf, tile_conf
        """
        self.eval()
        # Add batch dim if unbatched
        if arena_grid.dim() == 3:
            arena_grid = arena_grid.unsqueeze(0)
            scalars = scalars.unsqueeze(0)
            hand_card_ids = hand_card_ids.unsqueeze(0)
            legal_card_mask = legal_card_mask.unsqueeze(0)
            legal_tile_mask = legal_tile_mask.unsqueeze(0)

        card_logits, tile_logits = self.forward(
            arena_grid, scalars, hand_card_ids, legal_card_mask, legal_tile_mask
        )

        card_probs = F.softmax(card_logits / max(temperature, 1e-4), dim=-1)
        tile_probs = F.softmax(tile_logits / max(temperature, 1e-4), dim=-1)

        if deterministic:
            chosen_slot = int(torch.argmax(card_probs, dim=-1)[0].item())
            chosen_tile = int(torch.argmax(tile_probs, dim=-1)[0].item())
        else:
            chosen_slot = int(torch.multinomial(card_probs, num_samples=1)[0, 0].item())
            chosen_tile = int(torch.multinomial(tile_probs, num_samples=1)[0, 0].item())

        card_conf = float(card_probs[0, chosen_slot].item())
        tile_conf = float(tile_probs[0, chosen_tile].item())

        return chosen_slot, chosen_tile, card_conf, tile_conf
