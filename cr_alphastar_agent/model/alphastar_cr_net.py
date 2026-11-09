"""
AlphaStar + OpenAI Five Hybrid Neural Architecture for Clash Royale:
- Entity Transformer (processes active Crown Towers + troops with masked mean pooling)
- Spatial Grid CNN (processes 18x32 arena grid state)
- Scalar & Hand Card Embeddings (elixir, timers, hand & next cards)
- Core Temporal Fusion
- Autoregressive Action Heads (Card slot sampled first, Tile location conditioned on card)
- FP16-Safe Logit Masking (FP32 cast before -1e9 masking)
- Card-Conditioned Tile Masking (Full arena for spells like Fireball/Zap; restricted for troops)
- OpenAI Five Value Critic Head for PPO/GAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from cr_alphastar_agent.config import (
    ARENA_WIDTH, ARENA_HEIGHT, TOTAL_TILES, MAX_ENTITIES, D_MODEL,
    ENTITY_FEATURES, SPATIAL_CHANNELS, NUM_CARDS, SPELL_CARD_IDS
)


class AlphaStarCRNet(nn.Module):
    def __init__(self, num_cards: int = NUM_CARDS, max_entities: int = MAX_ENTITIES, d_model: int = D_MODEL):
        super().__init__()
        self.d_model = d_model
        self.max_entities = max_entities
        self.num_cards = num_cards

        # Register spell card IDs buffer for card-conditioned tile masking
        spell_tensor = torch.tensor(list(SPELL_CARD_IDS), dtype=torch.long)
        self.register_buffer("spell_ids", spell_tensor)

        # 1. Scalar & Hand Card Embeddings
        # 0 = empty/none, 1..num_cards = valid card IDs
        self.card_emb = nn.Embedding(num_cards + 1, 32, padding_idx=0)
        # 4 scalars: [own_elixir_norm, enemy_elixir_est_norm, time_norm, is_overtime] + 5 cards * 32
        self.scalar_fc = nn.Sequential(
            nn.Linear(4 + 5 * 32, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # 2. AlphaStar Entity Transformer (Crown Towers + active units)
        self.entity_proj = nn.Linear(ENTITY_FEATURES, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.entity_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. Spatial Grid Encoder (Input shape: [Batch, 6, 32, 18])
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(SPATIAL_CHANNELS, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [Batch, 64, 16, 9]
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 9, d_model),
            nn.SiLU()
        )

        # 4. Core Representation Fusion (Combines Scalar + Entity + Spatial)
        self.core_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )

        # 5. Autoregressive Action Heads
        # Head A: Card Slot (0, 1, 2, 3 = Hand Cards, 4 = Wait / No-Op)
        self.card_head = nn.Linear(d_model, 5)
        self.chosen_action_emb = nn.Embedding(5, d_model)

        # Head B: Tile Placement (18 * 32 = 576), conditioned on [core_state, chosen_action_emb]
        self.tile_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, TOTAL_TILES)
        )

        # 6. OpenAI Five Value Critic Head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.orthogonal_(self.card_head.weight, gain=0.01)
        nn.init.orthogonal_(self.tile_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def forward_core(self, scalars: torch.Tensor, hand_cards: torch.Tensor,
                     entities: torch.Tensor, entity_mask: torch.Tensor,
                     arena_grid: torch.Tensor) -> torch.Tensor:
        """
        Fuses scalar, entity, and spatial representations into unified core state.
        Guaranteed NaN-safe via masked mean pooling and non-empty entity guarantees.
        """
        # 1. Scalar + Hand Card features
        c_emb = self.card_emb(hand_cards).flatten(start_dim=1)  # [Batch, 5 * 32]
        s_feat = self.scalar_fc(torch.cat([scalars, c_emb], dim=-1))  # [Batch, d_model]

        # 2. Entity Transformer features
        # Safeguard: Ensure no batch item has all True padding mask (which causes NaN in softmax)
        safe_entity_mask = entity_mask.clone()
        all_padded = safe_entity_mask.all(dim=-1)
        if all_padded.any():
            safe_entity_mask[all_padded, 0] = False

        e_in = self.entity_proj(entities)  # [Batch, max_entities, d_model]
        e_trans = self.entity_transformer(e_in, src_key_padding_mask=safe_entity_mask)

        # Masked average pooling across active units
        # entity_mask is True for padding slots, False for valid units
        valid_mask = (~safe_entity_mask).float().unsqueeze(-1)  # [Batch, max_entities, 1]
        sum_entities = (e_trans * valid_mask).sum(dim=1)
        count_entities = valid_mask.sum(dim=1).clamp(min=1.0)
        e_feat = sum_entities / count_entities  # [Batch, d_model]

        # 3. Spatial CNN features
        sp_feat = self.spatial_cnn(arena_grid)  # [Batch, d_model]

        # 4. Temporal Core Fusion
        core_state = self.core_fusion(torch.cat([s_feat, e_feat, sp_feat], dim=-1))
        return core_state

    def get_action_and_value(self, obs: dict, action: torch.Tensor = None, deterministic: bool = False,
                             return_entropy_breakdown: bool = False):
        """
        Autoregressive action sampling & value estimation with:
        - FP16-safe logit masking (FP32 cast before -1e9 mask)
        - Card-conditioned tile masking (spells get full arena mask; troops get restricted deployment mask)
        - Separated card and tile entropy metrics to prevent No-Op exploration collapse
        """
        core = self.forward_core(
            obs["scalars"], obs["hand_cards"], obs["entities"],
            obs["entity_mask"], obs["arena_grid"]
        )
        value = self.value_head(core).squeeze(-1)

        batch_size = core.shape[0]
        card_mask = obs["card_mask"].bool()
        base_tile_mask = obs["tile_mask"].bool()
        card_tile_mask = obs.get("card_tile_mask")

        # Step 1: Predict Card Slot (0..3 hand cards, 4 No-Op)
        # FP16-safe: Cast logits to FP32 before applying -1e9 mask
        card_logits = self.card_head(core).float()
        masked_card_logits = card_logits.masked_fill(~card_mask, -1e9)
        card_dist = Categorical(logits=masked_card_logits)

        if action is None:
            if deterministic:
                chosen_card_slot = masked_card_logits.argmax(dim=-1)
            else:
                chosen_card_slot = card_dist.sample()
        else:
            chosen_card_slot = action[:, 0]

        # Step 2: Determine Card-Conditioned Tile Mask
        # If card is a spell, allow all 576 tiles (or valid spell arena mask)
        # If card is a troop or building, restrict to base_tile_mask
        hand_cards = obs["hand_cards"][:, :4]  # [Batch, 4]
        clamped_slot = chosen_card_slot.clamp(0, 3).unsqueeze(-1)  # [Batch, 1]
        chosen_card_id = torch.gather(hand_cards, 1, clamped_slot).squeeze(-1)  # [Batch]

        if card_tile_mask is not None:
            slots = chosen_card_slot.clamp(0, 4).view(batch_size, 1, 1).expand(-1, 1, TOTAL_TILES)
            effective_tile_mask = torch.gather(card_tile_mask.bool(), 1, slots).squeeze(1)
        else:
            is_spell = torch.isin(chosen_card_id, self.spell_ids) & (chosen_card_slot != 4)
            effective_tile_mask = torch.where(is_spell.unsqueeze(-1), torch.ones_like(base_tile_mask), base_tile_mask)

        # Step 3: Autoregressive Tile Prediction conditioned on chosen card
        chosen_emb = self.chosen_action_emb(chosen_card_slot)
        tile_input = torch.cat([core, chosen_emb], dim=-1)
        tile_logits = self.tile_head(tile_input).float()  # FP16-safe FP32 cast

        masked_tile_logits = tile_logits.masked_fill(~effective_tile_mask, -1e9)
        tile_dist = Categorical(logits=masked_tile_logits)

        if action is None:
            if deterministic:
                chosen_tile = masked_tile_logits.argmax(dim=-1)
            else:
                chosen_tile = tile_dist.sample()
            action = torch.stack([chosen_card_slot, chosen_tile], dim=-1)
        else:
            chosen_tile = action[:, 1]

        # For Wait/No-Op (slot == 4), tile placement has zero effect on the game transition probability
        is_play_action = (chosen_card_slot != 4).float()
        total_log_prob = card_dist.log_prob(chosen_card_slot) + is_play_action * tile_dist.log_prob(chosen_tile)

        # AlphaStar separate entropy regularization:
        # card_entropy regularizes card selection, while tile_entropy maintains active spatial exploration
        card_entropy = card_dist.entropy()
        tile_entropy = tile_dist.entropy()
        total_entropy = card_entropy + tile_entropy

        if return_entropy_breakdown:
            return action, total_log_prob, (card_entropy, tile_entropy), value
        return action, total_log_prob, total_entropy, value

    def get_value(self, obs: dict) -> torch.Tensor:
        """Critic-only evaluation for GAE bootstrapping."""
        core = self.forward_core(
            obs["scalars"], obs["hand_cards"], obs["entities"],
            obs["entity_mask"], obs["arena_grid"]
        )
        return self.value_head(core).squeeze(-1)

