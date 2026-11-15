"""Factored-action sampling and exact log-probabilities for the v2 policy.

Action decomposition
--------------------
    gate ~ Bernoulli(sigmoid(gate_logit))
    if gate:  card ~ Categorical(legal, affordable hand cards)
              cell ~ Categorical(legal cells for that card)
    else:     wait ~ Categorical(deck cards not currently in hand)  (optional)

The behaviour log-prob stored in the rollout is the exact log-prob of this
decomposition under the sampling temperature.  ``evaluate_actions`` recomputes
the same quantity (temperature 1, as used for training) for the PPO ratio.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .model import CompetitiveNet
from .obs import Action, Observation, collate_observations

BOOL_KEYS = {"unit_mask", "hand_mask", "deck_mask", "card_mask", "cell_masks", "wait_mask"}
ID_KEYS = {"unit_ids", "hand_ids", "deck_ids", "next_id"}


def obs_to_tensors(obs: Observation | Sequence[Observation] | dict,
                   device: torch.device) -> dict[str, torch.Tensor]:
    """Convert one observation, a list, or an already-collated dict to tensors."""
    if isinstance(obs, dict):
        batch = obs
    elif isinstance(obs, Observation):
        batch = collate_observations([obs])
    else:
        batch = collate_observations(list(obs))
    out: dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if k in ("aux_opp_elixir", "aux_opp_hand", "aux_opp_next", "phi",
                 "cell_mask", "gate", "hand_slot", "cell", "wait_slot",
                 "value_target", "source", "replay_id"):
            continue  # privileged labels / non-input dataset fields
        t = torch.as_tensor(v, device=device)
        out[k] = t.bool() if k in BOOL_KEYS else t.float()
    return out


def _playable_mask(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    """Affordable-and-placeable hand cards.  Falls back to the hand mask."""
    hand_mask = tensors["hand_mask"]
    card_mask = tensors.get("card_mask")
    return hand_mask if card_mask is None else (card_mask & hand_mask)


def _safe_cat_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked categorical logits.  Rows with no legal choice become uniform."""
    valid = mask.any(dim=-1, keepdim=True)
    logits = logits.masked_fill(~mask, -1e4)
    logits = torch.where(valid, logits, torch.zeros_like(logits))
    return logits


def masked_categorical_logprob(logits: torch.Tensor, mask: torch.Tensor,
                               idx: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(_safe_cat_logits(logits, mask), dim=-1)
    return logp.gather(-1, idx.long().clamp(min=0).unsqueeze(-1)).squeeze(-1)


def masked_categorical_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(_safe_cat_logits(logits, mask), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def sample_masked(logits: torch.Tensor, mask: torch.Tensor,
                  generator: torch.Generator | None = None,
                  greedy: bool = False) -> torch.Tensor:
    logits = _safe_cat_logits(logits, mask)
    if greedy:
        return logits.argmax(dim=-1)
    return torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator).squeeze(-1)


def binary_entropy(logit: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logit)
    return -(p * F.logsigmoid(logit) + (1 - p) * F.logsigmoid(-logit))


class ActionSampler:
    """Samples the factored action and returns exact behaviour log-probs."""

    def __init__(self, model: CompetitiveNet, device: torch.device, seed: int | None = None):
        self.model = model
        self.device = device
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

    def set_seed(self, seed: int) -> None:
        self.generator.manual_seed(int(seed))

    @torch.no_grad()
    def values(self, obs: Observation | Sequence[Observation] | dict) -> torch.Tensor:
        """Value estimates for bootstrapping a truncated rollout."""
        tensors = obs_to_tensors(obs, self.device)
        enc = self.model.encode(
            tensors["unit_ids"], tensors["unit_feats"], tensors["unit_mask"],
            tensors["global_feats"], tensors["hand_ids"], tensors["hand_mask"],
            tensors["deck_ids"], tensors["deck_mask"], tensors["next_id"],
            tensors["past_own"], tensors["past_opp"], tensors["belief"],
        )
        heads = self.model.heads(
            enc, tensors["hand_ids"], tensors["hand_mask"],
            tensors["deck_ids"], tensors["deck_mask"],
        )
        return heads["value"].squeeze(-1)

    @torch.no_grad()
    def sample(
        self,
        obs: Observation | Sequence[Observation] | dict,
        *,
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> tuple[list[Action], dict[str, torch.Tensor]]:
        tensors = obs_to_tensors(obs, self.device)
        enc = self.model.encode(
            tensors["unit_ids"], tensors["unit_feats"], tensors["unit_mask"],
            tensors["global_feats"], tensors["hand_ids"], tensors["hand_mask"],
            tensors["deck_ids"], tensors["deck_mask"], tensors["next_id"],
            tensors["past_own"], tensors["past_opp"], tensors["belief"],
        )
        heads = self.model.heads(
            enc, tensors["hand_ids"], tensors["hand_mask"],
            tensors["deck_ids"], tensors["deck_mask"],
        )
        temp = max(1e-6, float(temperature))
        playable = _playable_mask(tensors)
        can_play = playable.any(dim=-1)

        gate_logits = heads["gate"] / temp
        gate_logits = torch.where(can_play, gate_logits, torch.full_like(gate_logits, -30.0))
        p = torch.sigmoid(gate_logits)
        if greedy:
            gate = (p >= 0.5).long()
        else:
            u = torch.rand(p.shape, generator=self.generator, device=p.device, dtype=p.dtype)
            gate = (u < p).long()
        gate = torch.where(can_play, gate, torch.zeros_like(gate))

        card_logits = heads["card"] / temp
        card_choice = sample_masked(card_logits, playable,
                                    generator=self.generator, greedy=greedy)

        cell_masks = tensors["cell_masks"]
        chosen_cell_masks = cell_masks.gather(
            1, card_choice.view(-1, 1, 1).expand(-1, 1, cell_masks.shape[-1])
        ).squeeze(1)
        chosen_card_ids = tensors["hand_ids"].gather(1, card_choice.view(-1, 1)).squeeze(1)
        cell_logits = self.model.cell_logits(enc, chosen_card_ids) / temp
        cell = sample_masked(cell_logits, chosen_cell_masks,
                             generator=self.generator, greedy=greedy)
        cell = torch.where(gate.bool(), cell, torch.zeros_like(cell))

        wait_logits = heads["wait"] / temp
        wait_mask = tensors["wait_mask"]
        wait_choice = sample_masked(wait_logits, wait_mask,
                                    generator=self.generator, greedy=greedy)
        has_wait = wait_mask.any(dim=-1)
        wait_choice = torch.where((gate == 0) & has_wait, wait_choice,
                                  torch.full_like(wait_choice, -1))

        actions = [
            Action(gate=int(gate[i]), hand_slot=int(card_choice[i]),
                   cell=int(cell[i]), wait_slot=int(wait_choice[i]))
            for i in range(gate.shape[0])
        ]

        play = gate.bool()
        wait_rows = (~play) & has_wait
        logprob = logprob_from_logits(
            gate_logits, card_logits, cell_logits, wait_logits,
            gate, card_choice, cell, wait_choice,
            playable, chosen_cell_masks, wait_mask, has_wait,
        )
        entropy = (
            binary_entropy(gate_logits)
            + torch.where(play, masked_categorical_entropy(card_logits, playable),
                          torch.zeros_like(gate_logits))
            + torch.where(play, masked_categorical_entropy(cell_logits, chosen_cell_masks),
                          torch.zeros_like(gate_logits))
            + torch.where(wait_rows, masked_categorical_entropy(wait_logits, wait_mask),
                          torch.zeros_like(gate_logits))
        )
        meta = {
            "value": heads["value"].detach(),
            "value_logits": heads["value_logits"].detach(),
            "aux_elixir": heads["aux_elixir"].detach(),
            "aux_hand": heads["aux_hand"].detach(),
            "aux_next": heads["aux_next"].detach(),
            "chosen_card_ids": chosen_card_ids.detach(),
            "chosen_cell_mask": chosen_cell_masks.detach(),
            "playable_mask": playable.detach(),
            "wait_mask": wait_mask.detach(),
            "logprob": logprob.detach(),
            "entropy": entropy.detach(),
        }
        return actions, meta


def logprob_from_logits(
    gate_logits: torch.Tensor,
    card_logits: torch.Tensor,
    cell_logits: torch.Tensor,
    wait_logits: torch.Tensor,
    gate: torch.Tensor,
    card: torch.Tensor,
    cell: torch.Tensor,
    wait: torch.Tensor,
    card_mask: torch.Tensor,
    cell_mask: torch.Tensor,
    wait_mask: torch.Tensor,
    has_wait: torch.Tensor | None = None,
) -> torch.Tensor:
    gate = gate.long()
    play = gate == 1
    lp_gate = torch.where(play, F.logsigmoid(gate_logits), F.logsigmoid(-gate_logits))
    lp_card = masked_categorical_logprob(card_logits, card_mask, card)
    lp_cell = masked_categorical_logprob(cell_logits, cell_mask, cell)
    if has_wait is None:
        has_wait = wait_mask.any(dim=-1)
    wait_ok = (~play) & has_wait & (wait >= 0)
    lp_wait = masked_categorical_logprob(wait_logits, wait_mask, wait.clamp(min=0))
    return lp_gate + torch.where(
        play, lp_card + lp_cell,
        torch.where(wait_ok, lp_wait, torch.zeros_like(lp_gate)),
    )


def policy_distributions(
    model: CompetitiveNet,
    tensors: dict[str, torch.Tensor],
    card_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Factor distributions; logits are raw, masks are returned alongside."""
    enc = model.encode(
        tensors["unit_ids"], tensors["unit_feats"], tensors["unit_mask"],
        tensors["global_feats"], tensors["hand_ids"], tensors["hand_mask"],
        tensors["deck_ids"], tensors["deck_mask"], tensors["next_id"],
        tensors["past_own"], tensors["past_opp"], tensors["belief"],
    )
    heads = model.heads(
        enc, tensors["hand_ids"], tensors["hand_mask"],
        tensors["deck_ids"], tensors["deck_mask"],
    )
    playable = _playable_mask(tensors)
    can_play = playable.any(dim=-1)
    gate = torch.where(can_play, heads["gate"], torch.full_like(heads["gate"], -30.0))
    return {
        "gate": gate,
        "card": heads["card"],
        "card_mask": playable,
        "cell": model.cell_logits(enc, card_ids),
        "wait": heads["wait"],
        "wait_mask": tensors["wait_mask"],
        "value": heads["value"],
        "value_logits": heads["value_logits"],
        "aux_elixir": heads["aux_elixir"],
        "aux_hand": heads["aux_hand"],
        "aux_next": heads["aux_next"],
    }


def evaluate_actions(
    model: CompetitiveNet,
    tensors: dict[str, torch.Tensor],
    actions: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Recompute log-probs, entropy and values for stored actions (PPO)."""
    dist = policy_distributions(model, tensors, actions["card"])
    has_wait = dist["wait_mask"].any(dim=-1)
    logprob = logprob_from_logits(
        dist["gate"], dist["card"], dist["cell"], dist["wait"],
        actions["gate"], actions["card"], actions["cell"], actions["wait"],
        dist["card_mask"], actions["cell_mask"], dist["wait_mask"], has_wait,
    )
    play = actions["gate"].bool()
    wait_rows = (~play) & has_wait & (actions["wait"] >= 0)
    entropy = (
        binary_entropy(dist["gate"])
        + torch.where(play, masked_categorical_entropy(dist["card"], dist["card_mask"]),
                      torch.zeros_like(dist["gate"]))
        + torch.where(play, masked_categorical_entropy(dist["cell"], actions["cell_mask"]),
                      torch.zeros_like(dist["gate"]))
        + torch.where(wait_rows, masked_categorical_entropy(dist["wait"], dist["wait_mask"]),
                      torch.zeros_like(dist["gate"]))
    )
    return {
        "logprob": logprob,
        "entropy": entropy,
        "value": dist["value"].squeeze(-1),
        "value_logits": dist["value_logits"],
        "aux_elixir": dist["aux_elixir"],
        "aux_hand": dist["aux_hand"],
        "aux_next": dist["aux_next"],
        "gate": dist["gate"],
        "card": dist["card"],
        "cell": dist["cell"],
        "wait": dist["wait"],
        "card_mask": dist["card_mask"],
        "wait_mask": dist["wait_mask"],
    }


def categorical_kl(logits_p: torch.Tensor, logits_q: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) for masked categoricals; rows with no legal choice cost 0."""
    valid = mask.any(dim=-1)
    p = torch.softmax(_safe_cat_logits(logits_p, mask), dim=-1)
    lp = torch.log_softmax(_safe_cat_logits(logits_p, mask), dim=-1)
    lq = torch.log_softmax(_safe_cat_logits(logits_q, mask), dim=-1)
    kl = (p * (lp - lq)).sum(-1)
    return torch.where(valid, kl, torch.zeros_like(kl))


def bernoulli_kl(logit_p: torch.Tensor, logit_q: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logit_p)
    lp = F.logsigmoid(logit_p)
    l1p = F.logsigmoid(-logit_p)
    lq = F.logsigmoid(logit_q)
    l1q = F.logsigmoid(-logit_q)
    return p * (lp - lq) + (1.0 - p) * (l1p - l1q)
