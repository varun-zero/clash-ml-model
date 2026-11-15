"""Competitive v2 training entry point.

Usage:
    python train_competitive.py --config configs/competitive.yaml
    python train_competitive.py --config configs/competitive.yaml --smoke-test
    python train_competitive.py --config configs/competitive.yaml --init-bc checkpoints/competitive/bc_init.pt
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cr_alphastar_agent.competitive.bc import build_model_from_config
from cr_alphastar_agent.competitive.cards import (
    CardVocab, deck_pool_from_config, default_vocab, vocab_from_checkpoint,
)
from cr_alphastar_agent.competitive.env import CompetitiveEnv, CompetitiveEnvConfig
from cr_alphastar_agent.competitive.evaluation import evaluate_paired
from cr_alphastar_agent.competitive.league import (
    League, Matchup, ModelOpponent, ScriptedOpponent, load_competitive_model,
)
from cr_alphastar_agent.competitive.model import CompetitiveNet
from cr_alphastar_agent.competitive.obs import GLOBAL_FEATURE_DIM, UNIT_FEATURE_DIM, collate_observations
from cr_alphastar_agent.competitive.policy import ActionSampler
from cr_alphastar_agent.competitive.ppo import (
    PPOConfig, PPOUpdater, RolloutBuffer, compute_gae, normalize_advantages,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg.setdefault("train", {})
    cfg.setdefault("model", {})
    cfg.setdefault("env", {})
    cfg.setdefault("league", {})
    return cfg


def make_model(cfg: dict, n_cards: int) -> CompetitiveNet:
    m = cfg.get("model", {})
    return build_model_from_config(m, n_cards)


def build_eval_decks(cfg: dict, vocab: CardVocab, deck_pool: list[list[str]]) -> list[list[str]]:
    league_cfg = cfg.get("league", {})
    explicit = league_cfg.get("eval_decks") or []
    if explicit:
        return [list(d) for d in explicit]
    n_random = int(league_cfg.get("eval_random_decks", 4))
    rng = random.Random(int(cfg.get("seed", 7)) + 991)
    pool_keys = {tuple(sorted(d)) for d in deck_pool}
    eval_decks = []
    # Prefer held-out random decks; they are less likely to be in the training pool.
    for _ in range(n_random * 8):
        if len(eval_decks) >= n_random:
            break
        try:
            d = vocab.random_deck(rng)
        except ValueError:
            break
        if tuple(sorted(d)) not in pool_keys:
            eval_decks.append(d)
    if not eval_decks:
        eval_decks = [list(d) for d in deck_pool[:4]]
    return eval_decks


def make_eval_matchups(cfg: dict, vocab: CardVocab, deck_pool: list[list[str]],
                       pairs: int | None = None) -> list[tuple[list[str], list[str], int]]:
    eval_decks = build_eval_decks(cfg, vocab, deck_pool)
    pairs = int(cfg.get("train", {}).get("eval_pairs", 8) if pairs is None else pairs)
    rng = random.Random(int(cfg.get("seed", 7)) + 17)
    matchups = []
    for i in range(pairs):
        learner = eval_decks[i % len(eval_decks)]
        # Pair against a different deck when possible, and a fixed seed.
        opp = eval_decks[(i + 1 + len(eval_decks) // 2) % len(eval_decks)]
        matchups.append((learner, opp, rng.randrange(2**31)))
    return matchups


def evaluate(model, cfg, vocab, deck_pool, device, opponent_factory, opponent_name, pairs=None):
    matchups = make_eval_matchups(cfg, vocab, deck_pool, pairs=pairs)
    return evaluate_paired(
        model, device, vocab, deck_pool, matchups, opponent_factory,
        greedy=bool(cfg.get("train", {}).get("eval_greedy", True)),
        opponent_name=opponent_name,
    )


def save_checkpoint(path: Path, model: CompetitiveNet, prior: CompetitiveNet,
                    optimizer: torch.optim.Optimizer, update: int, step: int,
                    beta: float, ppo_cfg: PPOConfig, vocab: CardVocab,
                    cfg: dict, league: League) -> None:
    payload = {
        "model_state": model.state_dict(),
        "prior_state": prior.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config(),
        "vocab_names": vocab.names,
        "update": int(update),
        "step": int(step),
        "beta": float(beta),
        "ppo_config": dataclasses.asdict(ppo_cfg),
        "train_config": cfg,
        "deck_pool": [list(d) for d in league.deck_pool],
        "league_snapshots": [s.to_dict() for s in league.snapshots],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_checkpoint(path: str | Path, model: CompetitiveNet, prior: CompetitiveNet,
                    optimizer: torch.optim.Optimizer | None, device: torch.device) -> dict:
    payload = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    if "prior_state" in payload:
        prior.load_state_dict(payload["prior_state"], strict=True)
    else:
        prior.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


class Trainer:
    def __init__(self, cfg: dict, args):
        self.cfg = cfg
        self.args = args
        self.seed = int(cfg.get("seed", 7))
        set_seed(self.seed)
        self.device = torch.device(args.device if args.device != "auto"
                                   else ("cuda" if torch.cuda.is_available() else "cpu"))
        out_dir = Path(args.out or cfg.get("output_dir", "checkpoints/competitive"))
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir

        # Peek at a resume checkpoint before building the model so the vocab and
        # architecture come from the checkpoint, not from whatever config is
        # currently on disk.
        resume_path = None
        resume_payload = None
        if args.resume:
            resume_path = (Path(args.resume)
                           if isinstance(args.resume, str) and args.resume != "latest"
                           else out_dir / "latest.pt")
            if resume_path.exists():
                resume_payload = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        self.vocab = vocab_from_checkpoint(resume_payload) if resume_payload else default_vocab()
        league_cfg = cfg.get("league", {})
        self.deck_pool = deck_pool_from_config(league_cfg, self.vocab, seed=self.seed)
        if not self.deck_pool:
            raise RuntimeError("empty deck pool")
        if resume_payload and resume_payload.get("model_config"):
            self.model = build_model_from_config(resume_payload["model_config"], self.vocab.n_cards).to(self.device)
        else:
            self.model = make_model(cfg, self.vocab.n_cards).to(self.device)
        self.prior = copy.deepcopy(self.model).to(self.device).eval()
        for p in self.prior.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg["train"].get("lr", 3e-4)),
            weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
        )
        self.ppo_cfg = PPOConfig(
            gamma=float(cfg["train"].get("gamma", 0.997)),
            gae_lambda=float(cfg["train"].get("gae_lambda", 0.95)),
            clip=float(cfg["train"].get("clip", 0.2)),
            value_coef=float(cfg["train"].get("value_coef", 0.5)),
            value_ce_coef=float(cfg["train"].get("value_ce_coef", 0.25)),
            entropy_coef=float(cfg["train"].get("entropy_coef", 0.01)),
            aux_coef=float(cfg["train"].get("aux_coef", 0.05)),
            wait_coef=float(cfg["train"].get("wait_coef", 0.02)),
            kl_coef=float(cfg["train"].get("kl_coef", 0.5)),
            kl_target=float(cfg["train"].get("kl_target", 0.02)),
            kl_adapt=float(cfg["train"].get("kl_adapt", 1.5)),
            kl_min=float(cfg["train"].get("kl_min", 0.005)),
            kl_max=float(cfg["train"].get("kl_max", 5.0)),
            kl_leash=str(cfg["train"].get("kl_leash", "sum")),
            advantage=str(cfg["train"].get("advantage", "gae")),
            shaping=str(cfg["train"].get("shaping", "potential")),
            max_grad_norm=float(cfg["train"].get("max_grad_norm", 1.0)),
            epochs=int(cfg["train"].get("ppo_epochs", 4)),
            minibatch=int(cfg["train"].get("minibatch", 64)),
            normalize_advantage=bool(cfg["train"].get("normalize_advantage", True)),
            critic_warmup_updates=int(cfg["train"].get("critic_warmup_updates", 0)),
        )
        self.beta = float(cfg["train"].get("kl_coef", 0.5))
        self.pool_dir = out_dir / "pool"
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.league = League(
            league_cfg, self.vocab, self.deck_pool,
            checkpoint_dir=self.pool_dir, device=self.device,
            seed=self.seed, current_model=self.model,
        )
        self.updater = PPOUpdater(self.model, self.prior, self.optimizer, self.ppo_cfg, self.device)
        self.env_cfg = CompetitiveEnvConfig(
            decision_ticks=int(cfg["env"].get("decision_ticks", 5)),
            max_units=int(cfg["env"].get("max_units", 192)),
            n_particles=int(cfg["env"].get("n_particles", 128)),
            tower_weight=float(cfg["env"].get("tower_weight", 0.5)),
            crown_weight=float(cfg["env"].get("crown_weight", 0.15)),
            start_elixir=float(cfg["env"].get("start_elixir", 5.0)),
        )
        self.num_envs = max(1, int(cfg["train"].get("num_envs", 4)))
        self.rollout_steps = max(1, int(cfg["train"].get("rollout_steps", 256)))
        self.updates = max(1, int(cfg["train"].get("updates", 10000)))
        self.group_size = max(1, int(cfg["train"].get("group_size", self.num_envs)))
        self.train_start = time.time()
        self.total_steps = 0
        self.update_idx = 0
        self.match_counter = 0
        self.group_queue: list[Matchup] = []
        self.envs = [CompetitiveEnv(self.deck_pool, self.env_cfg, seed=self.seed + i, vocab=self.vocab)
                     for i in range(self.num_envs)]
        self.obs_list: list[Any] = [None] * self.num_envs
        self.matchups: list[Matchup | None] = [None] * self.num_envs
        self.match_ids: list[int] = [-1] * self.num_envs
        self.sampler = ActionSampler(self.model, self.device, seed=self.seed + 12345)
        self.metrics_path = self.out_dir / "metrics.jsonl"
        self.eval_baselines: dict[str, Any] = {}

        if args.init_bc:
            self._load_bc(args.init_bc)
        if args.resume:
            if resume_payload is not None and resume_path is not None:
                payload = load_checkpoint(resume_path, self.model, self.prior, self.optimizer, self.device)
                self.total_steps = int(payload.get("step", 0))
                self.update_idx = int(payload.get("update", 0))
                self.beta = float(payload.get("beta", self.beta))
                # Restore PPO fields that may have changed in config.
                saved = payload.get("ppo_config") or {}
                for k, v in saved.items():
                    if hasattr(self.ppo_cfg, k):
                        setattr(self.ppo_cfg, k, v)
                print(f"[resume] {resume_path} update={self.update_idx} step={self.total_steps} beta={self.beta:.3f}")
                if payload.get("vocab_names"):
                    print(f"[resume] vocab={len(payload['vocab_names'])} cards")
            else:
                print(f"[resume] {resume_path or out_dir / 'latest.pt'} not found; starting fresh")
        # Build frozen baselines once.
        for s in self.league.snapshots[-2:]:
            try:
                m, _, _ = load_competitive_model(s.path, self.device, self.vocab)
                self.eval_baselines[s.name] = m
            except Exception as exc:
                print(f"[eval] could not load snapshot {s.path}: {exc}")

    def _load_bc(self, path: str | Path) -> None:
        payload = torch.load(str(path), map_location=self.device, weights_only=False)
        state = payload.get("model_state") or payload.get("state_dict") or payload
        self.model.load_state_dict(state, strict=True)
        self.prior.load_state_dict(state, strict=True)
        self.prior.eval()
        for p in self.prior.parameters():
            p.requires_grad_(False)
        print(f"[init] loaded BC checkpoint {path}")

    # ------------------------------------------------------------------ match
    def _sample_matchup(self) -> Matchup:
        if self.ppo_cfg.advantage != "match_loo":
            return self.league.sample_matchup(self.update_idx)
        if not self.group_queue:
            base = self.league.sample_matchup(self.update_idx)
            for j in range(self.group_size):
                self.group_queue.append(dataclasses.replace(
                    base,
                    seed=int(base.seed + j * 7919),
                    learner_team=(base.learner_team + j) % 2,
                    group_id=base.group_id,
                ))
        return self.group_queue.pop(0)

    def _reset_env(self, i: int) -> None:
        m = self._sample_matchup()
        self.match_counter += 1
        opponent = self.league.make_opponent(m)
        self.obs_list[i] = self.envs[i].reset(
            m.learner_deck, m.opponent_deck, seed=m.seed,
            learner_team=m.learner_team, opponent=opponent, opponent_name=m.opponent_name,
        )
        self.matchups[i] = m
        self.match_ids[i] = self.match_counter

    def _collect(self) -> RolloutBuffer:
        buffer = RolloutBuffer()
        for i in range(self.num_envs):
            if self.obs_list[i] is None:
                self._reset_env(i)
        self.model.eval()
        for _ in range(self.rollout_steps):
            obs_batch = collate_observations(self.obs_list)
            actions, meta = self.sampler.sample(obs_batch)
            for i in range(self.num_envs):
                obs = self.obs_list[i]
                action = actions[i]
                meta_i = {k: v[i:i + 1] for k, v in meta.items()}
                m = self.matchups[i]
                _, reward, done, info = self.envs[i].step(action)
                buffer.add(
                    obs, action, meta_i, reward=reward, done=done,
                    env_id=i, match_id=self.match_ids[i],
                    group_id=m.group_id if m else 0,
                )
                if done:
                    self.league.record_result(
                        m.snapshot_path if m else None,
                        learner_won=(info["result"] == self.envs[i].learner_team),
                        draw=(info["result"] is None),
                    )
                    self._reset_env(i)
                else:
                    self.obs_list[i] = self.envs[i].observe_learner()
                self.total_steps += 1
        buffer.finalize_wait_targets()
        # Bootstrap the value of the state after the final rollout decision.
        boot = self.sampler.values(self.obs_list).tolist()
        buffer.set_bootstrap({
            i: (boot[i], float(self.obs_list[i].phi))
            for i in range(self.num_envs)
        })
        return buffer

    # ------------------------------------------------------------------ update
    def _train_step(self) -> dict[str, float]:
        buffer = self._collect()
        batch = buffer.build_batch()
        compute_gae(batch, self.ppo_cfg)
        if self.ppo_cfg.normalize_advantage:
            normalize_advantages(batch)
        self.model.train()
        metrics = self.updater.update(batch, self.beta)
        self.beta = self.updater.adapt_beta(self.beta, metrics.get("kl", 0.0))
        metrics["beta"] = self.beta
        metrics["buffer"] = float(len(buffer))
        metrics["total_steps"] = float(self.total_steps)
        return metrics

    # -------------------------------------------------------------------- run
    def run(self) -> None:
        save_every = int(self.cfg["train"].get("save_every", 10))
        snapshot_every = int(self.cfg["train"].get("snapshot_every", 20))
        eval_every = int(self.cfg["train"].get("eval_every", 20))
        log_every = int(self.cfg["train"].get("log_every", 1))
        for _ in range(self.updates):
            t0 = time.time()
            metrics = self._train_step()
            self.update_idx += 1
            metrics["update"] = float(self.update_idx)
            metrics["seconds"] = time.time() - t0
            with self.metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            if self.update_idx % log_every == 0:
                print(
                    f"[{self.update_idx}] step={self.total_steps} "
                    f"pg={metrics['pg']:.4f} vloss={metrics['value']:.4f} ent={metrics['entropy']:.3f} "
                    f"kl={metrics['kl']:.4f} beta={self.beta:.4f} aux={metrics['aux']:.3f} "
                    f"ratio={metrics['ratio']:.3f} ({metrics['seconds']:.1f}s)"
                )
            if self.update_idx % save_every == 0:
                save_checkpoint(self.out_dir / "latest.pt", self.model, self.prior,
                                self.optimizer, self.update_idx, self.total_steps,
                                self.beta, self.ppo_cfg, self.vocab, self.cfg, self.league)
            if snapshot_every > 0 and self.update_idx % snapshot_every == 0:
                snap_path = self.pool_dir / f"step_{self.total_steps:09d}.pt"
                save_checkpoint(snap_path, self.model, self.prior, self.optimizer,
                                self.update_idx, self.total_steps, self.beta,
                                self.ppo_cfg, self.vocab, self.cfg, self.league)
                self.league.add_snapshot(snap_path, self.total_steps)
                print(f"[snapshot] {snap_path}")
            if eval_every > 0 and self.update_idx % eval_every == 0:
                self._evaluate()
        save_checkpoint(self.out_dir / "latest.pt", self.model, self.prior,
                        self.optimizer, self.update_idx, self.total_steps,
                        self.beta, self.ppo_cfg, self.vocab, self.cfg, self.league)
        print(f"[done] update={self.update_idx} step={self.total_steps} out={self.out_dir}")

    def _evaluate(self) -> None:
        pairs = int(self.cfg["train"].get("eval_pairs", 8))
        self.model.eval()
        result = evaluate(
            self.model, self.cfg, self.vocab, self.deck_pool, self.device,
            lambda: ScriptedOpponent("tactical", seed=1234), "scripted:tactical", pairs=pairs,
        )
        print(f"[eval {self.update_idx}] vs scripted: {result.wins}W/{result.losses}L/{result.draws}D "
              f"wr={result.win_rate:.3f} CI95=[{result.ci_low:.3f},{result.ci_high:.3f}]")
        record = {"update": self.update_idx, "step": self.total_steps,
                  "opponent": "scripted:tactical", **result.to_dict()}
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"eval": record}) + "\n")
        for name, baseline in list(self.eval_baselines.items())[-1:]:
            res = evaluate(
                self.model, self.cfg, self.vocab, self.deck_pool, self.device,
                lambda b=baseline, n=name: ModelOpponent(b, self.device, n, greedy=True),
                name, pairs=pairs,
            )
            print(f"[eval {self.update_idx}] vs {name}: {res.wins}W/{res.losses}L/{res.draws}D "
                  f"wr={res.win_rate:.3f} CI95=[{res.ci_low:.3f},{res.ci_high:.3f}]")
            record = {"update": self.update_idx, "step": self.total_steps,
                      "opponent": name, **res.to_dict()}
            with self.metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"eval": record}) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the competitive v2 Clash Royale agent")
    ap.add_argument("--config", default="configs/competitive.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--snapshot-every", type=int, default=None)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--resume", nargs="?", const="latest", default=None)
    ap.add_argument("--init-bc", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.smoke_test:
        cfg["train"]["updates"] = 1
        cfg["train"]["num_envs"] = 2
        cfg["train"]["rollout_steps"] = 8
        cfg["train"]["ppo_epochs"] = 1
        cfg["train"]["minibatch"] = 8
        cfg["train"]["eval_every"] = 0
        cfg["train"]["save_every"] = 1000
        cfg["train"]["snapshot_every"] = 0
        cfg["league"]["random_decks"] = 0
        cfg["league"]["include_presets"] = True
    if args.updates is not None:
        cfg["train"]["updates"] = int(args.updates)
    if args.num_envs is not None:
        cfg["train"]["num_envs"] = int(args.num_envs)
    if args.rollout_steps is not None:
        cfg["train"]["rollout_steps"] = int(args.rollout_steps)
    if args.eval_every is not None:
        cfg["train"]["eval_every"] = int(args.eval_every)
    if args.save_every is not None:
        cfg["train"]["save_every"] = int(args.save_every)
    if args.snapshot_every is not None:
        cfg["train"]["snapshot_every"] = int(args.snapshot_every)
    trainer = Trainer(cfg, args)
    n_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"[setup] device={trainer.device} params={n_params:,} cards={trainer.vocab.n_cards} "
          f"decks={len(trainer.deck_pool)} envs={trainer.num_envs} updates={trainer.updates}")
    trainer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
