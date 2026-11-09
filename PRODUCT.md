# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The primary user is the project owner, who develops and evaluates an AI agent for Clash Royale. They watch matches, play against bots, inspect agent decisions, and review training behavior.

## Product Purpose

ClashBot is a private research simulator and training harness for studying game-playing agents. Success means being able to run, understand, and compare simulated matches and policy behavior.

## Positioning

ClashBot combines a data-driven battle simulator with a trainable policy and a live interface that lets the owner inspect and interact with matches.

## Operating Context

The web UI is used during local simulator runs. The main tasks are watching or stepping through a match, selecting decks, controlling playback, and comparing match state with policy diagnostics.

## Capabilities and Constraints

- The interface supports manual play, bot play, deck selection, randomization, pause, reset, speed control, replay export, and live telemetry.
- The arena, score, timer, hand, and match controls must remain easy to locate while diagnostics are available as secondary information.
- Simulation and policy behavior are approximate and must not be presented as live-game results.

## Brand Commitments

- Keep the interface dark and modern, with the quiet clarity the owner associates with Notion, ChatGPT, and Vercel.
- Use those products as references for restraint, not as templates to copy.
- Preserve the ClashBot identity and use restrained, purposeful color instead of a single saturated blue or decorative neon.

## Evidence on Hand

- Source implementation: `tools/ui.py` and `tools/ui_template.html`.
- The simulator and training code are in `crsim/` and `cr_alphastar_agent/`.
- The screenshot provided by the owner shows the current screen’s clipping and density problems.
- No separate brand assets or verified live-game claims were supplied.

## Product Principles

- Keep the match readable at a glance.
- Make controls predictable and available during play.
- Make diagnostics useful without letting them compete with the arena.
- Present simulated data honestly.
