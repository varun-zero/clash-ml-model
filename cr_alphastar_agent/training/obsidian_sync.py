"""Update the project telemetry note and an optional configured Obsidian note."""

import os
import re
import datetime
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_VAULT_NOTE = os.path.join(PROJECT_ROOT, "CLASHBOT_OBSIDIAN_NOTE.md")


def _note_targets() -> list[str]:
    """Return the local note plus an optional user-configured Obsidian note."""
    targets = [os.environ.get("CLASHBOT_OBSIDIAN_NOTE_PATH"), LOCAL_VAULT_NOTE]
    return list(dict.fromkeys(path for path in targets if path))


def sync_batch_to_obsidian(update_count: int, global_step: int,
                           metrics: Dict[str, float], fps: int,
                           total_time: float, phase_desc: str = "Stage 3: Pure AlphaStar Self-Play",
                           history_data: List[dict] = None):
    """Pushes live batch metrics and training curve into the Obsidian vault note."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Re-render PNG graph if history is provided
    if history_data:
        try:
            from cr_alphastar_agent.training.plot_progress import generate_training_plot
            generate_training_plot(history_data)
        except Exception:
            pass

    cur_w = int(metrics.get("total_wins", 0))
    cur_l = int(metrics.get("total_losses", 0))
    cur_d = int(metrics.get("total_draws", 0))
    cur_matches = int(metrics.get("total_matches", cur_w + cur_l + cur_d))
    cur_wr = metrics.get("win_rate", (cur_w / max(1, cur_matches)) * 100.0)

    row = (
        f"| Update {update_count:04d} | {global_step:,} | {metrics.get('pg_loss', 0.0):.4f} | "
        f"{metrics.get('v_loss', 0.0):.4f} | {metrics.get('entropy', 0.0):.3f} "
        f"({metrics.get('card_entropy_active', 0.0):.2f}/{metrics.get('tile_entropy', 0.0):.2f}) | "
        f"{metrics.get('approx_kl', 0.0):.5f} | {cur_wr:.1f}% ({cur_w}W-{cur_l}L-{cur_d}D) | {fps} FPS | {phase_desc} |\n"
    )

    for target_path in _note_targets():
        if not os.path.exists(target_path):
            continue

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 1. Update frontmatter fields
            content = re.sub(r'latest_checkpoint:\s*".*?"', f'latest_checkpoint: "Step {global_step:,} (Update {update_count})"', content)
            if "status:" in content:
                content = re.sub(r'status:\s*.*', f'status: training (live - {now_str})', content)
            if "win_rate:" in content:
                content = re.sub(r'win_rate:\s*".*?"', f'win_rate: "{cur_wr:.1f}% vs AI Self-Play"', content)

            # 2. Ensure Performance Graph Section exists
            graph_section = (
                "## 📊 Self-Play Performance Graph\n\n"
                f"> [!summary] Genuine Head-to-Head Telemetry\n"
                f"> **Total Matches:** {cur_matches:,} | **Wins:** {cur_w} ({cur_wr:.1f}%) | **Losses:** {cur_l} | **Draws:** {cur_d}\n\n"
                "![[clashbot_training_curve.png]]\n\n"
            )

            if "## 📊 Self-Play Performance Graph" in content:
                pattern = r"## 📊 Self-Play Performance Graph.*?(?=\n---|\n## |\Z)"
                content = re.sub(pattern, graph_section.strip() + "\n", content, flags=re.DOTALL)
            else:
                # Insert right before Live Training Telemetry
                if "## 📈 Live Training Telemetry" in content:
                    content = content.replace("## 📈 Live Training Telemetry", f"{graph_section}## 📈 Live Training Telemetry")
                else:
                    content += f"\n\n{graph_section}"

            # 3. Update Tactical Placement Zone Telemetry if available
            zone_pcts = metrics.get("tactical_zone_pcts", {})
            if zone_pcts:
                tactical_section = (
                    "## 🎯 Tactical Placement Zone Telemetry\n\n"
                    f"> [!abstract] Live Spatial Deployment Distribution (Batch Telemetry)\n"
                    f"> **Last Evaluated Batch:** Update {update_count:04d} (Step {global_step:,}) | **Time:** {now_str}\n\n"
                    "| Tactical Zone | Canonical Coordinates | Deployment % | Micro Role |\n"
                    "| :--- | :--- | :---: | :--- |\n"
                    f"| **Back-of-King Slow-Roll** | `y <= 4.0` | **{zone_pcts.get('Back-of-King Slow-Roll', 0.0):.1f}%** | Banking elixir, slow beatdown investment |\n"
                    f"| **Center Split** | `7.5 <= x <= 10.5, 4 < y <= 8` | **{zone_pcts.get('Center Split', 0.0):.1f}%** | Dual-lane pressure & splitting swarms |\n"
                    f"| **Center Kill-Zone / Kite** | `7.0 <= x <= 11.0, 8 < y <= 13` | **{zone_pcts.get('Center Kill-Zone / Kite', 0.0):.1f}%** | Opposite-lane kiting & dual-tower crossfire |\n"
                    f"| **Bridge / Tank-Trick** | `13.0 < y <= 15.5` | **{zone_pcts.get('Bridge / Tank-Trick', 0.0):.1f}%** | River pressure & meat-shielding glass cannons |\n"
                    f"| **Deep Enemy Spell** | `y > 16.0` | **{zone_pcts.get('Deep Enemy Spell', 0.0):.1f}%** | Direct tower damage & defensive troop snipes |\n"
                    f"| **Flank Lane Defense** | `x < 7.0` or `x > 11.0`, `4 < y <= 13` | **{zone_pcts.get('Flank Lane Defense', 0.0):.1f}%** | Direct lane response against incoming troops |\n\n"
                )
                if "## 🎯 Tactical Placement Zone Telemetry" in content:
                    pattern = r"## 🎯 Tactical Placement Zone Telemetry.*?(?=\n---|\n## |\Z)"
                    content = re.sub(pattern, tactical_section.strip() + "\n", content, flags=re.DOTALL)
                else:
                    if "## 📈 Live Training Telemetry" in content:
                        content = content.replace("## 📈 Live Training Telemetry", f"{tactical_section}## 📈 Live Training Telemetry")
                    else:
                        content += f"\n\n{tactical_section}"

            # 4. Update Live Telemetry Section
            section_header = "## 📈 Live Training Telemetry"
            table_header = (
                f"{section_header}\n\n"
                f"> [!tip] Live Training Status\n"
                f"> **Active Checkpoint:** Update {update_count:04d} (Step {global_step:,}) | **Speed:** {fps} FPS | **Cadence:** {total_time:.2f}s/batch | **Last Synced:** {now_str}\n"
                f"> **AI vs AI Record:** **{cur_w} Wins** • **{cur_l} Losses** • **{cur_d} Draws** ({cur_wr:.1f}% Win Rate)\n\n"
                "| Update | Step | Policy Loss | Value Loss | Entropy (Card/Tile) | KL Div | Record & Win % | Speed | Curriculum |\n"
                "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
            )

            if section_header in content:
                pattern = rf"{re.escape(section_header)}.*?(?=\n---|\n## |\Z)"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    existing_section = match.group(0)
                    existing_rows = [
                        line for line in existing_section.split("\n")
                        if re.match(r"^\|\s*Update\s+\d+", line)
                    ]
                    updated_rows = [row.strip()] + existing_rows[:14]  # Keep latest 15 rows
                    new_section = table_header + "\n".join(updated_rows) + "\n\n"
                    content = content[:match.start()] + new_section + content[match.end():]
            else:
                new_section = f"\n\n{table_header}{row}\n"
                if "\n---\n*Generated" in content:
                    idx = content.rfind("\n---\n*Generated")
                    content = content[:idx] + new_section + content[idx:]
                else:
                    content += new_section

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
            pass


def sync_tactical_zones_to_obsidian(tactical_zones: Dict[str, int], total_matches: int = 10,
                                    win_rate: float = None, opp_name: str = "ScriptedBot"):
    """Pushes dedicated benchmark tactical zone distribution into the Obsidian vault note."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_deploys = sum(tactical_zones.values()) or 1
    zone_pcts = {k: (v / total_deploys) * 100.0 for k, v in tactical_zones.items()}

    wr_str = f" | **Win Rate:** {win_rate:.1f}% vs {opp_name}" if win_rate is not None else ""
    tactical_section = (
        "## 🎯 Tactical Placement Zone Telemetry\n\n"
        f"> [!abstract] Spatial Deployment Breakdown ({total_matches} Verified Evaluation Matches{wr_str})\n"
        f"> **Total Cards Deployed:** {total_deploys:,} | **Last Evaluated:** {now_str}\n\n"
        "| Tactical Zone | Canonical Coordinates | Deployed Cards | Percentage | Strategic Tactical Role |\n"
        "| :--- | :--- | :---: | :---: | :--- |\n"
        f"| **Back-of-King Slow-Roll** | `y <= 4.0` | {tactical_zones.get('Back-of-King Slow-Roll', 0)} | **{zone_pcts.get('Back-of-King Slow-Roll', 0.0):.1f}%** | Banking elixir & starting slow beatdown tanks |\n"
        f"| **Center Split** | `7.5 <= x <= 10.5, 4 < y <= 8` | {tactical_zones.get('Center Split', 0)} | **{zone_pcts.get('Center Split', 0.0):.1f}%** | Splitting dual-lane swarms behind King Tower |\n"
        f"| **Center Kill-Zone / Kite** | `7.0 <= x <= 11.0, 8 < y <= 13` | {tactical_zones.get('Center Kill-Zone / Kite', 0)} | **{zone_pcts.get('Center Kill-Zone / Kite', 0.0):.1f}%** | Opposite-lane troop kiting & dual-tower crossfire |\n"
        f"| **Bridge / Tank-Trick** | `13.0 < y <= 15.5` | {tactical_zones.get('Bridge / Tank-Trick', 0)} | **{zone_pcts.get('Bridge / Tank-Trick', 0.0):.1f}%** | River pressure & meat-shielding glass cannons |\n"
        f"| **Deep Enemy Spell** | `y > 16.0` | {tactical_zones.get('Deep Enemy Spell', 0)} | **{zone_pcts.get('Deep Enemy Spell', 0.0):.1f}%** | Direct Crown Tower damage & enemy troop snipes |\n"
        f"| **Flank Lane Defense** | `x < 7.0` or `x > 11.0`, `4 < y <= 13` | {tactical_zones.get('Flank Lane Defense', 0)} | **{zone_pcts.get('Flank Lane Defense', 0.0):.1f}%** | Direct defensive reactions against lane pushes |\n\n"
    )

    for target_path in _note_targets():
        if not os.path.exists(target_path):
            continue

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()

            if "## 🎯 Tactical Placement Zone Telemetry" in content:
                pattern = r"## 🎯 Tactical Placement Zone Telemetry.*?(?=\n---|\n## |\Z)"
                content = re.sub(pattern, tactical_section.strip() + "\n", content, flags=re.DOTALL)
            else:
                if "## 📈 Live Training Telemetry" in content:
                    content = content.replace("## 📈 Live Training Telemetry", f"{tactical_section}## 📈 Live Training Telemetry")
                else:
                    content += f"\n\n{tactical_section}"

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            pass
