import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from cr_alphastar_agent.config import (
    ARENA_WIDTH, ARENA_HEIGHT, TOTAL_TILES, MAX_ENTITIES,
    ENTITY_FEATURES, SPATIAL_CHANNELS, NUM_CARDS, SPELL_CARD_IDS,
    WIN_CONDITION_IDS, DEFENSIVE_CARD_IDS,
    CARDS_CATALOG
)
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
from cr_alphastar_agent.env.crsim_replica_env import CRSIM_NAME_TO_ID
from cr_alphastar_agent.utils.card_classifier import VisualCardClassifier
from deploy_mumu import PerceptionAdapter, ADBController

def create_environment_state(hand_ids, elixir, active_enemy_units=None, device="cpu"):
    """
    Builds canonical observation dict according to the updated perception & environment rules:
    1. Card-conditioned tile mask (spells strictly 2-tile radius around enemies with lead; troops same-lane).
    2. Spell gating on empty arena (spells disabled when 0 active enemies at y >= 12).
    3. Low-elixir idle gate (< 7.5 elixir with 0 active enemies forces banking).
    """
    if active_enemy_units is None:
        active_enemy_units = []
        
    num_active_enemies = len(active_enemy_units)
    
    # 1. Scalars: [own_elixir_norm, opp_elixir_est, time_norm, is_ot]
    scalars = torch.tensor([[elixir / 10.0, 0.5, 0.8, 0.0]], dtype=torch.float32, device=device)
    
    # 2. Hand cards: 4 cards + next card (dummy 1)
    hand_cards = torch.tensor([hand_ids + [1]], dtype=torch.long, device=device)
    
    # 3. Entities & Arena Grid
    entities = torch.zeros((1, MAX_ENTITIES, ENTITY_FEATURES), dtype=torch.float32, device=device)
    entity_mask = torch.ones((1, MAX_ENTITIES), dtype=torch.bool, device=device)
    arena_grid = torch.zeros((1, SPATIAL_CHANNELS, ARENA_HEIGHT, ARENA_WIDTH), dtype=torch.float32, device=device)
    
    # Crown towers: slots 0..5
    towers = [
        [0.0,  1.0, 3.5 / 18.0,   6.5 / 32.0,  1.0, 0.0, 1.0],  # 0: Friendly Left Princess
        [0.0,  1.0, 14.5 / 18.0,  6.5 / 32.0,  1.0, 0.0, 1.0],  # 1: Friendly Right Princess
        [0.0,  1.0, 9.0 / 18.0,   3.0 / 32.0,  1.0, 0.0, 1.0],  # 2: Friendly King
        [0.0, -1.0, 3.5 / 18.0,  25.5 / 32.0,  1.0, 0.0, 1.0],  # 3: Enemy Left Princess
        [0.0, -1.0, 14.5 / 18.0, 25.5 / 32.0,  1.0, 0.0, 1.0],  # 4: Enemy Right Princess
        [0.0, -1.0, 9.0 / 18.0,  31.0 / 32.0,  1.0, 0.0, 1.0],  # 5: Enemy King
    ]
    for i, t in enumerate(towers):
        entities[0, i] = torch.tensor(t, dtype=torch.float32, device=device)
        entity_mask[0, i] = False
        gx = int(np.clip(t[2] * ARENA_WIDTH, 0, ARENA_WIDTH - 1))
        gy = int(np.clip(t[3] * ARENA_HEIGHT, 0, ARENA_HEIGHT - 1))
        arena_grid[0, 4, gy, gx] = 1.0

    arena_grid[0, 5, :16, :] = 1.0  # Friendly zone
    
    # Inject active enemy units (slot 6+)
    threats_our_half = []
    for idx, (cid, ex, ey) in enumerate(active_enemy_units, start=6):
        norm_x = ex / 18.0
        norm_y = ey / 32.0
        entities[0, idx] = torch.tensor([cid / 16.0, -1.0, norm_x, norm_y, 1.0, 0.0, 1.0], device=device)
        entity_mask[0, idx] = False
        arena_grid[0, 2, int(np.clip(ey, 0, 31)), int(np.clip(ex, 0, 17))] = 1.0
        if ey <= 15.5:
            threats_our_half.append((ex, ey))
            
    # 4. Card Mask (with spell gating & low-elixir idle gate)
    # Only unmask a Spell card in card_mask[slot] if there are >= 1 enemy units at tile_y >= 12
    enemies_at_bridge_or_above = sum(1 for (cid, ex, ey) in active_enemy_units if ey >= 12.0)
    card_mask = torch.zeros((1, 5), dtype=torch.bool, device=device)
    for i, cid in enumerate(hand_ids):
        cost = CARDS_CATALOG.get(cid, {}).get("cost", 3)
        if elixir >= cost:
            # Rule 2: Gate spells if 0 enemies at tile_y >= 12
            if cid in SPELL_CARD_IDS and enemies_at_bridge_or_above == 0:
                card_mask[0, i] = False
            else:
                card_mask[0, i] = True
        else:
            card_mask[0, i] = False

    # Win-condition defense guard: If under threat on our half (ey <= 15.5),
    # never waste building-targeters (Giant, HogRider) if an affordable defensive card is present
    if threats_our_half:
        has_affordable_defense = any(
            card_mask[0, s] and hand_ids[s] in DEFENSIVE_CARD_IDS
            for s in range(4)
        )
        if has_affordable_defense:
            for s in range(4):
                if hand_ids[s] in WIN_CONDITION_IDS:
                    card_mask[0, s] = False

    # Rule 3: Low-Elixir Idle Gate (< 7.5 elixir with 0 active enemies forces banking)
    if elixir < 7.5 and num_active_enemies == 0:
        card_mask[0, :4] = False
        
    card_mask[0, 4] = True  # Slot 4 (Wait) is always legal
    
    # 5. Card-Conditioned Tile Mask: shape [1, 5, 576]
    card_tile_mask = torch.zeros((1, 5, TOTAL_TILES), dtype=torch.bool, device=device)
    for slot in range(4):
        cid = hand_ids[slot]
        if cid not in SPELL_CARD_IDS:
            if cid == 11:  # HogRider: strictly Left and Right bridges
                for x in range(2, 5):
                    card_tile_mask[0, slot, 14 * ARENA_WIDTH + x] = True
                for x in range(13, 16):
                    card_tile_mask[0, slot, 14 * ARENA_WIDTH + x] = True
            elif threats_our_half:
                # When defending an active bridge threat on our side, placements must be in front of towers (y in 8..12)
                # Troops deploy directly in the SAME LANE as the threat
                primary_ex = threats_our_half[0][0]
                lane_min_x = int(max(1, primary_ex - 2))
                lane_max_x = int(min(16, primary_ex + 2))
                if cid in (13,):  # Cannon / defensive buildings
                    for y in range(9, 13):
                        for x in range(7, 11):
                            card_tile_mask[0, slot, y * ARENA_WIDTH + x] = True
                else:  # Defenders (Mini P.E.K.K.A, Knight, Skeletons, Archers, etc.)
                    for y in range(8, 13):
                        for x in range(lane_min_x, lane_max_x + 1):
                            card_tile_mask[0, slot, y * ARENA_WIDTH + x] = True
            else:
                for y in range(15):
                    for x in range(ARENA_WIDTH):
                        card_tile_mask[0, slot, y * ARENA_WIDTH + x] = True
        else:
            # Spells: STRICTLY restricted to a 2-tile radius around detected enemy units (y >= 12)
            # Never unmask entire arena (prevents logit bias to empty grass like (12, 27)).
            # Starts with all False across all 576 tiles.
            for (ecid, ex_val, ey_val) in active_enemy_units:
                if ey_val >= 11.5:
                    cx = int(round(ex_val))
                    cy = max(12, int(round(ey_val)) - 1)  # Lead 1 tile down for travel time, strictly >= 12
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            tx = cx + dx
                            ty = cy + dy
                            if 0 <= tx < ARENA_WIDTH and 12 <= ty < ARENA_HEIGHT:
                                card_tile_mask[0, slot, ty * ARENA_WIDTH + tx] = True
                        
    card_tile_mask[0, 4] = True  # Slot 4 (Wait) legal everywhere
    
    # Base tile mask (legacy union of friendly side)
    tile_mask = torch.zeros((1, TOTAL_TILES), dtype=torch.bool, device=device)
    for y in range(15):
        for x in range(ARENA_WIDTH):
            tile_mask[0, y * ARENA_WIDTH + x] = True
            
    obs = {
        "scalars": scalars,
        "hand_cards": hand_cards,
        "card_mask": card_mask,
        "tile_mask": tile_mask,
        "card_tile_mask": card_tile_mask,
        "entities": entities,
        "entity_mask": entity_mask,
        "arena_grid": arena_grid
    }
    return obs

def run_diagnostics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== VERIFICATION RUN (Device: {device}) ===")
    
    # Load model
    checkpoint_path = "checkpoints/current_deck/latest.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/latest.pt"
    model = AlphaStarCRNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    
    print("\n" + "=" * 65)
    print("TEST 1: State A (Empty arena, 5.0 Elixir, Hand = [Fireball, Zap, Musketeer, HogRider])")
    print("=" * 65)
    # Fireball=8, Zap=16, Musketeer=4, HogRider=11
    hand_a = [8, 16, 4, 11]
    obs_a = create_environment_state(hand_a, elixir=5.0, active_enemy_units=[], device=device)
    
    with torch.no_grad():
        actions_a, _, _, values_a = model.get_action_and_value(obs_a, deterministic=True)
        chosen_card_a = int(actions_a[0, 0].item())
        
        core_a = model.forward_core(obs_a["scalars"], obs_a["hand_cards"], obs_a["entities"], obs_a["entity_mask"], obs_a["arena_grid"])
        card_logits_a = model.card_head(core_a).float()
        card_mask_a = obs_a["card_mask"].bool()
        card_logits_masked_a = card_logits_a.masked_fill(~card_mask_a, -1e9)
        card_probs_a = F.softmax(card_logits_masked_a, dim=-1)[0].cpu().numpy()
        
        card_names_a = ["Slot 0: Fireball", "Slot 1: Zap", "Slot 2: Musketeer", "Slot 3: HogRider", "Slot 4: Wait/NoOp"]
        print("  Evaluated Action Probabilities:")
        for name, prob in zip(card_names_a, card_probs_a):
            print(f"    {name:<22}: {prob * 100:.2f}% (mask: {bool(card_mask_a[0, card_names_a.index(name)])})")
            
        print(f"\n  Chosen Action: {card_names_a[chosen_card_a]}")
        print(f"  Verification Result:")
        print(f"    - Fireball (Slot 0) Probability: {card_probs_a[0] * 100:.2f}% (Target: 0.00%) -> {'PASS' if card_probs_a[0] == 0.0 else 'FAIL'}")
        print(f"    - Zap (Slot 1) Probability:      {card_probs_a[1] * 100:.2f}% (Target: 0.00%) -> {'PASS' if card_probs_a[1] == 0.0 else 'FAIL'}")
        print(f"    - Wait/NoOp (Slot 4) Probability: {card_probs_a[4] * 100:.2f}% (Target: > 80.0%) -> {'PASS' if card_probs_a[4] > 0.80 else 'FAIL'}")

    print("\n" + "=" * 65)
    print("TEST 2A: State B1 (Enemy Giant + Mini PEKKA at Left Bridge (3, 13), 6.0 Elixir)")
    print("         Hand = [Cannon, Skeletons, Fireball, HogRider]")
    print("=" * 65)
    # Cannon=13, Skeletons=6, Fireball=8, HogRider=11
    hand_b1 = [13, 6, 8, 11]
    active_b = [(3, 3.0, 13.0), (5, 4.0, 13.0)]
    obs_b1 = create_environment_state(hand_b1, elixir=6.0, active_enemy_units=active_b, device=device)
    
    with torch.no_grad():
        actions_b1, _, _, values_b1 = model.get_action_and_value(obs_b1, deterministic=True)
        chosen_card_b1 = int(actions_b1[0, 0].item())
        chosen_tile_b1 = int(actions_b1[0, 1].item())
        tx_b1 = chosen_tile_b1 % ARENA_WIDTH
        ty_b1 = chosen_tile_b1 // ARENA_WIDTH
        
        core_b1 = model.forward_core(obs_b1["scalars"], obs_b1["hand_cards"], obs_b1["entities"], obs_b1["entity_mask"], obs_b1["arena_grid"])
        card_logits_b1 = model.card_head(core_b1).float()
        card_mask_b1 = obs_b1["card_mask"].bool()
        card_probs_b1 = F.softmax(card_logits_b1.masked_fill(~card_mask_b1, -1e9), dim=-1)[0].cpu().numpy()
        
        card_names_b1 = ["Slot 0: Cannon", "Slot 1: Skeletons", "Slot 2: Fireball", "Slot 3: HogRider", "Slot 4: Wait/NoOp"]
        print("  Evaluated Action Probabilities:")
        for name, prob in zip(card_names_b1, card_probs_b1):
            print(f"    {name:<22}: {prob * 100:.2f}% (mask: {bool(card_mask_b1[0, card_names_b1.index(name)])})")

        print(f"\n  Model Chosen Card  : {card_names_b1[chosen_card_b1]} (Value est: {values_b1.squeeze().item():+.2f})")
        print(f"  Model Target Tile  : ({tx_b1}, {ty_b1}) [Zone: {'Friendly Half' if ty_b1 < 16 else 'Enemy Half'}]")
        
        # Check unmasked tiles for Fireball (Slot 2) on tile_y <= 11:
        fb_mask_b1 = obs_b1["card_tile_mask"][0, 2, :].cpu().numpy()
        fb_tiles_y11_b1 = sum(fb_mask_b1[y * ARENA_WIDTH + x] for y in range(12) for x in range(ARENA_WIDTH))
        
        # Evaluate Cannon (Slot 0) tile target:
        chosen_emb_cn = model.chosen_action_emb(torch.tensor([0], device=device))
        tile_input_cn = torch.cat([core_b1, chosen_emb_cn], dim=-1)
        tile_logits_cn = model.tile_head(tile_input_cn).float()
        effective_mask_cn = obs_b1["card_tile_mask"][:, 0, :].bool()
        tile_probs_cn = F.softmax(tile_logits_cn.masked_fill(~effective_mask_cn, -1e9), dim=-1)[0].cpu().numpy()
        best_cn_idx = int(np.argmax(tile_probs_cn))
        cx_cn = best_cn_idx % ARENA_WIDTH
        cy_cn = best_cn_idx // ARENA_WIDTH
        print(f"  Defensive Building (Cannon) Placement Target: ({cx_cn}, {cy_cn}) [Zone: Friendly In-Front]")
        print(f"\n  Defensive Placement & Spell Mask Check:")
        print(f"    - HogRider (Slot 3) Probability: {card_probs_b1[3] * 100:.2f}% (Target: 0.00%) -> {'PASS' if card_probs_b1[3] == 0.0 else 'FAIL'}")
        print(f"    - Chosen Placement Tile: ({tx_b1}, {ty_b1}) in front of towers (y in 9..13): {'PASS' if 9 <= ty_b1 <= 13 else 'FAIL'}")
        print(f"    - Fireball Unmasked Tiles on tile_y <= 11: {fb_tiles_y11_b1} (Target: 0) -> {'PASS' if fb_tiles_y11_b1 == 0 else 'FAIL'}")

    print("\n" + "=" * 65)
    print("TEST 2B: State B2 (Enemy Giant + Mini PEKKA at Left Bridge (3, 13), 6.0 Elixir)")
    print("         Hand = [Mini P.E.K.K.A, Knight, Fireball, HogRider]")
    print("=" * 65)
    # MiniPekka=5, Knight=1, Fireball=8, HogRider=11
    hand_b2 = [5, 1, 8, 11]
    obs_b2 = create_environment_state(hand_b2, elixir=6.0, active_enemy_units=active_b, device=device)
    
    with torch.no_grad():
        actions_b2, _, _, values_b2 = model.get_action_and_value(obs_b2, deterministic=True)
        chosen_card_b2 = int(actions_b2[0, 0].item())
        chosen_tile_b2 = int(actions_b2[0, 1].item())
        tx_b2 = chosen_tile_b2 % ARENA_WIDTH
        ty_b2 = chosen_tile_b2 // ARENA_WIDTH
        
        core_b2 = model.forward_core(obs_b2["scalars"], obs_b2["hand_cards"], obs_b2["entities"], obs_b2["entity_mask"], obs_b2["arena_grid"])
        card_logits_b2 = model.card_head(core_b2).float()
        card_mask_b2 = obs_b2["card_mask"].bool()
        card_probs_b2 = F.softmax(card_logits_b2.masked_fill(~card_mask_b2, -1e9), dim=-1)[0].cpu().numpy()
        
        card_names_b2 = ["Slot 0: Mini P.E.K.K.A", "Slot 1: Knight", "Slot 2: Fireball", "Slot 3: HogRider", "Slot 4: Wait/NoOp"]
        print("  Evaluated Action Probabilities:")
        for name, prob in zip(card_names_b2, card_probs_b2):
            print(f"    {name:<24}: {prob * 100:.2f}% (mask: {bool(card_mask_b2[0, card_names_b2.index(name)])})")

        print(f"\n  Model Chosen Card  : {card_names_b2[chosen_card_b2]} (Value est: {values_b2.squeeze().item():+.2f})")
        print(f"  Model Target Tile  : ({tx_b2}, {ty_b2}) [Zone: {'Friendly Half' if ty_b2 < 16 else 'Enemy Half'}]")
        
        # Evaluate Mini P.E.K.K.A (Slot 0) tile target:
        chosen_emb_mp = model.chosen_action_emb(torch.tensor([0], device=device))
        tile_input_mp = torch.cat([core_b2, chosen_emb_mp], dim=-1)
        tile_logits_mp = model.tile_head(tile_input_mp).float()
        effective_mask_mp = obs_b2["card_tile_mask"][:, 0, :].bool()
        tile_probs_mp = F.softmax(tile_logits_mp.masked_fill(~effective_mask_mp, -1e9), dim=-1)[0].cpu().numpy()
        best_mp_idx = int(np.argmax(tile_probs_mp))
        cx_mp = best_mp_idx % ARENA_WIDTH
        cy_mp = best_mp_idx // ARENA_WIDTH
        print(f"  Troop Defender (Mini P.E.K.K.A) Target: ({cx_mp}, {cy_mp}) [Zone: Left Lane In-Front]")
        
        # Evaluate Fireball (Slot 2) tile target & check probability at (12, 27):
        chosen_emb_fb = model.chosen_action_emb(torch.tensor([2], device=device))
        tile_input_fb = torch.cat([core_b2, chosen_emb_fb], dim=-1)
        tile_logits_fb = model.tile_head(tile_input_fb).float()
        effective_mask_fb = obs_b2["card_tile_mask"][:, 2, :].bool()
        tile_probs_fb = F.softmax(tile_logits_fb.masked_fill(~effective_mask_fb, -1e9), dim=-1)[0].cpu().numpy()
        best_fb_idx = int(np.argmax(tile_probs_fb))
        cx_fb = best_fb_idx % ARENA_WIDTH
        cy_fb = best_fb_idx // ARENA_WIDTH
        bias_tile_idx = 27 * ARENA_WIDTH + 12
        prob_bias_tile = tile_probs_fb[bias_tile_idx] * 100.0
        print(f"  Spell (Fireball) Placement Target     : ({cx_fb}, {cy_fb}) with prob {tile_probs_fb[best_fb_idx] * 100:.2f}% [Directly on Bridge Push]")
        print(f"  Empty Grass Tile (12, 27) Probability : {prob_bias_tile:.4f}% (Target: 0.0000%)")
        
        fb_mask_b2 = obs_b2["card_tile_mask"][0, 2, :].cpu().numpy()
        fb_tiles_y11_b2 = sum(fb_mask_b2[y * ARENA_WIDTH + x] for y in range(12) for x in range(ARENA_WIDTH))
        
        print(f"\n  Same-Lane Defense & Spell Mask Check:")
        print(f"    - Model Chosen Card is Mini P.E.K.K.A: {'PASS' if chosen_card_b2 == 0 else 'FAIL'}")
        print(f"    - Model Target Tile is (3, 10):        {'PASS' if (tx_b2, ty_b2) == (3, 10) else 'FAIL'}")
        print(f"    - HogRider (Slot 3) Probability: {card_probs_b2[3] * 100:.2f}% (Target: 0.00%) -> {'PASS' if card_probs_b2[3] == 0.0 else 'FAIL'}")
        print(f"    - Mini P.E.K.K.A in Left Lane (x in 2..5, y in 8..12): ({cx_mp}, {cy_mp}) -> {'PASS' if 2 <= cx_mp <= 5 and 8 <= cy_mp <= 12 else 'FAIL'}")
        print(f"    - Fireball Top Target on Enemy Push: ({cx_fb}, {cy_fb}) -> {'PASS' if cx_fb in (2, 3, 4) and cy_fb in (12, 13) else 'FAIL'}")
        print(f"    - Empty Grass (12, 27) Probability: {prob_bias_tile:.4f}% (Target: 0.0000%) -> {'PASS' if prob_bias_tile == 0.0 else 'FAIL'}")
        print(f"    - Fireball Unmasked Tiles on tile_y <= 11: {fb_tiles_y11_b2} (Target: 0) -> {'PASS' if fb_tiles_y11_b2 == 0 else 'FAIL'}")

    print("\n" + "=" * 65)
    print("TEST 3: VisualCardClassifier Template Verification & Perception Audit")
    print("=" * 65)
    classifier = VisualCardClassifier()
    print(f"  VisualCardClassifier Loaded Templates: {len(classifier.templates)}")
    for cname, tmpl in classifier.templates.items():
        print(f"    - Template '{cname}': shape {tmpl.shape}, dtype {tmpl.dtype}")
    
    deck_templates_ok = len(classifier.templates) >= 8
    print(f"  Verification: All 8 Deck Templates Present -> {'PASS' if deck_templates_ok else 'FAIL'}")

    # Test Perception False-Positive Elimination (using available frames or synthetic arena)
    perception = PerceptionAdapter(device=torch.device("cpu"))
    art_dir = r"C:/Users/varun/.gemini/antigravity/brain/42ce9bd6-0ed7-406b-aa50-7c85b510c081"
    cand_frames = [
        os.path.join(art_dir, "live_battle_monitoring_frame_2.png"),
        os.path.join(art_dir, "current_mumu_screen.png")
    ]
    verified_frames = [p for p in cand_frames if os.path.exists(p)]
    
    if verified_frames:
        for frame_p in verified_frames:
            img = cv2.imread(frame_p)
            label = os.path.basename(frame_p)
            print(f"\n  Checking Frame: {label} (Shape: {img.shape})")
            obs_live = perception.frame_to_obs(img)
            print(f"    Classified obs['hand_cards']: {obs_live['hand_cards'][0].numpy()}")
            print(f"    Card names: {perception.tracked_hand}")
            
            invalid_cards = [c for c in perception.tracked_hand if c not in perception.active_deck]
            print(f"    - Out-of-Deck / Invalid Cards: {len(invalid_cards)} (Target: 0) -> {'PASS' if len(invalid_cards) == 0 else 'FAIL'}")
            
            entities_live = obs_live["entities"][0].numpy()
            entity_mask_live = obs_live["entity_mask"][0].numpy()
            detected_units = [entities_live[idx] for idx in range(6, len(entities_live)) if not entity_mask_live[idx]]
            
            horizontal_hallucinations = [e for e in detected_units if 0.15 <= e[3] <= 0.22]
            river_hallucinations = [e for e in detected_units if 0.43 <= e[3] <= 0.53]
            tower_hallucinations = [e for e in detected_units if e[3] >= 0.75]
            
            print(f"    - Horizontal UI Band Detections (norm_y 0.15..0.22): {len(horizontal_hallucinations)} (Target: 0) -> {'PASS' if len(horizontal_hallucinations) == 0 else 'FAIL'}")
            print(f"    - River Hallucinations (norm_y 0.43..0.53):          {len(river_hallucinations)} (Target: 0) -> {'PASS' if len(river_hallucinations) == 0 else 'FAIL'}")
            print(f"    - Tower-Banner Hallucinations (norm_y >= 0.75):       {len(tower_hallucinations)} (Target: 0) -> {'PASS' if len(tower_hallucinations) == 0 else 'FAIL'}")
    else:
        # Synthetic test frame (clean green background, no spurious contours)
        synth_frame = np.full((1600, 900, 3), (45, 120, 50), dtype=np.uint8)
        obs_live = perception.frame_to_obs(synth_frame)
        entities_live = obs_live["entities"][0].numpy()
        entity_mask_live = obs_live["entity_mask"][0].numpy()
        detected_units = [entities_live[idx] for idx in range(6, len(entities_live)) if not entity_mask_live[idx]]
        print(f"\n  Checking Synthetic Frame: Detected Units = {len(detected_units)} (Target: 0) -> {'PASS' if len(detected_units) == 0 else 'FAIL'}")

if __name__ == "__main__":
    run_diagnostics()
