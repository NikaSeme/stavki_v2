"""
Deep Interaction Network - Training Script (V2 Hardened)
========================================================
Fixes applied:
- Temporal split (no data leakage)
- Early stopping with patience
- Gradient clipping for embedding stability
- Shape squeeze for Poisson loss
- Home advantage flag in context
- League embedding (learnable league characteristics)
"""
import sys
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from stavki.config import PROJECT_ROOT
from stavki.data.datasets import StavkiDataset
from stavki.models.deep_interaction import DeepInteractionNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def train():
    # ── Config ──────────────────────────────────────────────
    BATCH_SIZE = 64
    LR = 2e-4
    EPOCHS = 100
    PATIENCE = 15
    GRAD_CLIP = 1.0
    DROPOUT = 0.35  # Higher for 18k+ dataset
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    SAVE_PATH = PROJECT_ROOT / "models" / "deep_interaction_v3.pth"
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Device: {DEVICE}")

    # ── 1. Load Data ────────────────────────────────────────
    full_ds = StavkiDataset()
    n = len(full_ds)
    logger.info(f"Total samples: {n}")

    if n < 20:
        logger.warning("Dataset too small for meaningful training.")

    # ── 2. Temporal Split (Bob Standard) ────────────────────
    dates = full_ds.dates
    sorted_indices = np.argsort(dates)  # Oldest → Newest

    split_point = int(0.8 * n)
    train_indices = sorted_indices[:split_point].tolist()
    val_indices = sorted_indices[split_point:].tolist()

    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} (Temporal 80/20)")

    # ── 3. Init Model ──────────────────────────────────────
    model = DeepInteractionNetwork(
        num_players=full_ds.num_players,
        num_teams=full_ds.num_teams,
        num_leagues=full_ds.num_leagues,
        num_referees=full_ds.num_referees,
        num_managers=full_ds.num_managers,
        num_season_phases=4,
        embed_dim=32,
        league_dim=8,
        venue_dim=8,
        referee_dim=8,
        season_dim=4,
        h2h_dim=8,
        h2h_input_dim=4,
        context_dim=13,
        momentum_dim=13,
        hidden_dim=128,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Losses
    criterion_outcome = nn.CrossEntropyLoss(label_smoothing=0.10)
    criterion_goals = nn.PoissonNLLLoss(log_input=False)

    # ── 4. Training Loop ───────────────────────────────────
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            h_p = batch['home_players'].to(DEVICE)
            a_p = batch['away_players'].to(DEVICE)
            h_pos = batch['home_positions'].to(DEVICE)
            a_pos = batch['away_positions'].to(DEVICE)
            h_man = batch['home_manager'].to(DEVICE)
            a_man = batch['away_manager'].to(DEVICE)
            h_c = batch['home_context'].to(DEVICE)
            a_c = batch['away_context'].to(DEVICE)
            h_m = batch['home_momentum'].to(DEVICE)
            a_m = batch['away_momentum'].to(DEVICE)
            league = batch['league_id'].to(DEVICE)
            venue = batch['venue_id'].to(DEVICE)
            season = batch['season_phase'].to(DEVICE)
            referee = batch['referee_id'].to(DEVICE)
            h2h = batch['h2h_features'].to(DEVICE)

            target_outcome = batch['outcome'].to(DEVICE)
            target_h_goals = batch['home_goals'].to(DEVICE)
            target_a_goals = batch['away_goals'].to(DEVICE)

            optimizer.zero_grad()
            logits, lam_h, lam_a = model(
                h_p, a_p, h_pos, a_pos, h_man, a_man, h_c, a_c, h_m, a_m, league, venue, referee, season, h2h
            )

            # Squeeze goal lambdas: (B, 1) → (B,)
            lam_h = lam_h.squeeze(-1)
            lam_a = lam_a.squeeze(-1)

            loss_out = criterion_outcome(logits, target_outcome)
            loss_hg = criterion_goals(lam_h, target_h_goals)
            loss_ag = criterion_goals(lam_a, target_a_goals)

            loss = loss_out + 0.5 * (loss_hg + loss_ag)

            loss.backward()

            # Gradient clipping (stabilize rare player embeddings)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

            optimizer.step()
            scheduler.step(epoch + len(pbar) / len(train_loader))

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == target_outcome).sum().item()
            train_total += target_outcome.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0

        # ── Validation ─────────────────────────────────────
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                h_p = batch['home_players'].to(DEVICE)
                a_p = batch['away_players'].to(DEVICE)
                h_pos = batch['home_positions'].to(DEVICE)
                a_pos = batch['away_positions'].to(DEVICE)
                h_man = batch['home_manager'].to(DEVICE)
                a_man = batch['away_manager'].to(DEVICE)
                h_c = batch['home_context'].to(DEVICE)
                a_c = batch['away_context'].to(DEVICE)
                h_m = batch['home_momentum'].to(DEVICE)
                a_m = batch['away_momentum'].to(DEVICE)
                league = batch['league_id'].to(DEVICE)
                venue = batch['venue_id'].to(DEVICE)
                referee = batch['referee_id'].to(DEVICE)
                season = batch['season_phase'].to(DEVICE)
                h2h = batch['h2h_features'].to(DEVICE)

                target_outcome = batch['outcome'].to(DEVICE)
                target_h_goals = batch['home_goals'].to(DEVICE)
                target_a_goals = batch['away_goals'].to(DEVICE)

                logits, lam_h, lam_a = model(
                    h_p, a_p, h_pos, a_pos, h_man, a_man, h_c, a_c, h_m, a_m, league, venue, referee, season, h2h
                )
                lam_h = lam_h.squeeze(-1)
                lam_a = lam_a.squeeze(-1)

                loss_out = criterion_outcome(logits, target_outcome)
                loss_hg = criterion_goals(lam_h, target_h_goals)
                loss_ag = criterion_goals(lam_a, target_a_goals)

                loss = loss_out + 0.5 * (loss_hg + loss_ag)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == target_outcome).sum().item()
                total += target_outcome.size(0)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = correct / total if total > 0 else 0

        logger.info(
            f"Epoch {epoch+1}: "
            f"Train Loss={avg_train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={avg_val_loss:.4f} Acc={val_acc:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        # ── Early Stopping ─────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Save with metadata for wrapper reconstruction
            torch.save({
                'state_dict': best_state,
                'num_players': full_ds.num_players,
                'num_teams': full_ds.num_teams,
                'num_leagues': full_ds.num_leagues,
                'num_referees': full_ds.num_referees,
                'num_managers': full_ds.num_managers,
                'league_map': full_ds.league_map,
                'team_map': getattr(full_ds, 'team_map', {}),
                'referee_map': getattr(full_ds, 'ref_map', {}),
                'manager_map': getattr(full_ds, 'man_map', {}),
                'epoch': epoch,
                'val_loss': best_val_loss
            }, SAVE_PATH)
            logger.info(f"  ✅ New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"  ⏹ Early stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best model (Val Loss: {best_val_loss:.4f})")

    logger.info(f"Training complete. Best checkpoint at: {SAVE_PATH}")


if __name__ == "__main__":
    train()
