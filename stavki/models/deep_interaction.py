"""
Deep Interaction Network V2
============================
Architecture:
- Self-Attention Player Encoder (position-aware)
- Bidirectional Cross-Team Interaction (gated matchup learning)
- League Embedding (learnable league characteristics)
- Venue Embedding (home ground advantage patterns)
- Head-to-Head History Encoder (computed features, not pair embedding)
- Time Decay on Momentum (learnable per-dimension decay)
- Season Phase Embedding (early/mid/late/summer)
- Context & Momentum MLPs
- Multi-Task Heads (Outcome + Goals)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .components.embeddings import (
    PlayerEncoder, ContextEncoder, CrossTeamInteraction,
    LeagueEncoder, VenueEncoder, RefereeEncoder,
    TimeDecayLayer, SeasonPhaseEncoder, H2HEncoder, ManagerEncoder
)


class DeepInteractionNetwork(nn.Module):
    def __init__(
        self,
        num_players: int,
        num_teams: int = 500,
        num_leagues: int = 10,
        num_referees: int = 500,
        num_managers: int = 500,
        num_season_phases: int = 4,
        embed_dim: int = 32,
        manager_dim: int = 16,
        league_dim: int = 8,
        venue_dim: int = 8,
        referee_dim: int = 8,
        season_dim: int = 4,
        h2h_dim: int = 8,
        h2h_input_dim: int = 4,
        context_dim: int = 17,
        momentum_dim: int = 13,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        """
        Deep Interaction Network V2 — The Bob Standard.

        Args:
            num_players: Player vocabulary size.
            num_teams: Number of teams (for venue embedding).
            num_leagues: Number of leagues.
            num_season_phases: Number of season phases (4).
            embed_dim: Dimension of player/team vectors.
            league_dim: Dimension of league embedding.
            venue_dim: Dimension of venue embedding.
            season_dim: Dimension of season phase embedding.
            h2h_dim: Dimension of H2H encoder output.
            h2h_input_dim: Dimension of H2H input features (4).
            context_dim: Team context features (XI stats + history + home flag).
            momentum_dim: Momentum features.
            hidden_dim: Internal hidden size.
            num_heads: Attention heads.
            dropout: Dropout rate.
        """
        super().__init__()

        # 1. Player Encoder (shared for Home & Away)
        self.player_encoder = PlayerEncoder(num_players, embed_dim, num_heads, dropout)
        cross_dim = embed_dim + self.player_encoder.pos_dim  # 36

        # 2. Bidirectional Cross-Team Interaction
        self.cross_interaction = CrossTeamInteraction(cross_dim, num_heads, dropout)

        # 3. League Encoder
        self.league_encoder = LeagueEncoder(num_leagues, league_dim, dropout)

        # 4. Venue Encoder (home_team_id → venue)
        self.venue_encoder = VenueEncoder(num_teams, venue_dim, dropout)

        # 4b. Referee Encoder
        self.referee_encoder = RefereeEncoder(num_referees, referee_dim, dropout)
        
        # 4c. Manager Encoder
        self.manager_encoder = ManagerEncoder(num_managers, manager_dim, dropout)

        # 5. Season Phase Encoder
        self.season_encoder = SeasonPhaseEncoder(num_season_phases, season_dim, dropout)

        # 6. H2H Encoder
        self.h2h_encoder = H2HEncoder(h2h_input_dim, hidden_dim=16, output_dim=h2h_dim, dropout=dropout)

        # 7. Time Decay on Momentum
        self.time_decay = TimeDecayLayer(momentum_dim)

        # 8. Context & Momentum Encoders
        self.context_encoder = ContextEncoder(context_dim, embed_dim, dropout)
        self.momentum_encoder = ContextEncoder(momentum_dim, embed_dim, dropout)

        # 9. Fusion
        # [HomeTeam(36), AwayTeam(36), Matchup(36), HomeCtx(32), AwayCtx(32), HomeMom(32), AwayMom(32)]
        # + League + Venue + Referee + Season + H2H + 2x Manager
        fusion_dim = (cross_dim * 3) + (embed_dim * 4) + \
                     league_dim + venue_dim + referee_dim + season_dim + h2h_dim + (manager_dim * 2)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
        )

        final_dim = hidden_dim // 2

        # 10. Prediction Heads
        self.head_outcome = nn.Linear(final_dim, 3)

        self.head_home_goals = nn.Sequential(
            nn.Linear(final_dim, 1),
            nn.Softplus(),
        )

        self.head_away_goals = nn.Sequential(
            nn.Linear(final_dim, 1),
            nn.Softplus(),
        )

    def enable_mc_dropout(self):
        """
        Force all dropout layers back into train mode for Bayesian Stochastic Sampling
        during inference (Monte Carlo Dropout).
        """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
    def forward(self, h_players, a_players, h_positions, a_positions, 
                h_manager, a_manager,
                h_ctx, a_ctx, h_mom, a_mom,
                league_id, venue_id, referee_id, season_phase, h2h_features):
        # 1. Encode each team independently (self-attention)
        h_vec = self.player_encoder(h_players, h_positions)  # (B, 36)
        a_vec = self.player_encoder(a_players, a_positions)

        # 2. Bidirectional cross-team matchup
        matchup = self.cross_interaction(
            h_players, a_players,
            h_positions, a_positions,
            self.player_encoder.embedding,
            self.player_encoder.position_enc,
        )

        # 3. League embedding
        league_vec = self.league_encoder(league_id)

        # 4. Venue embedding
        venue_vec = self.venue_encoder(venue_id)

        # 4b. Referee embedding
        ref_vec = self.referee_encoder(referee_id)

        # 5. Season phase
        season_vec = self.season_encoder(season_phase)

        # 6. H2H history
        h2h_vec = self.h2h_encoder(h2h_features)

        # 7. Context
        h_ctx_vec = self.context_encoder(h_ctx)
        a_ctx_vec = self.context_encoder(a_ctx)

        # 8. Momentum (with learnable time decay)
        h_mom_decayed = self.time_decay(h_mom)
        a_mom_decayed = self.time_decay(a_mom)
        h_mom_vec = self.momentum_encoder(h_mom_decayed)
        a_mom_vec = self.momentum_encoder(a_mom_decayed)

        # 8b. Managers
        h_man_vec = self.manager_encoder(h_manager)
        a_man_vec = self.manager_encoder(a_manager)

        # 9. Fusion
        x = torch.cat([
            h_vec, a_vec, matchup,
            h_ctx_vec, a_ctx_vec,
            h_mom_vec, a_mom_vec,
            league_vec, venue_vec, ref_vec, season_vec, h2h_vec,
            h_man_vec, a_man_vec
        ], dim=1)
        features = self.fusion(x)

        # 10. Predictions
        logits = self.head_outcome(features)
        lambda_home = self.head_home_goals(features) + 1e-6
        lambda_away = self.head_away_goals(features) + 1e-6

        return logits, lambda_home, lambda_away
