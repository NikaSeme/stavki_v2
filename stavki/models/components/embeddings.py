"""
Player & Context Encoders for Deep Interaction Network V2
==========================================================
Components:
- Self-Attention player aggregation (position-aware, not just sum)
- Bidirectional Cross-Team Interaction (how teams match up, both directions)
- League Embedding (learnable league characteristics)
- Venue Embedding (home ground advantage patterns)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlayerEncoder(nn.Module):
    """Learnable Player Embeddings with Self-Attention aggregation."""

    def __init__(self, num_players: int, embedding_dim: int = 32, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_players, embedding_dim, padding_idx=0)
        self.pos_dim = 4
        self.position_enc = nn.Embedding(50, self.pos_dim, padding_idx=0)
        
        attn_dim = embedding_dim + self.pos_dim

        self.attention = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(attn_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.position_enc.weight, std=0.01)

    def forward(self, player_indices: torch.LongTensor, position_indices: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            player_indices: (B, 11) LongTensor
            position_indices: (B, 11) LongTensor of real SportMonks positions

        Returns:
            team_vector: (B, embed_dim + pos_dim)
        """
        # (B, 11, D)
        vectors = self.embedding(player_indices)

        # Add positional encoding (pitch position awareness)
        pos_vectors = self.position_enc(position_indices)
        
        # Concat ID vector and Position vector
        vectors = torch.cat([vectors, pos_vectors], dim=-1)

        # Create attention mask for padded players (index 0)
        key_padding_mask = (player_indices == 0)  # (B, 11) True = ignore

        # Guard: if ALL players are padding (no lineup data), skip attention
        all_padding = key_padding_mask.all(dim=1)  # (B,)
        if all_padding.any():
            safe_mask = key_padding_mask.clone()
            safe_mask[all_padding] = False
        else:
            safe_mask = key_padding_mask

        # Self-attention: players attend to each other
        attended, _ = self.attention(vectors, vectors, vectors, key_padding_mask=safe_mask)

        # Residual connection
        vectors = vectors + attended

        # Aggregate: Attention-weighted mean (exclude padding)
        mask = (~key_padding_mask).unsqueeze(-1).float()  # (B, 11, 1)
        team_vector = (vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        team_vector = self.norm(team_vector)
        team_vector = self.dropout(team_vector)

        return team_vector


class ContextEncoder(nn.Module):
    """MLP encoder for tabular context features."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeagueEncoder(nn.Module):
    """
    Learnable league embeddings.

    Different leagues have fundamentally different characteristics:
    - EPL: Physical, fast transitions
    - La Liga: Possession-based, tactical
    - Serie A: Defensive, set-piece dominant
    - Bundesliga: High-pressing, open play

    The embedding lets the model learn these patterns.
    """

    def __init__(self, num_leagues: int, embed_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_leagues, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, league_id: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            league_id: (B,) LongTensor of dense league indices

        Returns:
            league_vector: (B, embed_dim)
        """
        vec = self.embedding(league_id)
        vec = self.norm(vec)
        return self.dropout(vec)


class VenueEncoder(nn.Module):
    """Learnable embeddings for home advantage based on venue/team."""
    def __init__(self, num_teams: int, embed_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_teams + 1, embed_dim, padding_idx=0)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, venue_id: torch.LongTensor) -> torch.Tensor:
        vec = self.embedding(venue_id)
        vec = self.norm(vec)
        return self.dropout(vec)
        

class RefereeEncoder(nn.Module):
    """Learnable embeddings for match referee tendencies."""
    def __init__(self, num_referees: int, embed_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_referees + 1, embed_dim, padding_idx=0)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, referee_id: torch.LongTensor) -> torch.Tensor:
        vec = self.embedding(referee_id)
        vec = self.norm(vec)
        return self.dropout(vec)

class ManagerEncoder(nn.Module):
    """Learnable embeddings for tactical managers/coaches."""
    def __init__(self, num_managers: int, embed_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_managers + 1, embed_dim, padding_idx=0)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, manager_id: torch.LongTensor) -> torch.Tensor:
        vec = self.embedding(manager_id)
        vec = self.norm(vec)
        return self.dropout(vec)


class CrossTeamInteraction(nn.Module):
    """
    Bidirectional Cross-Attention between Home and Away player sets.

    Two directions of matchup intelligence:
    - Home→Away: "How do our attackers exploit their defense?"
    - Away→Home: "How do their attackers threaten our defense?"

    The two matchup vectors are combined via a learned gate that
    determines how much each direction contributes.
    """

    def __init__(self, embed_dim: int = 32, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Home attends to Away (offensive analysis)
        self.cross_attn_h2a = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Away attends to Home (defensive analysis)
        self.cross_attn_a2h = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learned fusion gate: combines both directions
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def _safe_mask(self, mask: torch.BoolTensor) -> torch.BoolTensor:
        """Disable padding mask if ALL positions are padding (prevents NaN)."""
        all_pad = mask.all(dim=1)
        if all_pad.any():
            safe = mask.clone()
            safe[all_pad] = False
            return safe
        return mask

    def _aggregate(self, cross_out: torch.Tensor, query_mask: torch.BoolTensor) -> torch.Tensor:
        """Masked mean aggregation with all-padding guard."""
        safe_mask = self._safe_mask(query_mask)
        mask_float = (~safe_mask).unsqueeze(-1).float()  # (B, 11, 1)
        return (cross_out * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)

    def forward(self, home_players: torch.LongTensor, away_players: torch.LongTensor,
                h_positions: torch.LongTensor, a_positions: torch.LongTensor,
                embedding: nn.Embedding, position_enc: nn.Embedding) -> torch.Tensor:
        """
        Returns a bidirectional matchup vector capturing how teams interact globally.

        Args:
            home_players: (B, 11) player indices
            away_players: (B, 11) player indices
            h_positions: (B, 11) position indices
            a_positions: (B, 11) position indices
            embedding: shared embedding layer
            position_enc: shared position encoding embedding

        Returns:
            matchup_vector: (B, embed_dim) — fused bidirectional matchup
        """
        h_vecs = torch.cat([embedding(home_players), position_enc(h_positions)], dim=-1)
        a_vecs = torch.cat([embedding(away_players), position_enc(a_positions)], dim=-1)

        h_mask = (home_players == 0)
        a_mask = (away_players == 0)
        safe_a_mask = self._safe_mask(a_mask)
        safe_h_mask = self._safe_mask(h_mask)

        # Direction 1: Home→Away ("How do our players match against theirs?")
        h2a_out, _ = self.cross_attn_h2a(h_vecs, a_vecs, a_vecs, key_padding_mask=safe_a_mask)
        h2a_vec = self._aggregate(h2a_out, h_mask)  # (B, D)

        # Direction 2: Away→Home ("How do their players threaten ours?")
        a2h_out, _ = self.cross_attn_a2h(a_vecs, h_vecs, h_vecs, key_padding_mask=safe_h_mask)
        a2h_vec = self._aggregate(a2h_out, a_mask)  # (B, D)

        # Gated fusion: learn how much each direction matters
        concat = torch.cat([h2a_vec, a2h_vec], dim=1)  # (B, 2D)
        gate_weight = self.gate(concat)  # (B, D) — values in [0, 1]
        matchup = gate_weight * h2a_vec + (1 - gate_weight) * a2h_vec

        return self.norm(matchup)


class TimeDecayLayer(nn.Module):
    """
    Learnable exponential time decay on momentum features.

    Instead of treating all momentum windows equally, this learns an
    optimal decay rate *per feature dimension*, so the model can decide:
    - "Goals scored last match matters a lot" (high decay = recent-heavy)
    - "Season-long xG is stable" (low decay = long-memory)

    Applied BEFORE the momentum encoder:
    momentum_out = MomentumEncoder(TimeDecayLayer(raw_momentum))
    """

    def __init__(self, input_dim: int):
        super().__init__()
        # Learnable decay rates: one per momentum feature
        # Initialized near 0 → sigmoid(0) = 0.5 → moderate decay
        self.decay_logits = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, momentum_dim) raw momentum features

        Returns:
            (B, momentum_dim) decay-weighted momentum
        """
        # sigmoid → [0, 1] range for each feature's recency weight
        weights = torch.sigmoid(self.decay_logits)  # (D,)
        return x * weights.unsqueeze(0)  # (B, D) * (1, D)


class SeasonPhaseEncoder(nn.Module):
    """
    Learnable season phase embeddings.

    European football has distinct phases:
    - Early (Aug-Oct): Teams finding form, new signings settling
    - Mid (Nov-Feb): Winter period, congested fixtures, form stabilizes
    - Late (Mar-May): Relegation battles, title races, high stakes change behavior
    - Summer (Jun-Jul): Cups, international, reduced squads
    """

    def __init__(self, num_phases: int = 4, embed_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_phases, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, phase_id: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            phase_id: (B,) LongTensor — 0=early, 1=mid, 2=late, 3=summer

        Returns:
            phase_vector: (B, embed_dim)
        """
        vec = self.embedding(phase_id)
        vec = self.norm(vec)
        return self.dropout(vec)


class H2HEncoder(nn.Module):
    """
    Head-to-Head history encoder.

    Processes computed H2H features (NOT learned embeddings — the pair
    space is too sparse for that with only 6k matches).

    Input: 4-dim vector per match:
    - home_h2h_winrate: Home team's historical win rate vs this opponent
    - h2h_draw_rate: Historical draw rate between these teams
    - h2h_avg_goals: Average goals in past meetings (normalized)
    - h2h_familiarity: How many times they've met (log-scaled)
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, h2h_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h2h_features: (B, 4) computed H2H feature vector

        Returns:
            h2h_vector: (B, output_dim)
        """
        return self.net(h2h_features)

