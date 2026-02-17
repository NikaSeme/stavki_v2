import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List

from stavki.features.builders.form import GoalsBuilder, FormCalculator
from stavki.features.builders.h2h import H2HBuilder as H2HFeatureBuilder
from stavki.features.builders.advanced_stats import AdvancedFeatureBuilder
from stavki.features.builders.corners import CornersFeatureBuilder
from stavki.features.builders.seasonal import SeasonalFeatureBuilder
from stavki.models.poisson.dixon_coles import DixonColesModel
from stavki.data.schemas import Match, Team, MatchStats

@pytest.fixture
def sample_matches():
    """Create a sequence of matches for testing."""
    matches = []
    base_date = datetime(2023, 8, 12, 15, 0)
    
    teams = ["Arsenal", "Chelsea", "Liverpool", "Man City"]
    
    for i in range(10):
        # Default scores: Home Win (2-1) or Away Win (1-2)
        h_score = 2 if i % 2 == 0 else 1
        a_score = 1 if i % 2 == 0 else 2
        
        # Make match 2 (Liverpool vs Man City) a Draw
        if i == 2:
            h_score = 1
            a_score = 1
            
        m = Match(
            id=f"m_{i}",
            league="soccer_epl",
            home_team=Team(name=teams[i % 4], id=i % 4),
            away_team=Team(name=teams[(i + 1) % 4], id=(i + 1) % 4),
            commence_time=base_date + timedelta(days=i*7),
            home_score=h_score,
            away_score=a_score,
            stats=MatchStats(
                match_id=f"m_{i}",
                possession_home=50, possession_away=50,
                shots_home=10, shots_away=8,
                sot_home=4, sot_away=3,
                corners_home=5, corners_away=3,
                # Official xG missing to test fallback
                xg_home=None, xg_away=None
            ),
            is_completed=True
        )
        matches.append(m)
        
    return matches

def test_goals_builder_temporal_leak(sample_matches):
    """Verify GoalsBuilder respects as_of date."""
    builder = GoalsBuilder(window=5)
    builder.fit(sample_matches)
    
    # Predict for the LAST match, as of just before it starts
    target_match = sample_matches[-1]
    as_of = target_match.commence_time
    
    # 1. Compute with strict as_of
    features_strict = builder.get_features(target_match, as_of=as_of)
    league_avg_strict = features_strict.get("league_avg_goals", 0)
    
    # 2. Compute correct expectation
    # Previous 9 matches:
    # Match 0,1,3,4,5,6,7,8 (win/loss): 3 goals each. (8 matches) -> 24 goals
    # Match 2 (draw): 1-1 -> 2 goals.
    # Total goals = 24 + 2 = 26.
    # Matches = 9.
    # Avg = 26 / 9 = 2.888...
    expected_avg = 26.0 / 9.0
    
    # Check dynamic average computation
    assert abs(league_avg_strict - expected_avg) < 0.1, \
        f"League avg {league_avg_strict} should match expected {expected_avg}"

def test_h2h_continuous(sample_matches):
    """Verify H2H home advantage is continuous [-1, 1]."""
    builder = H2HFeatureBuilder(max_meetings=10, recent_window=5)
    
    # m0: Arsenal vs Chelsea (2-1) -> Home Win
    # m4: Arsenal vs Chelsea (2-1) -> Home Win
    target_match = sample_matches[4] 
    
    features = builder.get_features(
        target_match.home_team.normalized_name,
        target_match.away_team.normalized_name,
        sample_matches,
        as_of=target_match.commence_time
    )
    h2h_adv = features.get("h2h_home_advantage")
    
    assert isinstance(h2h_adv, float)
    assert -1.0 <= h2h_adv <= 1.0
    # History: m0 (Home Win). Adv = (1-0)/1 = 1.0.
    assert h2h_adv > 0.0, "Arsenal should have positive H2H advantage"

def test_h2h_draws(sample_matches):
    """Verify H2H handles draws correctly."""
    builder = H2HFeatureBuilder(max_meetings=10)
    
    # Match 2: Liverpool (2) vs Man City (3) -> 1-1 (Draw)
    # Target: Match 6: Liverpool (2) vs Man City (3)
    target_match = sample_matches[6]
    
    features = builder.get_features(
        target_match.home_team.normalized_name,
        target_match.away_team.normalized_name,
        sample_matches,
        as_of=target_match.commence_time
    )
    
    # Verify draw metrics
    assert features["h2h_draws"] == 1
    assert features["h2h_draw_rate"] == 1.0
    # Verify advantage logic handles draw neutrally
    # (HomeWins - AwayWins) / Total = (0 - 0) / 1 = 0.0
    assert features["h2h_home_advantage"] == 0.0

def test_dead_xg_fallback(sample_matches):
    """Verify dead xG features use synthetic fallback."""
    builder = AdvancedFeatureBuilder(window=5)
    builder.fit(sample_matches)
    
    target_match = sample_matches[5]
    
    features = builder.get_features(target_match, as_of=target_match.commence_time)
    
    # Updated keys to match what get_features returns
    assert features["home_xg_for"] > 0.01, "xG For should be > 0 (synthetic)"
    assert features["home_xg_against"] > 0.01, "xG Against should be > 0"

def test_corners_features(sample_matches):
    """Verify corners features are computed."""
    builder = CornersFeatureBuilder(rolling_window=5)
    builder.fit(sample_matches)
    
    target_match = sample_matches[5]
    features = builder.get_features(target_match, as_of=target_match.commence_time)
    
    # Corners keys: rolling_corners_for_home
    assert "rolling_corners_for_home" in features
    assert features["rolling_corners_for_home"] > 0
    assert "corners_diff" in features

def test_seasonal_features(sample_matches):
    """Verify seasonal features."""
    builder = SeasonalFeatureBuilder()
    
    target_match = sample_matches[0] # Aug 12
    features = builder.get_features(target_match)
    
    assert "season_progress" in features
    assert 0.0 <= features["season_progress"] <= 1.0
    
    last_match = sample_matches[-1]
    features_last = builder.get_features(last_match)
    assert features_last["season_progress"] > features["season_progress"]

def test_dixon_coles_rho_fit(sample_matches):
    """Verify Dixon-Coles fits rho parameter."""
    data = []
    for m in sample_matches:
        data.append({
            "HomeTeam": m.home_team.name,
            "AwayTeam": m.away_team.name,
            "FTHG": m.home_score,
            "FTAG": m.away_score,
            "Date": m.commence_time
        })
    df = pd.DataFrame(data)
    
    model = DixonColesModel()
    model.fit(df)
    
    assert hasattr(model, "rho")
    assert isinstance(model.rho, float)
    print(f"Fitted rho: {model.rho}")
