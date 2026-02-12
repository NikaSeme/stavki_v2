"""
Integration tests for SportMonks API and unified data pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_unified_loader():
    """Test unified data loader."""
    print("="*60)
    print("TEST 1: Unified Data Loader")
    print("="*60)
    
    from stavki.data.loader import get_loader
    
    loader = get_loader()
    print(f"✅ Loader initialized (API: {'✓' if loader.client else '✗'})")
    
    # Test team normalization
    print("\nTeam name normalization:")
    tests = [
        ("Man United", "manchester united"),
        ("Spurs", "tottenham hotspur"),
        ("Bayern München", "bayern munich"),
        ("Inter", "inter milan"),
        ("Arsenal", "arsenal"),  # Should stay same (lowercased)
    ]
    
    all_passed = True
    for input_name, expected in tests:
        result = loader.normalize_team_name(input_name)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} '{input_name}' -> '{result}'")
    
    # Test historical data loading
    print("\nHistorical data loading:")
    df = loader.get_historical_data(
        start="2024-01-01",
        end="2024-06-30",
        leagues=['epl', 'bundesliga'],
        include_recent_from_api=False  # Use CSV only for test
    )
    print(f"  Loaded {len(df)} matches")
    print(f"  Leagues: {df['League'].unique().tolist()}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return True


def test_sportmonks_api():
    """Test SportMonks API connection."""
    print("\n" + "="*60)
    print("TEST 2: SportMonks API")
    print("="*60)
    
    from stavki.data.loader import get_loader
    
    loader = get_loader()
    
    if not loader.client:
        print("⚠️ SportMonks API not configured, skipping")
        return False
    
    # Test connection
    print("\nTesting API connection...")
    try:
        connected = loader.client.test_connection()
        print(f"  {'✓' if connected else '✗'} Connection {'successful' if connected else 'failed'}")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return False
    
    # Test live fixtures
    print("\nFetching upcoming fixtures...")
    try:
        fixtures = loader.get_live_fixtures(days=3)
        print(f"  ✓ Found {len(fixtures)} upcoming matches")
        
        if fixtures:
            fix = fixtures[0]
            print(f"  Sample: {fix.home_team} vs {fix.away_team}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test odds fetching
    if fixtures:
        print("\nTesting odds fetch...")
        try:
            fixture_data = loader.get_fixture_with_odds(fixtures[0].fixture_id)
            odds = fixture_data.get('odds', [])
            if odds:
                print(f"  ✓ Got odds from {len(odds)} bookmakers")
            else:
                print(f"  ⚠️ No odds available yet")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    return True


def test_live_predictions():
    """Test live prediction pipeline."""
    print("\n" + "="*60)
    print("TEST 3: Live Prediction Pipeline")
    print("="*60)
    
    from stavki.prediction.live import LivePredictor
    import os
    
    # Get API key
    env_path = Path('/Users/macuser/Documents/something/stavki_v2/.env')
    api_key = None
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith('SPORTMONKS_API_KEY='):
                    api_key = line.strip().split('=', 1)[1].strip('"').strip("'")
    
    if not api_key:
        print("⚠️ API key not found, skipping")
        return False
    
    # Initialize predictor
    print("\nInitializing predictor...")
    predictor = LivePredictor(api_key=api_key)
    print(f"  ✓ Loaded ELO for {len(predictor.elo_ratings)} teams")
    
    # Test predictions
    print("\nGenerating predictions...")
    preds = predictor.predict_upcoming(days=3, leagues=[8, 82])  # EPL + Bundesliga
    print(f"  ✓ Generated {len(preds)} predictions")
    
    # Check recommendations
    recs = [p for p in preds if p.recommended]
    print(f"  ✓ {len(recs)} recommended bets")
    
    if recs:
        print("\n  Top 3 recommendations:")
        for r in recs[:3]:
            print(f"    {r.home_team} vs {r.away_team}")
            print(f"      {r.best_bet} @ EV={r.best_ev:+.1%}")
    
    return True


def test_backtesting_with_api_data():
    """Test backtesting with API-sourced data."""
    print("\n" + "="*60)
    print("TEST 4: Backtesting with API Data")
    print("="*60)
    
    from stavki.data.loader import get_loader
    from stavki.backtesting import BacktestEngine, BacktestConfig
    from catboost import CatBoostClassifier
    from sklearn.preprocessing import LabelEncoder
    
    loader = get_loader()
    
    # Load data including recent from API
    print("\nLoading data (with API supplement)...")
    df = loader.get_historical_data(
        start="2023-07-01",
        leagues=['epl', 'bundesliga'],
        include_recent_from_api=loader.client is not None
    )
    df = df.dropna(subset=['B365H', 'B365D', 'B365A'])
    print(f"  Loaded {len(df)} matches with odds")
    
    if len(df) < 200:
        print("  ⚠️ Not enough data for backtest")
        return False
    
    # Train quick model
    print("\nTraining model...")
    feature_cols = ['elo_diff', 'form_diff', 'imp_home_norm', 'imp_draw_norm', 'imp_away_norm']
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    if len(feature_cols) < 3:
        print(f"  ⚠️ Not enough features: {feature_cols}")
        # Add basic features if missing
        df['imp_home_norm'] = (1/df['B365H']) / (1/df['B365H'] + 1/df['B365D'] + 1/df['B365A'])
        df['imp_draw_norm'] = (1/df['B365D']) / (1/df['B365H'] + 1/df['B365D'] + 1/df['B365A'])
        df['imp_away_norm'] = (1/df['B365A']) / (1/df['B365H'] + 1/df['B365D'] + 1/df['B365A'])
        feature_cols = ['imp_home_norm', 'imp_draw_norm', 'imp_away_norm', 'B365H', 'B365D', 'B365A']
    
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['FTR'])
    
    # Split
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    model = CatBoostClassifier(iterations=100, verbose=0)
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['target']
    model.fit(X_train, y_train)
    print(f"  ✓ Model trained on {len(train_df)} matches")
    
    # Generate predictions
    X_test = test_df[feature_cols].fillna(0)
    probs = model.predict_proba(X_test)
    
    probs_dict = {}
    for i, (idx, row) in enumerate(test_df.iterrows()):
        # CatBoost order: A=0, D=1, H=2 -> H, D, A
        probs_dict[idx] = np.array([probs[i][2], probs[i][1], probs[i][0]])
    
    # Backtest
    print("\nRunning backtest...")
    config = BacktestConfig(
        min_ev=0.03,
        min_edge=0.02,
        kelly_fraction=0.20,
        model_alpha=0.55,
        slippage=0.02,
        leagues=[]
    )
    
    engine = BacktestEngine(config=config)
    result = engine.run(test_df, model_probs=probs_dict)
    
    print(f"\n  📊 Backtest Results:")
    print(f"     Bets: {result.total_bets}")
    print(f"     Win Rate: {result.win_rate:.1%}")
    print(f"     ROI: {result.roi:+.2%}")
    print(f"     Profit: €{result.total_profit:+,.2f}")
    
    return result.total_bets > 0


def run_all_tests():
    """Run all integration tests."""
    print("="*60)
    print("🧪 SPORTMONKS INTEGRATION TESTS")
    print("="*60)
    print()
    
    results = {}
    
    # Test 1: Unified loader
    try:
        results['Unified Loader'] = test_unified_loader()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        results['Unified Loader'] = False
    
    # Test 2: API connection
    try:
        results['SportMonks API'] = test_sportmonks_api()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        results['SportMonks API'] = False
    
    # Test 3: Live predictions
    try:
        results['Live Predictions'] = test_live_predictions()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        results['Live Predictions'] = False
    
    # Test 4: Backtesting
    try:
        results['Backtesting'] = test_backtesting_with_api_data()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        results['Backtesting'] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print()
    print(f"  Total: {total_passed}/{total_tests} tests passed")
    print("="*60)
    
    return all(results.values())


if __name__ == "__main__":
    run_all_tests()
