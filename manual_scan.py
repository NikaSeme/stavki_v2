import sys
import logging
import os

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', 
    stream=sys.stdout,
    force=True
)

print('🚀 Starting Manual Scan...')

try:
    from stavki.pipelines.daily import DailyPipeline, PipelineConfig
    
    # Use default config
    config = PipelineConfig() 
    # Use bankroll from env or default
    bankroll = float(os.getenv('BANKROLL', 1000.0))
    
    pipeline = DailyPipeline(config=config, bankroll=bankroll)
    bets = pipeline.run()
    
    print(f'\n✅ Scan Complete. Found {len(bets)} bets.')
    for b in bets:
        # Construct summary string
        print(f'  - {b.home_team} vs {b.away_team}: {b.selection} @ {b.odds} (EV: {b.ev:.1%}, Stake: ${b.stake_amount:.2f})')
        
except Exception as e:
    print(f'\n❌ Scan Failed: {e}')
    import traceback
    traceback.print_exc()
