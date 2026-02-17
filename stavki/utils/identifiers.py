
import pandas as pd
from typing import Optional, Union
import re

def generate_match_id(
    home_team: str, 
    away_team: str, 
    date: Optional[Union[str, pd.Timestamp]] = None
) -> str:
    """
    Generate a consistent unique identifier for a match.
    
    Format: {home_team}_vs_{away_team}_{date_str}
    
    Args:
        home_team: Name of home team
        away_team: Name of away team
        date: Match date (optional, but recommended for uniqueness)
    
    Returns:
        standardized match_id string
    """
    # Normalize names: lowercase, remove special chars, replace spaces with underscores
    def clean(s):
        if not isinstance(s, str):
            return str(s)
        s = s.lower().strip()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        return s.strip('_')

    h = clean(home_team)
    a = clean(away_team)
    
    base_id = f"{h}_vs_{a}"
    
    if date is not None:
        # Format date as YYYYMMDD if possible
        if hasattr(date, 'strftime'):
            d = date.strftime('%Y%m%d')
        else:
            # Try to parse string or just use first 10 chars (YYYY-MM-DD)
            try:
                d = pd.to_datetime(date).strftime('%Y%m%d')
            except:
                d = str(date)[:10].replace('-', '').replace('/', '')
        
        return f"{base_id}_{d}"
        
    return base_id
