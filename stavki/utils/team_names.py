"""
Team Name Normalization Utilities
=================================

Handles mapping between API team names (SportMonks, OddsAPI) and historical CSV names.
Critical for connecting live data to historical features (ELO, Form).
"""

import logging

logger = logging.getLogger(__name__)

# Mapping: API Name -> CSV Name
# CSV names are usually shorter/older style from football-data.co.uk
TEAM_MAPPING = {
    # Premier League
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "West Ham United": "West Ham",
    "Brighton & Hove Albion": "Brighton",
    "Brighton & Hove": "Brighton",
    "Leeds United": "Leeds",
    "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Stoke City": "Stoke",
    "Hull City": "Hull",
    "Cardiff City": "Cardiff",
    "Swansea City": "Swansea",
    "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR",
    "Blackburn Rovers": "Blackburn",
    "Bolton Wanderers": "Bolton",
    "Wigan Athletic": "Wigan",
    "Tottenham Hotspur": "Tottenham",
    "Nottingham Forest": "Nott'm Forest",
    "Luton Town": "Luton",
    
    # Championship variations
    "Sheffield Wednesday": "Sheffield Weds",
    "Preston North End": "Preston",
    
    # La Liga
    "Athletic Club": "Ath Bilbao",
    "Athletic Bilbao": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Atlético Madrid": "Ath Madrid",
    "Celta de Vigo": "Celta",
    "Celta Vigo": "Celta",
    "RCD Espanyol": "Espanyol",
    "RCD Mallorca": "Mallorca",
    "Real Betis": "Betis",
    "Real Sociedad": "Sociedad",
    "Real Valladolid": "Valladolid",
    "Rayo Vallecano": "Vallecano",
    "Deportivo Alavés": "Alaves",
    "Alavés": "Alaves",
    "Cádiz": "Cadiz",
    "UD Almería": "Almeria",
    "Granada CF": "Granada",
    "Sporting Gijón": "Sp Gijon",
    "Málaga": "Malaga",
    "Deportivo La Coruña": "La Coruna",
    
    # Bundesliga
    "FC Bayern München": "Bayern Munich",
    "Bayern Munich": "Bayern Munich",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Mönchengladbach": "M'gladbach",
    "Borussia Monchengladbach": "M'gladbach",
    "Borussia Dortmund": "Dortmund",
    "Dortmund": "Dortmund",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "SC Freiburg": "Freiburg",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "TSG Hoffenheim": "Hoffenheim",
    "1. FC Köln": "FC Koln",
    "FC Köln": "FC Koln",
    "1. FSV Mainz 05": "Mainz",
    "FSV Mainz 05": "Mainz",
    "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "VfL Bochum": "Bochum",
    "FC Augsburg": "Augsburg",
    "1. FC Union Berlin": "Union Berlin",
    "FC Union Berlin": "Union Berlin",
    "Union Berlin": "Union Berlin",
    "Arminia Bielefeld": "Bielefeld",
    "Greuther Fürth": "Greuther Furth",
    "Hertha BSC": "Hertha",
    "FC Schalke 04": "Schalke 04",
    "Werder Bremen": "Werder Bremen",
    "Hamburger SV": "Hamburg",
    "Fortuna Düsseldorf": "Fortuna Dusseldorf",
    "Hannover 96": "Hannover",
    "1. FC Nürnberg": "Nurnberg",
    "RB Leipzig": "RB Leipzig",
    
    # Premier League / English
    "AFC Bournemouth": "Bournemouth",
    "Nottingham Forest": "Nott'm Forest",
    
    # La Liga
    "RCD Espanyol": "Espanol",
    "RCD Espanyol de Barcelona": "Espanol",
    "Espanyol": "Espanol",
    
    # Serie A
    "Inter Milan": "Inter",
    "Internazionale": "Inter",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    "Hellas Verona": "Verona",
    "SPAL": "Spal",
    
    # Ligue 1
    "Paris Saint-Germain": "Paris SG",
    "Paris Saint Germain": "Paris SG",
    "PSG": "Paris SG",
    "AS Monaco": "Monaco",
    "Olympique Lyonnais": "Lyon",
    "Olympique Marseille": "Marseille",
    "AS Saint-Étienne": "St Etienne",
    "Saint-Étienne": "St Etienne",
}

def normalize_team_name(name: str) -> str:
    """
    Normalize team name to match historical CSV data.
    """
    if not name:
        return ""
        
    # 1. Clean basic noise
    clean_name = name.strip()
    
    # 2. Check direct mapping
    if clean_name in TEAM_MAPPING:
        return TEAM_MAPPING[clean_name]
    
    # 3. Fallback: try removing accents if no match found
    # (Simplified for now, can be expanded)
    
    return clean_name
