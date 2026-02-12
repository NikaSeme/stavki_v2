"""
Weather Feature Builder (Tier 2).

Features from weather conditions at the venue:
- Temperature impact on play speed
- Rain/precipitation → lower scoring
- Wind → affects long balls and crossing
- Humidity → fatigue factor
"""

from typing import Dict, Optional
from datetime import datetime
import logging

from stavki.data.schemas import Match

logger = logging.getLogger(__name__)


class WeatherFeatureBuilder:
    """
    Compute weather-impact features for a match.
    
    Uses enrichment.weather data (from SportMonks weather API).
    For historical data without weather info, returns neutral defaults.
    """
    
    name = "weather"
    
    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Get weather features for a match."""
        defaults = {
            "temperature_c": 15.0,      # Room temp, neutral
            "wind_speed_ms": 3.0,       # Light breeze
            "is_rain": 0.0,
            "weather_scoring_impact": 0.0,  # Neutral
        }
        
        if not match or not match.enrichment or not match.enrichment.weather:
            return defaults
        
        w = match.enrichment.weather
        temp = w.temperature_c if w.temperature_c is not None else 15.0
        wind = w.wind_speed_ms if w.wind_speed_ms is not None else 3.0
        precip = w.precipitation_mm if w.precipitation_mm is not None else 0.0
        desc = (w.description or "").lower()
        
        # Rain detection
        is_rain = 1.0 if (precip > 0.5 or "rain" in desc or "shower" in desc) else 0.0
        
        # Scoring impact model:
        # Heavy rain: ~-10% goals
        # Strong wind (>10 m/s): ~-8% goals
        # Extreme cold (<0°C): ~-5% goals
        # Extreme heat (>30°C): ~-3% goals (fatigue)
        impact = 0.0
        
        # Rain effect
        if precip > 2.0:
            impact -= 0.10  # Heavy rain
        elif precip > 0.5:
            impact -= 0.05  # Light rain
        
        # Wind effect
        if wind > 15:
            impact -= 0.08  # Strong wind
        elif wind > 10:
            impact -= 0.04  # Moderate wind
        
        # Temperature extremes
        if temp < 0:
            impact -= 0.05
        elif temp > 30:
            impact -= 0.03
        
        return {
            "temperature_c": round(temp, 1),
            "wind_speed_ms": round(wind, 1),
            "is_rain": is_rain,
            "weather_scoring_impact": round(impact, 3),
        }
