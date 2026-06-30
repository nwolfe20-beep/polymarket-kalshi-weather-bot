"""Weather data fetcher using Open-Meteo Ensemble API, NWS observations, and Wethr API."""
import httpx
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import statistics
import time

logger = logging.getLogger("trading_bot")

# City configurations with lat/lon and NWS station identifiers
CITY_CONFIG: Dict[str, dict] = {
    "nyc": {
        "name": "New York City",
        "lat": 40.7128,
        "lon": -74.0060,
        "nws_station": "KNYC",
        "nws_office": "OKX",
        "nws_gridpoint": "OKX/33,37",
        "wethr_station": "KNYC",
    },
    "chicago": {
        "name": "Chicago",
        "lat": 41.8781,
        "lon": -87.6298,
        "nws_station": "KORD",
        "nws_office": "LOT",
        "nws_gridpoint": "LOT/75,72",
        "wethr_station": "KORD",
    },
    "miami": {
        "name": "Miami",
        "lat": 25.7617,
        "lon": -80.1918,
        "nws_station": "KMIA",
        "nws_office": "MFL",
        "nws_gridpoint": "MFL/75,53",
        "wethr_station": "KMIA",
    },
    "los_angeles": {
        "name": "Los Angeles",
        "lat": 34.0522,
        "lon": -118.2437,
        "nws_station": "KLAX",
        "nws_office": "LOX",
        "nws_gridpoint": "LOX/154,44",
        "wethr_station": "KLAX",
    },
    "denver": {
        "name": "Denver",
        "lat": 39.7392,
        "lon": -104.9903,
        "nws_station": "KDEN",
        "nws_office": "BOU",
        "nws_gridpoint": "BOU/62,60",
        "wethr_station": "KDEN",
    },
}

@dataclass
class EnsembleForecast:
    """Ensemble weather forecast with per-member data."""
    city_key: str
    city_name: str
    target_date: date
    member_highs: List[float]  # Daily max temps (F) per ensemble member
    member_lows: List[float]   # Daily min temps (F) per ensemble member
    mean_high: float = 0.0
    std_high: float = 0.0
    mean_low: float = 0.0
    std_low: float = 0.0
    num_members: int = 0
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    # Wethr metadata
    wethr_high: Optional[float] = None   # Wethr model forecast high
    wethr_source: str = "gfs_only"       # "gfs_only", "wethr_blended"

    def __post_init__(self):
        if self.member_highs:
            self.mean_high = statistics.mean(self.member_highs)
            self.std_high = statistics.stdev(self.member_highs) if len(self.member_highs) > 1 else 0.0
            self.num_members = len(self.member_highs)
        if self.member_lows:
            self.mean_low = statistics.mean(self.member_lows)
            self.std_low = statistics.stdev(self.member_lows) if len(self.member_lows) > 1 else 0.0

    def probability_high_above(self, threshold_f: float) -> float:
        """Fraction of ensemble members with daily high above threshold."""
        if not self.member_highs:
            return 0.5
        count = sum(1 for h in self.member_highs if h > threshold_f)
        return count / len(self.member_highs)

    def probability_high_below(self, threshold_f: float) -> float:
        return 1.0 - self.probability_high_above(threshold_f)

    def probability_low_above(self, threshold_f: float) -> float:
        if not self.member_lows:
            return 0.5
        count = sum(1 for l in self.member_lows if l > threshold_f)
        return count / len(self.member_lows)

    def probability_low_below(self, threshold_f: float) -> float:
        return 1.0 - self.probability_low_above(threshold_f)

    @property
    def ensemble_agreement(self) -> float:
        if not self.member_highs:
            return 0.5
        median = statistics.median(self.member_highs)
        above = sum(1 for h in self.member_highs if h > median)
        frac = above / len(self.member_highs)
        return max(frac, 1 - frac)


# Cache: (city_key, target_date_str) -> (timestamp, EnsembleForecast)
_forecast_cache: Dict[str, tuple] = {}
_CACHE_TTL = 900  # 15 minutes

def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


async def fetch_wethr_forecast(city_key: str, target_date: date, api_key: str) -> Optional[float]:
    """
    Fetch Wethr model forecast high for a city.
    Returns predicted high temp in Fahrenheit, or None if unavailable.
    Uses Wethr's /forecasts.php endpoint in daily mode.
    """
    city = CITY_CONFIG.get(city_key)
    if not city or not api_key:
        return None

    station = city.get("wethr_station", city.get("nws_station"))
    date_str = target_date.isoformat()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://wethr.net/api/v2/forecasts.php",
                params={
                    "location_name": station,
                    "mode": "daily",
                    "tz_mode": "standard",
                },
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code != 200:
                logger.warning(f"Wethr forecast HTTP {resp.status_code} for {station}")
                return None

            data = resp.json()
            if not isinstance(data, list):
                return None

            # Find today's or tomorrow's entry
            for entry in data:
                if entry.get("date") == date_str and entry.get("high_f") is not None:
                    high_f = float(entry["high_f"])
                    logger.info(f"Wethr forecast {station} {date_str}: {high_f}F")
                    return high_f

            logger.debug(f"Wethr no entry for {station} {date_str}")
            return None

    except Exception as e:
        logger.warning(f"Wethr forecast fetch error for {station}: {e}")
        return None


async def fetch_ensemble_forecast(
    city_key: str,
    target_date: date,
    wethr_api_key: Optional[str] = None,
    wethr_blend_weight: float = 0.5,
) -> Optional[EnsembleForecast]:
    """
    Fetch ensemble forecast for a city blending GFS + Wethr.

    If WETHR_API_KEY is set:
      - Fetch Wethr forecast high
      - Blend: effective_mean = (gfs_mean * (1-w)) + (wethr_high * w)
      - Shift all GFS member highs by the delta to produce blended members

    If no Wethr key, falls back to GFS-only (original behavior).
    """
    cache_key = f"{city_key}_{target_date.isoformat()}"
    if cache_key in _forecast_cache:
        ts, cached = _forecast_cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return cached

    city = CITY_CONFIG.get(city_key)
    if not city:
        logger.error(f"Unknown city key: {city_key}")
        return None

    # Fetch GFS ensemble from Open-Meteo
    try:
        target_str = target_date.isoformat()
        url = (
            f"https://ensemble-api.open-meteo.com/v1/ensemble"
            f"?latitude={city['lat']}&longitude={city['lon']}"
            f"&hourly=temperature_2m"
            f"&models=gfs_seamless"
            f"&temperature_unit=fahrenheit"
            f"&timezone=auto"
            f"&forecast_days=7"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        times = data.get("hourly", {}).get("time", [])
        all_keys = list(data.get("hourly", {}).keys())
        member_keys = [k for k in all_keys if k.startswith("temperature_2m_member")]

        if not member_keys:
            # Single-member fallback
            member_keys = ["temperature_2m"] if "temperature_2m" in all_keys else []

        member_highs = []
        member_lows = []

        for mk in member_keys:
            temps = data["hourly"].get(mk, [])
            day_highs = []
            day_lows = []
            for i, t in enumerate(times):
                if t.startswith(target_str):
                    v = temps[i] if i < len(temps) else None
                    if v is not None:
                        day_highs.append(v)
                        day_lows.append(v)
            if day_highs:
                member_highs.append(max(day_highs))
                member_lows.append(min(day_lows))

        if not member_highs:
            logger.warning(f"No GFS ensemble data for {city_key} on {target_date}")
            return None

        gfs_mean = statistics.mean(member_highs)

        # Fetch Wethr forecast if API key available
        wethr_high = None
        source = "gfs_only"

        if wethr_api_key:
            wethr_high = await fetch_wethr_forecast(city_key, target_date, wethr_api_key)

        if wethr_high is not None and wethr_blend_weight > 0:
            # Blend: shift GFS members so mean matches the weighted blend
            blended_mean = gfs_mean * (1 - wethr_blend_weight) + wethr_high * wethr_blend_weight
            delta = blended_mean - gfs_mean
            member_highs = [h + delta for h in member_highs]
            source = "wethr_blended"
            logger.info(
                f"{city_key} {target_date}: GFS mean={gfs_mean:.1f}F "
                f"Wethr={wethr_high:.1f}F blended={blended_mean:.1f}F (delta={delta:+.1f}F)"
            )

        forecast = EnsembleForecast(
            city_key=city_key,
            city_name=city["name"],
            target_date=target_date,
            member_highs=member_highs,
            member_lows=member_lows,
            wethr_high=wethr_high,
            wethr_source=source,
        )

        _forecast_cache[cache_key] = (time.time(), forecast)
        return forecast

    except Exception as e:
        logger.error(f"Error fetching ensemble forecast for {city_key}: {e}")
        return None


async def fetch_nws_observed_high(city_key: str, target_date: date) -> Optional[float]:
    """Fetch observed high temperature from NWS for settlement checking."""
    city = CITY_CONFIG.get(city_key)
    if not city:
        return None

    try:
        station = city["nws_station"]
        url = f"https://api.weather.gov/stations/{station}/observations"
        date_str = target_date.isoformat()

        async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": "TradingBot/1.0"}) as client:
            resp = await client.get(url, params={"start": f"{date_str}T00:00:00Z", "end": f"{date_str}T23:59:59Z"})
            if resp.status_code != 200:
                return None
            obs = resp.json()

        features = obs.get("features", [])
        temps_f = []
        for f in features:
            temp_c = f.get("properties", {}).get("temperature", {}).get("value")
            if temp_c is not None:
                temps_f.append(_celsius_to_fahrenheit(temp_c))

        return max(temps_f) if temps_f else None

    except Exception as e:
        logger.warning(f"NWS observed high fetch error for {city_key}: {e}")
        return None
