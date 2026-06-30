"""Signal generator for weather temperature markets using ensemble forecasts + Wethr blend."""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from backend.config import settings
from backend.core.signals import calculate_edge, calculate_kelly_size
from backend.data.weather import fetch_ensemble_forecast, EnsembleForecast, CITY_CONFIG
from backend.data.weather_markets import WeatherMarket, fetch_polymarket_weather_markets
from backend.models.database import SessionLocal, Signal

logger = logging.getLogger("trading_bot")

@dataclass
class WeatherTradingSignal:
    """A trading signal for a weather temperature market."""
    market: WeatherMarket

    # Core signal data
    model_probability: float = 0.5   # Ensemble probability of YES outcome
    market_probability: float = 0.5  # Market's implied YES probability
    edge: float = 0.0
    direction: str = "yes"           # "yes" or "no"

    # Confidence and sizing
    confidence: float = 0.5
    kelly_fraction: float = 0.0
    suggested_size: float = 0.0

    # Metadata
    sources: List[str] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Forecast context
    ensemble_mean: float = 0.0
    ensemble_std: float = 0.0
    ensemble_members: int = 0
    wethr_high: Optional[float] = None   # Wethr model forecast if available
    wethr_source: str = "gfs_only"       # "gfs_only" or "wethr_blended"

    @property
    def passes_threshold(self) -> bool:
        """Check if signal passes minimum edge threshold."""
        return abs(self.edge) >= settings.WEATHER_MIN_EDGE_THRESHOLD


async def generate_weather_signal(market: WeatherMarket) -> Optional[WeatherTradingSignal]:
    """
    Generate a trading signal for a weather temperature market.

    Uses blended ensemble forecast (GFS + Wethr if API key available):
    - Count fraction of blended ensemble members above/below the threshold
    - Compare to market price to find edge
    - Size using Kelly criterion
    """
    # Pass Wethr credentials if configured
    forecast = await fetch_ensemble_forecast(
        market.city_key,
        market.target_date,
        wethr_api_key=settings.WETHR_API_KEY,
        wethr_blend_weight=settings.WETHR_BLEND_WEIGHT,
    )
    if not forecast or not forecast.member_highs:
        return None

    # Calculate model probability based on market's question
    if market.metric == "high":
        if market.direction == "above":
            model_yes_prob = forecast.probability_high_above(market.threshold_f)
        else:
            model_yes_prob = forecast.probability_high_below(market.threshold_f)
    else:  # "low"
        if market.direction == "above":
            model_yes_prob = forecast.probability_low_above(market.threshold_f)
        else:
            model_yes_prob = forecast.probability_low_below(market.threshold_f)

    # Clip extreme probabilities
    model_yes_prob = max(0.05, min(0.95, model_yes_prob))

    market_yes_prob = market.yes_price

    # Edge calculation
    edge, direction_raw = calculate_edge(model_yes_prob, market_yes_prob)
    direction = "yes" if direction_raw == "up" else "no"

    # Entry price filter
    entry_price = market.yes_price if direction == "yes" else market.no_price
    if entry_price > settings.WEATHER_MAX_ENTRY_PRICE:
        edge = 0.0

    # Confidence = ensemble agreement
    if market.metric == "high":
        members = forecast.member_highs
    else:
        members = forecast.member_lows

    above_count = sum(1 for m in members if m > market.threshold_f)
    agreement_frac = max(above_count, len(members) - above_count) / len(members)
    confidence = min(0.9, agreement_frac)

    # Kelly sizing
    bankroll = settings.INITIAL_BANKROLL
    suggested_size = calculate_kelly_size(
        edge=abs(edge),
        probability=model_yes_prob,
        market_price=market_yes_prob,
        direction=direction_raw,
        bankroll=bankroll,
    )
    suggested_size = min(suggested_size, settings.WEATHER_MAX_TRADE_SIZE)

    # Ensemble stats for display
    mean_val = forecast.mean_high if market.metric == "high" else forecast.mean_low
    std_val = forecast.std_high if market.metric == "high" else forecast.std_low

    # Build reasoning — include Wethr source tag
    filter_status = "ACTIONABLE" if abs(edge) >= settings.WEATHER_MIN_EDGE_THRESHOLD else "FILTERED"
    filter_notes = []
    if entry_price > settings.WEATHER_MAX_ENTRY_PRICE:
        filter_notes.append(f"entry {entry_price:.0%} > {settings.WEATHER_MAX_ENTRY_PRICE:.0%}")
    filter_note = f" [{', '.join(filter_notes)}]" if filter_notes else ""
    source_tag = f"[{forecast.wethr_source}]"
    wethr_note = f" wethr={forecast.wethr_high:.1f}F" if forecast.wethr_high else ""

    reasoning = (
        f"[{filter_status}]{filter_note}{source_tag} "
        f"model={model_yes_prob:.1%} mkt={market_yes_prob:.1%} edge={edge:+.1%} "
        f"dir={direction} entry={entry_price:.0%} "
        f"ensemble: mean={mean_val:.1f}F std={std_val:.1f}F n={forecast.num_members}"
        f"{wethr_note}"
    )

    signal = WeatherTradingSignal(
        market=market,
        model_probability=model_yes_prob,
        market_probability=market_yes_prob,
        edge=edge,
        direction=direction,
        confidence=confidence,
        kelly_fraction=suggested_size / bankroll if bankroll > 0 else 0,
        suggested_size=suggested_size,
        sources=["open_meteo_gfs"] + (["wethr"] if forecast.wethr_high else []),
        reasoning=reasoning,
        ensemble_mean=mean_val,
        ensemble_std=std_val,
        ensemble_members=forecast.num_members,
        wethr_high=forecast.wethr_high,
        wethr_source=forecast.wethr_source,
    )

    # Log to DB
    try:
        with SessionLocal() as db:
            db_signal = Signal(
                market_id=market.market_id,
                platform=market.platform,
                market_type="weather",
                city=market.city_key,
                direction=direction,
                model_probability=model_yes_prob,
                market_probability=market_yes_prob,
                edge=edge,
                confidence=confidence,
                suggested_size=suggested_size,
                reasoning=reasoning,
            )
            db.add(db_signal)
            db.commit()
    except Exception as e:
        logger.warning(f"Failed to log weather signal to DB: {e}")

    return signal


async def scan_weather_markets() -> List[WeatherTradingSignal]:
    """Scan all weather markets and generate signals."""
    from backend.data.kalshi_markets import fetch_kalshi_weather_markets

    signals = []
    cities = [c.strip() for c in settings.WEATHER_CITIES.split(",") if c.strip()]

    # Fetch from both Kalshi and Polymarket
    all_markets: List[WeatherMarket] = []
    try:
        kalshi_markets = await fetch_kalshi_weather_markets(cities)
        all_markets.extend(kalshi_markets)
        logger.info(f"Fetched {len(kalshi_markets)} Kalshi weather markets")
    except Exception as e:
        logger.warning(f"Kalshi weather market fetch failed: {e}")

    try:
        poly_markets = await fetch_polymarket_weather_markets(cities)
        all_markets.extend(poly_markets)
        logger.info(f"Fetched {len(poly_markets)} Polymarket weather markets")
    except Exception as e:
        logger.warning(f"Polymarket weather market fetch failed: {e}")

    if not all_markets:
        logger.warning("No weather markets found")
        return signals

    # Generate signals
    for market in all_markets:
        try:
            signal = await generate_weather_signal(market)
            if signal:
                signals.append(signal)
                if signal.passes_threshold:
                    logger.info(f"WEATHER SIGNAL: {signal.reasoning}")
                else:
                    logger.debug(f"Weather signal filtered: {signal.reasoning}")
        except Exception as e:
            logger.error(f"Error generating weather signal for {market.market_id}: {e}")

    return signals
