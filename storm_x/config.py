"""Pydantic settings loaded from config.yaml at the project root."""
from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class EnsembleModel(BaseModel):
    name: str
    expected_members: int


class EnsembleConfig(BaseModel):
    models: List[EnsembleModel]
    cache_ttl_minutes: int = 10


class CityConfig(BaseModel):
    lat: float
    lon: float
    station: str
    wunderground_path: str
    timezone: str


class BettingConfig(BaseModel):
    bankroll: float
    daily_exposure_cap: float
    per_market_cap: float
    kelly_multiplier: float


class FiltersConfig(BaseModel):
    min_volume_usdc: float
    max_spread_cents: int
    min_traders_24h: int
    min_open_interest_usdc: float


class SizingConfig(BaseModel):
    bootstrap_samples: int
    conservative_percentile: int


class EdgesConfig(BaseModel):
    base_edge_threshold: float
    tail_threshold_price: float
    tail_edge_threshold: float
    model_agreement_max_disagree: float
    max_pool_spread: float
    climatology_blend_long_horizon_weight: float
    climatology_blend_short_horizon_weight: float


class BiasConfig(BaseModel):
    rolling_window_days: int = 90
    min_samples_same_month: int = 30


class ExitsConfig(BaseModel):
    profit_trigger_fraction: float = 0.70


class ExecutionLayer(BaseModel):
    fraction: float
    discount_cents: int


class ExecutionConfig(BaseModel):
    layers: List[ExecutionLayer]
    fill_timeout_seconds: int = 300
    fallback_premium_cents: int = 1


class PaperTraderConfig(BaseModel):
    enabled: bool = True


class HealthConfig(BaseModel):
    port: int = 8003


class StorageConfig(BaseModel):
    db_path: str
    climatology_years: int = 10


class LoggingConfig(BaseModel):
    level: str = "INFO"
    rotation: str = "100 MB"
    retention: str = "30 days"


class Settings(BaseModel):
    cities: dict[str, CityConfig]
    ensemble: EnsembleConfig
    betting: BettingConfig
    filters: FiltersConfig
    sizing: SizingConfig
    edges: EdgesConfig
    bias: BiasConfig = Field(default_factory=BiasConfig)
    calibration: dict = Field(default_factory=lambda: {"min_resolved_bets": 60})
    exits: ExitsConfig = Field(default_factory=ExitsConfig)
    execution: ExecutionConfig
    paper_trader: PaperTraderConfig = Field(default_factory=PaperTraderConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    storage: StorageConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def city(self, name: str) -> CityConfig:
        if name not in self.cities:
            raise KeyError(f"Unknown city '{name}'. Available: {list(self.cities)}")
        return self.cities[name]


def load_settings(path: Path = _CONFIG_PATH) -> Settings:
    """Load and validate config.yaml into a Settings object."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Settings.model_validate(raw)


# Module-level singleton — import this everywhere
settings: Settings = load_settings()
