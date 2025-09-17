"""Physics-based rate-of-spread helpers built around a simplified Rothermel model.

The goal of this module is to expose a deterministic interface for the simulation
engine to reason about directional spread probabilities using fuel, weather, and
terrain inputs. The implementation intentionally keeps the math lightweight so it
can run inside Celery workers without scientific dependencies, while still
surfacing the key multipliers from the Rothermel formulation (moisture, wind,
slope). Future work can swap in a higher-fidelity backend without changing the
public API defined here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

__all__ = [
    "FuelModel",
    "FUEL_CATALOG",
    "RateOfSpreadRequest",
    "RateOfSpreadResult",
    "RothermelCalculator",
]


@dataclass(frozen=True)
class FuelModel:
    """Parameters describing a surface fuel bed.

    Values are stored in metric units to make downstream calculations easier.
    The numbers below are adapted from the standard 13 fuel models published by
    Anderson (1982) and converted from imperial units.
    """

    code: str
    description: str
    fuel_load_kg_m2: float  # Total surface fuel load
    heat_content_kJ_kg: float
    fuel_bed_depth_m: float
    surface_area_to_volume: float  # 1/m
    base_ros_m_min: float  # Reference spread rate (no wind/slope, dry fuel)
    ext_moisture_dead: float  # Moisture of extinction (fraction 0-1)
    ext_moisture_live: float  # Moisture of extinction (fraction 0-1)


FUEL_CATALOG: Mapping[str, FuelModel] = {
    # Grass models
    "GR1": FuelModel(
        code="GR1",
        description="Short grass (1 ft)",
        fuel_load_kg_m2=0.17,
        heat_content_kJ_kg=18600.0,
        fuel_bed_depth_m=0.3,
        surface_area_to_volume=3500.0,
        base_ros_m_min=2.0,
        ext_moisture_dead=0.12,
        ext_moisture_live=0.60,
    ),
    "GR4": FuelModel(
        code="GR4",
        description="Moderate load, dry climate grass",
        fuel_load_kg_m2=0.43,
        heat_content_kJ_kg=18600.0,
        fuel_bed_depth_m=0.46,
        surface_area_to_volume=2200.0,
        base_ros_m_min=3.0,
        ext_moisture_dead=0.15,
        ext_moisture_live=0.70,
    ),
    # Grass-shrub models
    "GS2": FuelModel(
        code="GS2",
        description="Moderate load, dry climate grass-shrub",
        fuel_load_kg_m2=0.75,
        heat_content_kJ_kg=18600.0,
        fuel_bed_depth_m=0.5,
        surface_area_to_volume=2000.0,
        base_ros_m_min=1.6,
        ext_moisture_dead=0.18,
        ext_moisture_live=1.20,
    ),
    # Shrub models
    "SH5": FuelModel(
        code="SH5",
        description="High load, dry climate shrub",
        fuel_load_kg_m2=1.12,
        heat_content_kJ_kg=19000.0,
        fuel_bed_depth_m=0.9,
        surface_area_to_volume=1800.0,
        base_ros_m_min=1.3,
        ext_moisture_dead=0.20,
        ext_moisture_live=1.50,
    ),
    # Timber litter models
    "TL3": FuelModel(
        code="TL3",
        description="Moderate load conifer litter",
        fuel_load_kg_m2=0.74,
        heat_content_kJ_kg=18000.0,
        fuel_bed_depth_m=0.3,
        surface_area_to_volume=1400.0,
        base_ros_m_min=0.7,
        ext_moisture_dead=0.25,
        ext_moisture_live=1.80,
    ),
    "TU5": FuelModel(
        code="TU5",
        description="Very high load conifer with shrubs",
        fuel_load_kg_m2=1.96,
        heat_content_kJ_kg=19500.0,
        fuel_bed_depth_m=0.9,
        surface_area_to_volume=1600.0,
        base_ros_m_min=1.0,
        ext_moisture_dead=0.25,
        ext_moisture_live=1.80,
    ),
}


@dataclass(frozen=True)
class RateOfSpreadRequest:
    """Input bundle for computing a rate-of-spread (ROS).

    The engine collects per-scenario conditions and packages them into this
    request before calling :class:`RothermelCalculator`.
    """

    fuel_model: str
    wind_speed_ms: float  # 10-m wind speed (m/s)
    wind_direction_deg: float  # Meteorological direction [0-360), blowing *from*
    slope_degrees: float
    aspect_degrees: float
    moisture_dead: float  # Fraction 0-1
    moisture_live: float  # Fraction 0-1
    air_temperature_c: float
    wind_reduction_factor: float = 0.4  # Convert 10 m wind to mid-flame
    fuel_moisture_adjustment: float = 1.0
    crown_fire: bool = False


@dataclass(frozen=True)
class RateOfSpreadResult:
    ros_m_min: float
    ros_m_s: float
    effective_wind_ms: float
    wind_factor: float
    slope_factor: float
    moisture_factor: float

    def as_probability(self, cell_size_m: float, timestep_minutes: float) -> float:
        """Convert rate-of-spread to a per-timestep ignition probability.

        The formulation assumes a backing fire needs to traverse ``cell_size``
        metres in ``timestep`` minutes. The probability is capped to keep Monte
        Carlo sampling stable.
        """

        if cell_size_m <= 0 or timestep_minutes <= 0:
            return 0.0
        distance_possible = self.ros_m_min * timestep_minutes
        probability = distance_possible / max(cell_size_m, 1e-6)
        return float(np.clip(probability, 0.0, 0.98))


class RothermelCalculator:
    """Lightweight ROS calculator with caching.

    The calculator is stateless; repeated queries for the same fuel model will
    hit a small LRU cache to avoid repeated dictionary lookups and expensive
    exponentials.
    """

    DIRECTION_DEGREES: Mapping[str, float] = {
        "N": 0.0,
        "NE": 45.0,
        "E": 90.0,
        "SE": 135.0,
        "S": 180.0,
        "SW": 225.0,
        "W": 270.0,
        "NW": 315.0,
    }

    def __init__(self, fuel_catalog: Mapping[str, FuelModel] | None = None) -> None:
        self._catalog = fuel_catalog or FUEL_CATALOG

    @lru_cache(maxsize=32)
    def get_fuel(self, fuel_code: str) -> FuelModel:
        if fuel_code not in self._catalog:
            raise KeyError(f"Unknown fuel model '{fuel_code}'. Available: {sorted(self._catalog.keys())}")
        return self._catalog[fuel_code]

    def compute(self, request: RateOfSpreadRequest) -> RateOfSpreadResult:
        """Compute an omni-directional ROS.

        The implementation blends canonical Rothermel multipliers with a handful
        of pragmatic clamps to keep simulation inputs sane even when scenarios
        provide extreme or missing data.
        """

        fuel = self.get_fuel(request.fuel_model)

        moisture_dead = np.clip(request.moisture_dead, 0.01, 0.6)
        moisture_live = np.clip(request.moisture_live, 0.01, 3.0)

        dead_factor = max(0.05, 1.0 - (moisture_dead / max(fuel.ext_moisture_dead, 1e-3)))
        live_factor = max(0.05, 1.0 - (moisture_live / max(fuel.ext_moisture_live, 1e-3)))
        moisture_factor = float(np.clip((dead_factor + live_factor) / 2.0, 0.05, 2.0))

        midflame_wind = max(request.wind_speed_ms * request.wind_reduction_factor, 0.0)
        wind_factor = math.exp(0.1783 * min(midflame_wind, 30.0))

        slope_radians = math.radians(np.clip(request.slope_degrees, 0.0, 80.0))
        slope_factor = math.exp(5.275 * (fuel.surface_area_to_volume ** -0.3) * (math.tan(slope_radians) ** 2))

        base_ros = fuel.base_ros_m_min * request.fuel_moisture_adjustment
        if request.crown_fire:
            base_ros *= 1.5

        ros_m_min = base_ros * moisture_factor * wind_factor * slope_factor
        ros_m_min = float(np.clip(ros_m_min, 0.01, 120.0))

        return RateOfSpreadResult(
            ros_m_min=ros_m_min,
            ros_m_s=ros_m_min / 60.0,
            effective_wind_ms=midflame_wind,
            wind_factor=wind_factor,
            slope_factor=slope_factor,
            moisture_factor=moisture_factor,
        )

    def directional_distribution(
        self,
        request: RateOfSpreadRequest,
        cell_size_m: float,
        timestep_minutes: float,
    ) -> Dict[str, float]:
        """Return a per-octant ignition probability distribution.

        The distribution respects the alignment of the wind and terrain aspect
        such that the downwind + upslope octants receive more weight.
        """

        base_result = self.compute(request)

        wind_dir = request.wind_direction_deg % 360.0
        aspect_dir = request.aspect_degrees % 360.0

        weights: Dict[str, float] = {}
        for cardinal, degrees in self.DIRECTION_DEGREES.items():
            wind_alignment = _circular_cosine(wind_dir, degrees)
            slope_alignment = _circular_cosine(aspect_dir, degrees)
            alignment = 1.0 + 0.6 * wind_alignment + 0.4 * slope_alignment
            weight = max(alignment, 0.05)
            weights[cardinal] = weight

        total_weight = sum(weights.values())
        if total_weight <= 0:
            normalized = {cardinal: 1.0 / len(weights) for cardinal in weights}
        else:
            normalized = {cardinal: weight / total_weight for cardinal, weight in weights.items()}

        prob_base = base_result.as_probability(cell_size_m=cell_size_m, timestep_minutes=timestep_minutes)

        return {cardinal: prob_base * normalized[cardinal] for cardinal in normalized}

    def spread_multiplier(self, request: RateOfSpreadRequest, baseline_rate: float) -> float:
        """Translate ROS into a multiplier for the temporal spread rate.

        ``baseline_rate`` is the historical Markov spread rate. The multiplier
        increases or decreases that rate based on physics. Clamped to keep the
        Poisson process stable.
        """

        base_result = self.compute(request)
        ros_relative = base_result.ros_m_min / max(1e-3, baseline_rate * 60.0)
        return float(np.clip(ros_relative, 0.1, 10.0))


def _circular_cosine(angle_a: float, angle_b: float) -> float:
    """Return the cosine of the smallest angle between two bearings (degrees)."""

    delta = math.radians((angle_a - angle_b + 180.0) % 360.0 - 180.0)
    return math.cos(delta)


def build_ros_probability(
    calculator: RothermelCalculator,
    request: RateOfSpreadRequest,
    cell_size_m: float,
    timestep_minutes: float,
) -> Tuple[RateOfSpreadResult, Dict[str, float]]:
    """Convenience wrapper used by the engine.

    Returns the omni-directional ROS result along with the per-octant ignition
    probability distribution.
    """

    result = calculator.compute(request)
    distribution = calculator.directional_distribution(request, cell_size_m, timestep_minutes)
    return result, distribution
