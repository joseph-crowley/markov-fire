from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TemporalParameters:
    spread_rate: float = 0.05
    extinguish_rate: float = 0.05
    firefighting_rate: float = 0.02
    initial_population: int = 10


@dataclass
class PhysicsParameters:
    """Controls for the physics-backed rate-of-spread module."""

    enabled: bool = False
    fuel_model: str = 'GR1'
    cell_size_m: float = 30.0
    timestep_minutes: float = 1.0
    wind_speed_ms: float = 3.0
    wind_direction_deg: float = 270.0
    slope_degrees: float = 5.0
    aspect_degrees: float = 180.0
    moisture_dead: float = 0.08
    moisture_live: float = 0.9
    wind_reduction_factor: float = 0.4
    fuel_moisture_adjustment: float = 1.0
    crown_fire: bool = False


@dataclass
class SpatialParameters:
    ignition_density: float = 0.9  # proportion of trees initially
    empty_density: float = 0.1
    variance: float = 4.0
    wind_bias: Dict[str, float] = field(default_factory=lambda: {
        'N': 1.0,
        'NE': 1.0,
        'E': 1.0,
        'SE': 1.0,
        'S': 1.0,
        'SW': 1.0,
        'W': 1.0,
        'NW': 1.0,
    })
    physics: PhysicsParameters = field(default_factory=PhysicsParameters)


@dataclass
class EnvironmentParameters:
    slope: float = 0.1
    vegetation_density: float = 0.9
    wind_vector: List[float] = field(default_factory=lambda: [1.0, 0.5])
    moisture: float = 0.2
    fuel_type: str = 'grass'
    humidity: float = 0.7
    temperature: float = 25.0
    natural_barriers: float = 0.1
    weather_conditions: float = 0.1
    base_budget: float = 100.0


@dataclass
class SimulationParameters:
    grid_size: int
    time_steps: int
    temporal: TemporalParameters
    spatial: SpatialParameters
    environment: EnvironmentParameters
    resources: List[Dict]
    stochastic_seed: Optional[int] = None


DEFAULT_RESOURCES = [
    {'type': 'firebreak', 'strength': 0.05},
    {'type': 'thinning', 'strength': 0.02},
]


def build_parameters(config_payload: Dict) -> SimulationParameters:
    temporal_raw = config_payload.get('temporal_parameters', {})
    spatial_raw = config_payload.get('spatial_parameters', {})
    environment_raw = config_payload.get('environment_parameters', {})
    resources = config_payload.get('resource_parameters') or DEFAULT_RESOURCES

    temporal = TemporalParameters(
        spread_rate=temporal_raw.get('spread_rate', 0.05),
        extinguish_rate=temporal_raw.get('extinguish_rate', 0.05),
        firefighting_rate=temporal_raw.get('firefighting_rate', 0.02),
        initial_population=temporal_raw.get('initial_population', 10),
    )

    physics_raw = spatial_raw.get('physics', {})
    physics = PhysicsParameters(
        enabled=physics_raw.get('enabled', False),
        fuel_model=physics_raw.get('fuel_model', 'GR1'),
        cell_size_m=physics_raw.get('cell_size_m', 30.0),
        timestep_minutes=physics_raw.get('timestep_minutes', 1.0),
        wind_speed_ms=physics_raw.get('wind_speed_ms', 3.0),
        wind_direction_deg=physics_raw.get('wind_direction_deg', 270.0),
        slope_degrees=physics_raw.get('slope_degrees', environment_raw.get('slope', 0.1) * 100 if environment_raw.get('slope', 0.1) < 1 else environment_raw.get('slope', 0.1)),
        aspect_degrees=physics_raw.get('aspect_degrees', 180.0),
        moisture_dead=physics_raw.get('moisture_dead', environment_raw.get('moisture', 0.2)),
        moisture_live=physics_raw.get('moisture_live', environment_raw.get('moisture', 0.2) * 5),
        wind_reduction_factor=physics_raw.get('wind_reduction_factor', 0.4),
        fuel_moisture_adjustment=physics_raw.get('fuel_moisture_adjustment', 1.0),
        crown_fire=physics_raw.get('crown_fire', False),
    )

    spatial = SpatialParameters(
        ignition_density=spatial_raw.get('ignition_density', 0.9),
        empty_density=spatial_raw.get('empty_density', 0.1),
        variance=spatial_raw.get('variance', 4.0),
        wind_bias=spatial_raw.get('wind_bias', SpatialParameters().wind_bias),
        physics=physics,
    )

    environment = EnvironmentParameters(
        slope=environment_raw.get('slope', 0.1),
        vegetation_density=environment_raw.get('vegetation_density', 0.9),
        wind_vector=environment_raw.get('wind_vector', [1.0, 0.5]),
        moisture=environment_raw.get('moisture', 0.2),
        fuel_type=environment_raw.get('fuel_type', 'grass'),
        humidity=environment_raw.get('humidity', 0.7),
        temperature=environment_raw.get('temperature', 25.0),
        natural_barriers=environment_raw.get('natural_barriers', 0.1),
        weather_conditions=environment_raw.get('weather_conditions', 0.1),
        base_budget=environment_raw.get('base_budget', 100.0),
    )

    params = SimulationParameters(
        grid_size=config_payload.get('grid_size'),
        time_steps=config_payload.get('time_steps'),
        temporal=temporal,
        spatial=spatial,
        environment=environment,
        resources=resources,
        stochastic_seed=config_payload.get('stochastic_seed'),
    )

    return params
