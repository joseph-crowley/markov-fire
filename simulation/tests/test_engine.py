import io

import numpy as np

from simulation.services.engine import WildfireSimulator
from simulation.services.parameters import (
    SimulationParameters,
    TemporalParameters,
    SpatialParameters,
    EnvironmentParameters,
    PhysicsParameters,
    DEFAULT_RESOURCES,
)


def build_params():
    return SimulationParameters(
        grid_size=20,
        time_steps=10,
        temporal=TemporalParameters(spread_rate=0.05, extinguish_rate=0.02, firefighting_rate=0.01, initial_population=5),
        spatial=SpatialParameters(),
        environment=EnvironmentParameters(),
        resources=DEFAULT_RESOURCES,
        stochastic_seed=123,
    )


def test_simulator_runs_until_extinguish():
    params = build_params()
    simulator = WildfireSimulator(params)
    ticks = list(simulator.run())
    assert len(ticks) > 0
    assert all(tick.grid.shape == (params.grid_size, params.grid_size) for tick in ticks)
    assert ticks[0].index == 0
    assert ticks[-1].index <= params.time_steps


def test_grid_serialization_round_trip():
    params = build_params()
    simulator = WildfireSimulator(params)
    tick = next(simulator.run())
    payload = tick.serialize_grid()
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    assert array.shape == (params.grid_size, params.grid_size)


def test_ros_distribution_populated_when_physics_enabled():
    params = build_params()
    params.spatial.physics = PhysicsParameters(
        enabled=True,
        fuel_model='GR4',
        wind_speed_ms=6.0,
        wind_direction_deg=270.0,
        slope_degrees=5.0,
        aspect_degrees=225.0,
        moisture_dead=0.06,
        moisture_live=0.9,
        timestep_minutes=1.0,
        cell_size_m=30.0,
    )
    simulator = WildfireSimulator(params)
    next(simulator.run())
    assert simulator._last_ros_distribution is not None
    assert set(simulator._last_ros_distribution.keys()) == {'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'}
    total_probability = sum(simulator._last_ros_distribution.values())
    assert 0.0 < total_probability <= 1.0


def test_ros_distribution_absent_when_physics_disabled():
    params = build_params()
    simulator = WildfireSimulator(params)
    next(simulator.run())
    assert simulator._last_ros_distribution is None
