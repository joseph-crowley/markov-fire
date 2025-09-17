import io

import numpy as np

from simulation.services.engine import WildfireSimulator
from simulation.services.parameters import SimulationParameters, TemporalParameters, SpatialParameters, EnvironmentParameters, DEFAULT_RESOURCES


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
