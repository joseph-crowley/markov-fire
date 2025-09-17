from __future__ import annotations

import enum
import io
import math
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np

from .parameters import (
    SimulationParameters,
    TemporalParameters,
    EnvironmentParameters,
    PhysicsParameters,
)
from .ros import (
    RateOfSpreadRequest,
    RothermelCalculator,
    build_ros_probability,
)


class GridState(enum.IntEnum):
    EMPTY = 0
    TREE = 1
    ON_FIRE = 2
    BURNED = 3
    PREVIOUSLY_BURNED = 4
    FIREBREAK = 5
    REMOVED = 6
    SUPPRESSED = 7
    PREVIOUSLY_SUPPRESSED = 8


@dataclass
class TickResult:
    index: int
    active_cells: int
    burned_cells: int
    suppressed_cells: int
    footprint: int
    population: int
    extinguished: bool
    grid: np.ndarray
    spread: int
    extinguish: int
    suppress: int

    def serialize_grid(self) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, self.grid.astype(np.uint8), allow_pickle=False)
        return buffer.getvalue()

    def as_message(self) -> Dict:
        return {
            'tick': self.index,
            'activeCells': int(self.active_cells),
            'burnedCells': int(self.burned_cells),
            'suppressedCells': int(self.suppressed_cells),
            'footprint': int(self.footprint),
            'population': int(self.population),
            'extinguished': bool(self.extinguished),
            'spread': int(self.spread),
            'extinguish': int(self.extinguish),
            'suppress': int(self.suppress),
            'grid': self.grid.tolist(),
        }


class EnvironmentModel:
    def __init__(self, params: EnvironmentParameters):
        self.params = params
        self.budget = params.base_budget
        self.wind_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    def get_wind_direction(self, rng: np.random.Generator, bias: Dict[str, float]) -> str:
        weights = np.array([bias.get(d, 1.0) for d in self.wind_directions], dtype=float)
        weights = weights / weights.sum()
        return rng.choice(self.wind_directions, p=weights)

    def fuel_factor(self, vegetation_density: float, moisture: float, fuel_type: str) -> float:
        moisture_factor = max(0.0, 1.0 - (moisture / 100.0 if moisture > 1 else moisture))
        density_factor = vegetation_density
        fuel_type_factor = {
            'grass': 0.6,
            'brush': 0.8,
            'timber': 1.0,
        }.get(fuel_type, 0.5)
        return float(np.clip(moisture_factor * density_factor * fuel_type_factor, 0.0, 1.0))

    def atmospheric_factor(self, humidity: float, temperature: float) -> float:
        humidity_factor = max(0.0, 1.0 - humidity / 100.0 if humidity > 1 else 1.0 - humidity)
        temperature_factor = (temperature - 20) / 30.0 if temperature > 20 else 0.0
        return float(np.clip(humidity_factor + temperature_factor, 0.0, 1.0))

    def calculate_spread_rate(self, fire_proximity_value: float) -> float:
        slope, vegetation_density = self.params.slope, self.params.vegetation_density
        wind_vector = np.array(self.params.wind_vector)
        moisture = self.params.moisture * 100 if self.params.moisture <= 1 else self.params.moisture
        fuel_factor = self.fuel_factor(vegetation_density, moisture, self.params.fuel_type)
        atmospheric_factor = self.atmospheric_factor(self.params.humidity * 100 if self.params.humidity <= 1 else self.params.humidity, self.params.temperature)
        directional_speed = float(np.dot(wind_vector, np.array([math.sin(slope), math.cos(slope)])))
        return float((directional_speed + 1) * fuel_factor * atmospheric_factor * (1 + fire_proximity_value))

    def environment_factor(self, natural_barriers: float, weather_conditions: float) -> float:
        return float(np.clip(natural_barriers + weather_conditions, 0.0, 1.0))

    def calculate_extinguish_rate(self, fire_proximity_value: float) -> float:
        return self.environment_factor(self.params.natural_barriers, self.params.weather_conditions) * (1 - fire_proximity_value)

    def calculate_firefighting_rate(self, current_phase: str = 'initial', mobility: float = 1.0, potency: float = 1.0, cost: float = 1.0) -> float:
        phase_factor = {
            'initial': 1.0,
            'developed': 0.8,
            'controlled': 0.5,
        }.get(current_phase, 0.5)
        resource_factor = mobility * 0.4 + potency * 0.4 + cost * 0.2
        base_rate = min(phase_factor * resource_factor, 1.0)
        if self.budget <= 0:
            return 0.0
        applied = min(base_rate, self.budget)
        self.budget -= applied
        return applied


class WildfireSpreadProcess:
    def __init__(self, params: TemporalParameters, system_size: int, rng: np.random.Generator):
        self.params = params
        self.system_size = system_size
        self.rng = rng

    def simulate_step(self, current_population: int, spread_multiplier: float = 1.0) -> Tuple[int, int, int]:
        spread_rate = max(self.params.spread_rate * spread_multiplier, 0.0)
        spread = self.rng.poisson(spread_rate * current_population)
        extinguish = self.rng.poisson(self.params.extinguish_rate * current_population)
        suppression = self.rng.poisson(self.params.firefighting_rate * current_population)
        return int(spread), int(extinguish), int(suppression)


class WildfireSimulator:
    def __init__(self, params: SimulationParameters):
        self.params = params
        self.rng = np.random.default_rng(params.stochastic_seed)
        self.environment = EnvironmentModel(params.environment)
        self.physics: PhysicsParameters = params.spatial.physics
        self.ros_calculator: Optional[RothermelCalculator] = (
            RothermelCalculator() if self.physics.enabled else None
        )
        self._last_ros_result = None
        self._last_ros_distribution: Optional[Dict[str, float]] = None
        self.temporal = WildfireSpreadProcess(params.temporal, params.grid_size ** 2, self.rng)
        self.grid = self._initial_grid()
        self.population = params.temporal.initial_population
        self.footprint = params.temporal.initial_population
        self.extinguishment_time: Optional[int] = None
        self._seed_initial_fire()

    def _initial_grid(self) -> np.ndarray:
        size = self.params.grid_size
        tree_probability = self.params.spatial.ignition_density
        probabilities = np.array([self.params.spatial.empty_density, tree_probability], dtype=float)
        total = probabilities.sum()
        if total <= 0:
            probabilities = np.array([0.5, 0.5])
        else:
            probabilities = probabilities / total
        grid = self.rng.choice(
            [GridState.EMPTY.value, GridState.TREE.value],
            size=(size, size),
            p=probabilities,
        )
        return grid.astype(np.uint8)

    def _seed_initial_fire(self) -> None:
        indices = np.argwhere(self.grid == GridState.TREE.value)
        if len(indices) == 0:
            return
        center_idx = self.rng.choice(len(indices))
        center = indices[center_idx]
        radius = max(1, int(math.sqrt(self.params.temporal.initial_population)))
        count = 0
        for _ in range(self.params.temporal.initial_population * 3):
            offset = self.rng.integers(-radius, radius + 1, size=2)
            cell = tuple(np.clip(center + offset, 0, self.params.grid_size - 1))
            if self.grid[cell] == GridState.TREE.value:
                self.grid[cell] = GridState.ON_FIRE.value
                count += 1
            if count >= self.params.temporal.initial_population:
                break
        self.population = int(np.sum(self.grid == GridState.ON_FIRE.value))
        self.footprint = int(np.sum((self.grid == GridState.ON_FIRE.value) | (self.grid == GridState.BURNED.value)))

    def _fire_cells(self) -> np.ndarray:
        return np.argwhere(self.grid == GridState.ON_FIRE.value)

    def _tree_cells(self) -> np.ndarray:
        return np.argwhere(self.grid == GridState.TREE.value)

    def _fire_proximity(self, variance: float) -> np.ndarray:
        fire_coords = self._fire_cells()
        size = self.grid.shape[0]
        proximity = np.zeros((size, size), dtype=float)
        if len(fire_coords) == 0:
            proximity.fill(1.0 / (size * size))
            return proximity
        for cell in fire_coords:
            distances = np.linalg.norm(fire_coords - cell, axis=1)
            weight = np.exp(-(distances ** 2) / variance)
            proximity[cell[0], cell[1]] += weight.sum()
        total = proximity.sum()
        if total == 0:
            proximity += 1.0
            total = proximity.sum()
        return proximity / total

    @staticmethod
    def _cardinal_to_degrees(cardinal: str) -> float:
        mapping = {
            'N': 0.0,
            'NE': 45.0,
            'E': 90.0,
            'SE': 135.0,
            'S': 180.0,
            'SW': 225.0,
            'W': 270.0,
            'NW': 315.0,
        }
        return mapping.get(cardinal, 0.0)

    def _angle_to_cardinal(self, angle_deg: float) -> str:
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        index = int(((angle_deg + 22.5) % 360.0) / 45.0)
        return directions[index]

    def _direction_from_center(self, center: np.ndarray, cell: Tuple[int, int]) -> str:
        dy = cell[0] - center[0]
        dx = cell[1] - center[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 'N'
        angle = math.degrees(math.atan2(-dy, dx)) % 360.0
        angle = (90.0 - angle) % 360.0
        return self._angle_to_cardinal(angle)

    def _compute_ros_context(
        self,
        wind_direction: str,
        current_population: int,
    ) -> Tuple[Optional[Dict[str, float]], float]:
        if not self.ros_calculator:
            return None, 1.0

        physics = self.physics
        env = self.environment.params

        wind_speed = physics.wind_speed_ms if physics.wind_speed_ms > 0 else float(np.linalg.norm(env.wind_vector))
        wind_dir_deg = physics.wind_direction_deg if wind_direction is None else self._cardinal_to_degrees(wind_direction)

        request = RateOfSpreadRequest(
            fuel_model=physics.fuel_model,
            wind_speed_ms=float(wind_speed),
            wind_direction_deg=float(wind_dir_deg),
            slope_degrees=float(physics.slope_degrees),
            aspect_degrees=float(physics.aspect_degrees),
            moisture_dead=float(physics.moisture_dead),
            moisture_live=float(max(physics.moisture_live, physics.moisture_dead)),
            air_temperature_c=float(env.temperature),
            wind_reduction_factor=float(physics.wind_reduction_factor),
            fuel_moisture_adjustment=float(physics.fuel_moisture_adjustment),
            crown_fire=self._crown_fire_active(current_population),
        )

        result, distribution = build_ros_probability(
            self.ros_calculator,
            request,
            cell_size_m=max(float(physics.cell_size_m), 1.0),
            timestep_minutes=max(float(physics.timestep_minutes), 0.1),
        )

        self._last_ros_result = result
        self._last_ros_distribution = distribution

        spread_multiplier = self.ros_calculator.spread_multiplier(
            request,
            baseline_rate=self.params.temporal.spread_rate,
        )

        return distribution, spread_multiplier

    def _crown_fire_active(self, current_population: int) -> bool:
        if self.physics.crown_fire:
            return True
        threshold = int(self.params.grid_size ** 2 * 0.15)
        return current_population >= max(threshold, 1)

    def _apply_resources(self, proximity: np.ndarray) -> None:
        for resource in self.params.resources:
            r_type = resource.get('type')
            strength = float(resource.get('strength', 0))
            if strength <= 0:
                continue
            if r_type == 'firebreak':
                burned = np.argwhere(self.grid == GridState.PREVIOUSLY_BURNED.value)
                for (i, j) in burned:
                    neighbors = self._neighbors(i, j)
                    for ni, nj in neighbors:
                        if self.grid[ni, nj] == GridState.TREE.value and self.rng.random() < strength:
                            self.grid[ni, nj] = GridState.FIREBREAK.value
            elif r_type == 'thinning':
                trees = np.argwhere(self.grid == GridState.TREE.value)
                if len(trees) == 0:
                    continue
                sample_size = max(1, int(len(trees) * min(strength, 1.0)))
                choices = self.rng.choice(len(trees), size=sample_size, replace=False)
                for idx in choices:
                    i, j = trees[idx]
                    self.grid[i, j] = GridState.REMOVED.value
            elif r_type == 'suppression':
                fire_cells = self._fire_cells()
                if len(fire_cells) == 0:
                    continue
                sample_size = max(1, int(len(fire_cells) * min(strength, 1.0)))
                indices = self.rng.choice(len(fire_cells), size=sample_size, replace=False)
                for idx in indices:
                    cell = tuple(fire_cells[idx])
                    if proximity[cell] > 0.5:
                        self.grid[cell] = GridState.SUPPRESSED.value

    def _neighbors(self, i: int, j: int) -> Iterable[Tuple[int, int]]:
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < self.params.grid_size and 0 <= nj < self.params.grid_size:
                yield ni, nj

    def _spread_fire(
        self,
        new_fire_cells: int,
        proximity: np.ndarray,
        wind_direction: str,
        directional_bias: Optional[Dict[str, float]] = None,
        fire_cells: Optional[np.ndarray] = None,
    ) -> int:
        trees = self._tree_cells()
        if len(trees) == 0 or new_fire_cells <= 0:
            return 0
        center = None
        if directional_bias and fire_cells is not None and len(fire_cells) > 0:
            center = np.mean(fire_cells, axis=0)
        weights = []
        for i, j in trees:
            score = proximity[i, j]
            if directional_bias and center is not None:
                cardinal = self._direction_from_center(center, (i, j))
                score *= directional_bias.get(cardinal, 1.0)
            elif self._aligned_with_wind((i, j), wind_direction):
                score *= 1.5
            weights.append(score)
        weights = np.array(weights, dtype=float)
        if weights.sum() == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights.sum()
        sample_size = min(new_fire_cells, len(trees))
        selections = self.rng.choice(len(trees), size=sample_size, replace=False, p=weights)
        for idx in selections:
            i, j = trees[idx]
            self.grid[i, j] = GridState.ON_FIRE.value
        return int(sample_size)

    def _aligned_with_wind(self, cell: Tuple[int, int], wind_direction: str) -> bool:
        if len(self._fire_cells()) == 0:
            return False
        center = np.mean(self._fire_cells(), axis=0)
        direction_vector = np.array(cell) - center
        if np.all(direction_vector == 0):
            return True
        mapping = {
            'N': np.array([-1, 0]),
            'NE': np.array([-1, 1]),
            'E': np.array([0, 1]),
            'SE': np.array([1, 1]),
            'S': np.array([1, 0]),
            'SW': np.array([1, -1]),
            'W': np.array([0, -1]),
            'NW': np.array([-1, -1]),
        }
        target = mapping.get(wind_direction, np.array([0, 0]))
        if np.linalg.norm(target) == 0:
            return False
        cos_sim = np.dot(direction_vector, target) / (
            np.linalg.norm(direction_vector) * np.linalg.norm(target)
        )
        return cos_sim > 0.5

    def _update_fire_cells(self, extinguishments: List[Tuple[int, int]], suppressions: List[Tuple[int, int]]) -> None:
        for i, j in extinguishments:
            self.grid[i, j] = GridState.PREVIOUSLY_BURNED.value
        for i, j in suppressions:
            self.grid[i, j] = GridState.SUPPRESSED.value

    def _progress_existing_fire(self) -> None:
        rows, cols = np.where(self.grid == GridState.BURNED.value)
        for i, j in zip(rows, cols):
            self.grid[i, j] = GridState.PREVIOUSLY_BURNED.value
        rows, cols = np.where(self.grid == GridState.SUPPRESSED.value)
        for i, j in zip(rows, cols):
            self.grid[i, j] = GridState.PREVIOUSLY_SUPPRESSED.value
        rows, cols = np.where(self.grid == GridState.ON_FIRE.value)
        for i, j in zip(rows, cols):
            self.grid[i, j] = GridState.BURNED.value

    def run(self) -> Generator[TickResult, None, None]:
        for tick in range(self.params.time_steps):
            active_cells = int(np.sum(self.grid == GridState.ON_FIRE.value))
            current_population = active_cells

            proximity = self._fire_proximity(self.params.spatial.variance)
            wind_direction = self.environment.get_wind_direction(self.rng, self.params.spatial.wind_bias)
            directional_bias, spread_multiplier = self._compute_ros_context(wind_direction, active_cells)

            spread, extinguish, suppress = self.temporal.simulate_step(
                current_population,
                spread_multiplier=spread_multiplier,
            )

            self._apply_resources(proximity)

            on_fire_cells = np.argwhere(self.grid == GridState.ON_FIRE.value)
            extinguish_count = min(extinguish, len(on_fire_cells))
            suppress_count = min(suppress, len(on_fire_cells) - extinguish_count)
            selections = (
                self.rng.choice(len(on_fire_cells), size=extinguish_count + suppress_count, replace=False)
                if len(on_fire_cells) > 0 and (extinguish_count + suppress_count) > 0
                else []
            )
            extinguishments: List[Tuple[int, int]] = []
            suppressions: List[Tuple[int, int]] = []
            if len(on_fire_cells) > 0 and len(selections) > 0:
                extinguish_indices = selections[:extinguish_count]
                suppress_indices = selections[extinguish_count:extinguish_count + suppress_count]
                extinguishments = [tuple(on_fire_cells[i]) for i in extinguish_indices]
                suppressions = [tuple(on_fire_cells[i]) for i in suppress_indices]

            self._progress_existing_fire()
            self._update_fire_cells(extinguishments, suppressions)

            added = self._spread_fire(
                spread,
                proximity,
                wind_direction,
                directional_bias=directional_bias,
                fire_cells=on_fire_cells,
            )

            active_cells = int(np.sum(self.grid == GridState.ON_FIRE.value))
            burned_cells = int(np.sum((self.grid == GridState.BURNED.value) | (self.grid == GridState.PREVIOUSLY_BURNED.value)))
            suppressed_cells = int(np.sum((self.grid == GridState.SUPPRESSED.value) | (self.grid == GridState.PREVIOUSLY_SUPPRESSED.value)))
            footprint = burned_cells + active_cells + suppressed_cells

            if active_cells == 0 and self.extinguishment_time is None:
                self.extinguishment_time = tick

            yield TickResult(
                index=tick,
                active_cells=active_cells,
                burned_cells=burned_cells,
                suppressed_cells=suppressed_cells,
                footprint=footprint,
                population=active_cells,
                extinguished=active_cells == 0,
                grid=self.grid.copy(),
                spread=added,
                extinguish=len(extinguishments),
                suppress=len(suppressions),
            )

            if active_cells == 0:
                break
