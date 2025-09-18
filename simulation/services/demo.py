from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from django.db import transaction
from django.utils import timezone

from simulation.models import (
    SimulationConfig,
    Scenario,
    ScenarioVersion,
    ScenarioTag,
    SimulationRun,
    SimulationTick,
    SimulationAnalytics,
    Checkpoint,
)
from simulation.services.engine import GridState


@dataclass(frozen=True)
class DemoResult:
    scenario: Scenario
    version: ScenarioVersion
    run: SimulationRun


@transaction.atomic
def generate_demo_run(reset: bool = False) -> DemoResult:
    demo_slug = 'demo-fire-corridor'
    config_slug = f'{demo_slug}-config'

    tag, _ = ScenarioTag.objects.get_or_create(
        slug='demo', defaults={'name': 'Demo', 'description': 'Demo-ready scenarios'}
    )

    config_defaults = {
        'name': 'Demo Fire Corridor',
        'description': 'Wind-aligned canyon corridor designed for long burn visualisations.',
        'grid_size': 140,
        'time_steps': 480,
        'stochastic_seed': 5150,
        'temporal_parameters': {
            'spread_rate': 0.12,
            'extinguish_rate': 0.01,
            'firefighting_rate': 0.002,
            'initial_population': 24,
        },
        'spatial_parameters': {
            'ignition_density': 0.92,
            'empty_density': 0.08,
            'variance': 6.5,
            'wind_bias': {
                'N': 0.85,
                'NE': 1.0,
                'E': 1.2,
                'SE': 1.35,
                'S': 1.25,
                'SW': 1.05,
                'W': 0.9,
                'NW': 0.85,
            },
            'physics': {
                'enabled': False,
            },
        },
        'environment_parameters': {
            'slope': 0.18,
            'vegetation_density': 0.94,
            'wind_vector': [1.4, 1.1],
            'moisture': 0.16,
            'fuel_type': 'brush',
            'humidity': 0.38,
            'temperature': 31,
            'natural_barriers': 0.1,
            'weather_conditions': 0.2,
            'base_budget': 80,
        },
        'resource_parameters': [],
    }

    config, _ = SimulationConfig.objects.update_or_create(slug=config_slug, defaults=config_defaults)

    scenario_defaults = {
        'name': config_defaults['name'],
        'description': config_defaults['description'],
        'base_config': config,
        'metadata': {'theme': 'demo-corridor', 'purpose': 'visualisation'},
        'is_active': True,
    }

    scenario, _ = Scenario.objects.update_or_create(slug=demo_slug, defaults=scenario_defaults)
    scenario.tags.add(tag)

    if reset:
        ScenarioVersion.objects.filter(scenario=scenario).delete()
        scenario.active_version = None
        scenario.save(update_fields=['active_version', 'updated_at'])

    if scenario.versions.exists():
        version = scenario.active_version or scenario.versions.order_by('-version').first()
        version.config = config
        version.metadata = scenario_defaults['metadata']
        version.label = 'Demo Corridor'
        version.notes = 'Auto-generated for front-end demos.'
        version.save(update_fields=['config', 'metadata', 'label', 'notes', 'updated_at'])
    else:
        version = ScenarioVersion.objects.create(
            scenario=scenario,
            config=config,
            label='Demo Corridor',
            notes='Auto-generated for front-end demos.',
            metadata=scenario_defaults['metadata'],
        )

    if scenario.active_version_id is None:
        scenario.active_version = version
        scenario.save(update_fields=['active_version', 'updated_at'])

    if reset:
        SimulationRun.objects.filter(config=config).delete()

    run = SimulationRun.objects.create(
        config=config,
        scenario_version=version,
        seed=config.stochastic_seed,
        status=SimulationRun.Status.RUNNING,
        started_at=timezone.now(),
    )

    size = config.grid_size
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size * 0.55, size * 0.35
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    burned_mask = np.zeros((size, size), dtype=bool)
    suppressed_mask = np.zeros((size, size), dtype=bool)

    total_ticks = 200
    tick_records = []
    max_active = 0
    footprint = 0
    burned_cells = 0
    suppressed_cells = 0

    active_series: list[Tuple[int, int]] = []
    burned_series: list[Tuple[int, int]] = []
    suppressed_series: list[Tuple[int, int]] = []
    footprint_series: list[Tuple[int, int]] = []

    checkpoints = []

    for tick in range(total_ticks):
        outer_radius = min(size / 2, 8 + tick * 0.45)
        inner_radius = max(0, outer_radius - (4 + (tick % 5)))

        new_burned = (dist <= inner_radius)
        burned_mask |= new_burned

        on_fire = (dist > inner_radius) & (dist <= outer_radius)
        on_fire &= ~burned_mask

        suppressed_band = (dist > inner_radius - 5) & (dist <= inner_radius - 2)
        suppressed_cycle = (tick // 8) % 3
        if suppressed_cycle == 1:
            suppressed_band &= (x % 3 == 0)
        elif suppressed_cycle == 2:
            suppressed_band &= (y % 3 == 0)
        suppressed_mask = (suppressed_mask | suppressed_band) & ~burned_mask

        grid = np.full((size, size), GridState.TREE.value, dtype=np.uint8)
        grid[burned_mask] = GridState.BURNED.value
        grid[suppressed_mask] = GridState.SUPPRESSED.value
        grid[on_fire] = GridState.ON_FIRE.value

        active_cells = int(on_fire.sum())
        burned_cells = int(burned_mask.sum())
        suppressed_cells = int(suppressed_mask.sum())
        footprint = active_cells + burned_cells + suppressed_cells

        active_series.append((tick, active_cells))
        burned_series.append((tick, burned_cells))
        suppressed_series.append((tick, suppressed_cells))
        footprint_series.append((tick, footprint))

        buffer = io.BytesIO()
        np.save(buffer, grid, allow_pickle=False)
        payload = buffer.getvalue()

        tick_records.append(
            SimulationTick(
                run=run,
                tick_index=tick,
                active_cells=active_cells,
                burned_cells=burned_cells,
                suppressed_cells=suppressed_cells,
                footprint=footprint,
                grid_payload=payload,
            )
        )

        max_active = max(max_active, active_cells)

        if tick % 40 == 0:
            checkpoints.append(
                Checkpoint(
                    run=run,
                    tick_index=tick,
                    payload=pickle.dumps({'tick': tick, 'grid_payload': payload}),
                    metadata={'active_cells': active_cells, 'burned_cells': burned_cells},
                )
            )

    SimulationTick.objects.bulk_create(tick_records)
    Checkpoint.objects.bulk_create(checkpoints)

    run.status = SimulationRun.Status.COMPLETED
    run.finished_at = timezone.now()
    run.total_ticks = total_ticks
    run.extinguishment_step = total_ticks - 1
    run.max_active_cells = max_active
    run.save(update_fields=['status', 'finished_at', 'total_ticks', 'extinguishment_step', 'max_active_cells'])

    SimulationAnalytics.objects.update_or_create(
        run=run,
        defaults={
            'summary': {
                'total_ticks': total_ticks,
                'max_active_cells': max_active,
                'final_footprint': footprint,
                'final_burned_cells': burned_cells,
                'final_suppressed_cells': suppressed_cells,
            },
            'distributions': {
                'active_cells': active_series,
                'burned_cells': burned_series,
                'suppressed_cells': suppressed_series,
                'footprint': footprint_series,
            },
            'computed_at': timezone.now(),
        },
    )

    return DemoResult(scenario=scenario, version=version, run=run)
