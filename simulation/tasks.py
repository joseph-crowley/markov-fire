import io
import logging
import pickle
from typing import Dict, Optional

from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer
from django.utils import timezone
import numpy as np

from .models import (
    SimulationRun,
    SimulationTick,
    LiveMetric,
    SimulationAnalytics,
    ValidationResult,
    Checkpoint,
)
from .services.engine import WildfireSimulator
from .services.parameters import build_parameters, DEFAULT_RESOURCES
from .services.ros import RateOfSpreadResult

logger = logging.getLogger(__name__)

CHECKPOINT_INTERVAL = 25


def _merge_presets(config) -> Dict:
    resource_parameters = config.resource_parameters or []
    environment_parameters = config.environment_parameters or {}

    if config.resource_preset:
        preset_resources = config.resource_preset.resources or []
        resource_parameters = preset_resources + resource_parameters
    if config.environment_preset:
        preset_env = config.environment_preset.parameters or {}
        merged = preset_env.copy()
        merged.update(environment_parameters)
        environment_parameters = merged

    payload = {
        'grid_size': config.grid_size,
        'time_steps': config.time_steps,
        'temporal_parameters': config.temporal_parameters,
        'spatial_parameters': config.spatial_parameters,
        'environment_parameters': environment_parameters,
        'resource_parameters': resource_parameters or DEFAULT_RESOURCES,
        'stochastic_seed': config.stochastic_seed,
    }
    return payload


@shared_task(bind=True)
def run_simulation_task(self, run_id: str) -> None:
    logger.info('Starting simulation task for run %s', run_id)
    try:
        run = SimulationRun.objects.select_related('config', 'config__resource_preset', 'config__environment_preset').get(pk=run_id)
    except SimulationRun.DoesNotExist:
        logger.error('SimulationRun %s does not exist', run_id)
        return

    if run.status not in {SimulationRun.Status.PENDING, SimulationRun.Status.FAILED}:
        logger.info('SimulationRun %s has status %s; skipping', run_id, run.status)
        return

    run.status = SimulationRun.Status.RUNNING
    run.started_at = timezone.now()
    run.celery_task_id = self.request.id
    run.save(update_fields=['status', 'started_at', 'celery_task_id'])

    payload = _merge_presets(run.config)
    payload['stochastic_seed'] = run.seed or payload.get('stochastic_seed')

    try:
        params = build_parameters(payload)
        simulator = WildfireSimulator(params)

        if run.resume_from_id:
            _load_checkpoint_state(simulator, run.resume_from)
        channel_layer = get_channel_layer()
        tick_records = []
        max_active = 0
        extinguishment_step = None
        last_tick_index = -1

        active_series = []
        burned_series = []
        suppressed_series = []
        footprint_series = []
        spread_series = []
        extinguish_series = []
        suppress_series = []

        for tick in simulator.run():
            last_tick_index = tick.index
            max_active = max(max_active, tick.active_cells)
            if tick.extinguished and extinguishment_step is None:
                extinguishment_step = tick.index

            active_series.append((tick.index, tick.active_cells))
            burned_series.append((tick.index, tick.burned_cells))
            suppressed_series.append((tick.index, tick.suppressed_cells))
            footprint_series.append((tick.index, tick.footprint))
            spread_series.append((tick.index, tick.spread))
            extinguish_series.append((tick.index, tick.extinguish))
            suppress_series.append((tick.index, tick.suppress))

            tick_records.append(
                SimulationTick(
                    run=run,
                    tick_index=tick.index,
                    active_cells=tick.active_cells,
                    burned_cells=tick.burned_cells,
                    suppressed_cells=tick.suppressed_cells,
                    footprint=tick.footprint,
                    grid_payload=tick.serialize_grid(),
                )
            )

            async_to_sync(channel_layer.group_send)(
                f'simulation_{run.id}',
                {
                    'type': 'simulation.tick',
                    'message': tick.as_message(),
                }
            )

            if len(tick_records) >= 25:
                SimulationTick.objects.bulk_create(tick_records)
                tick_records.clear()

                LiveMetric.objects.update_or_create(
                    run=run,
                    defaults={'metrics': {
                        'max_active_cells': max_active,
                        'last_tick': tick.index,
                        'extinguishment_step': extinguishment_step,
                    }}
                )

            if CHECKPOINT_INTERVAL and (tick.index + 1) % CHECKPOINT_INTERVAL == 0:
                _persist_checkpoint(run, simulator, tick.index, max_active, extinguishment_step)

        if tick_records:
            SimulationTick.objects.bulk_create(tick_records)

        LiveMetric.objects.update_or_create(
            run=run,
            defaults={'metrics': {
                'max_active_cells': max_active,
                'extinguishment_step': extinguishment_step,
                'total_ticks': last_tick_index + 1,
            }}
        )

        run.status = SimulationRun.Status.COMPLETED
        run.finished_at = timezone.now()
        run.total_ticks = last_tick_index + 1
        run.extinguishment_step = extinguishment_step
        run.max_active_cells = max_active
        run.save(update_fields=['status', 'finished_at', 'total_ticks', 'extinguishment_step', 'max_active_cells'])

        summary = {
            'total_ticks': run.total_ticks,
            'extinguishment_step': extinguishment_step,
            'max_active_cells': max_active,
            'final_active_cells': active_series[-1][1] if active_series else 0,
            'final_burned_cells': burned_series[-1][1] if burned_series else 0,
            'final_suppressed_cells': suppressed_series[-1][1] if suppressed_series else 0,
            'final_footprint': footprint_series[-1][1] if footprint_series else 0,
        }
        distributions = {
            'active_cells': active_series,
            'burned_cells': burned_series,
            'suppressed_cells': suppressed_series,
            'footprint': footprint_series,
            'spread_events': spread_series,
            'extinguish_events': extinguish_series,
            'suppress_events': suppress_series,
        }

        SimulationAnalytics.objects.update_or_create(
            run=run,
            defaults={
                'summary': summary,
                'distributions': distributions,
                'computed_at': timezone.now(),
            },
        )

        if run.scenario_version_id:
            ValidationResult.objects.update_or_create(
                run=run,
                scenario_version=run.scenario_version,
                defaults={
                    'status': ValidationResult.Status.COMPLETED,
                    'metrics': {
                        'final_footprint': summary['final_footprint'],
                        'max_active_cells': max_active,
                        'extinguishment_step': extinguishment_step,
                    },
                },
            )

        _persist_checkpoint(run, simulator, last_tick_index, max_active, extinguishment_step, final=True)

        async_to_sync(channel_layer.group_send)(
            f'simulation_{run.id}',
            {
                'type': 'simulation.completed',
                'message': {
                    'status': 'completed',
                    'totalTicks': run.total_ticks,
                    'extinguishmentStep': extinguishment_step,
                    'maxActiveCells': max_active,
                },
            }
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception('Simulation task failed: %s', exc)
        run.status = SimulationRun.Status.FAILED
        run.finished_at = timezone.now()
        run.notes = f'Task failed: {exc}'
        run.save(update_fields=['status', 'finished_at', 'notes'])

        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f'simulation_{run.id}',
            {
                'type': 'simulation.failed',
                'message': {
                    'status': 'failed',
                    'error': str(exc),
                },
            }
        )
        raise


def _serialize_grid(grid) -> bytes:
    buffer = io.BytesIO()
    np_grid = grid.astype('uint8', copy=False)
    np.save(buffer, np_grid, allow_pickle=False)
    return buffer.getvalue()


def _deserialize_grid(blob: bytes) -> np.ndarray:
    buffer = io.BytesIO(blob)
    return np.load(buffer, allow_pickle=False)


def _serialize_ros_result(result: Optional[RateOfSpreadResult]) -> Optional[Dict[str, float]]:
    if result is None:
        return None
    return {
        'ros_m_min': result.ros_m_min,
        'ros_m_s': result.ros_m_s,
        'effective_wind_ms': result.effective_wind_ms,
        'wind_factor': result.wind_factor,
        'slope_factor': result.slope_factor,
        'moisture_factor': result.moisture_factor,
    }


def _deserialize_ros_result(data: Optional[Dict[str, float]]) -> Optional[RateOfSpreadResult]:
    if not data:
        return None
    return RateOfSpreadResult(**data)


def _persist_checkpoint(
    run: SimulationRun,
    simulator: WildfireSimulator,
    tick_index: int,
    max_active: int,
    extinguishment_step: Optional[int],
    final: bool = False,
) -> None:
    payload = {
        'grid': _serialize_grid(simulator.grid),
        'rng_state': simulator.rng.bit_generator.state,
        'environment_budget': simulator.environment.budget,
        'population': simulator.population,
        'footprint': simulator.footprint,
        'extinguishment_time': simulator.extinguishment_time,
        'tick_index': tick_index,
        'last_ros_result': _serialize_ros_result(simulator._last_ros_result),
        'last_ros_distribution': simulator._last_ros_distribution,
    }
    metadata = {
        'active_cells': int(simulator.population),
        'burned_cells': int(simulator.footprint),
        'max_active_cells': int(max_active),
        'extinguishment_step': extinguishment_step,
        'final': final,
    }
    blob = pickle.dumps(payload)
    Checkpoint.objects.update_or_create(
        run=run,
        tick_index=tick_index,
        defaults={
            'payload': blob,
            'metadata': metadata,
        },
    )


def _load_checkpoint_state(simulator: WildfireSimulator, checkpoint: Checkpoint) -> None:
    data = pickle.loads(checkpoint.payload)
    simulator.grid = _deserialize_grid(data['grid'])
    simulator.environment.budget = data.get('environment_budget', simulator.environment.budget)
    simulator.population = data.get('population', simulator.population)
    simulator.footprint = data.get('footprint', simulator.footprint)
    simulator.extinguishment_time = data.get('extinguishment_time')
    simulator.rng.bit_generator.state = data.get('rng_state', simulator.rng.bit_generator.state)
    simulator._last_ros_result = _deserialize_ros_result(data.get('last_ros_result'))
    simulator._last_ros_distribution = data.get('last_ros_distribution')
