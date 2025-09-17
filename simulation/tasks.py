import logging
from typing import Dict

from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer
from django.utils import timezone

from .models import SimulationRun, SimulationTick, LiveMetric
from .services.engine import WildfireSimulator
from .services.parameters import build_parameters, DEFAULT_RESOURCES

logger = logging.getLogger(__name__)


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
        channel_layer = get_channel_layer()
        tick_records = []
        max_active = 0
        extinguishment_step = None
        last_tick_index = -1

        for tick in simulator.run():
            last_tick_index = tick.index
            max_active = max(max_active, tick.active_cells)
            if tick.extinguished and extinguishment_step is None:
                extinguishment_step = tick.index

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
