import uuid

import pytest
from django.test import override_settings

from simulation.models import (
    SimulationConfig,
    SimulationRun,
    Scenario,
    ScenarioVersion,
    SimulationAnalytics,
    ValidationResult,
)
from simulation.tasks import run_simulation_task


@pytest.fixture
@pytest.mark.django_db
def config(django_user_model):
    user = django_user_model.objects.create_user('analytics-user')
    return SimulationConfig.objects.create(
        name='Analytics Config',
        slug=f'analytics-{uuid.uuid4().hex[:6]}',
        description='Config for analytics tests',
        grid_size=15,
        time_steps=12,
        stochastic_seed=202,
        temporal_parameters={'spread_rate': 0.04},
        spatial_parameters={'ignition_density': 0.85, 'empty_density': 0.15},
        resource_parameters=[],
        environment_parameters={},
        created_by=user,
    )


@override_settings(CHANNEL_LAYERS={'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'}})
@pytest.mark.django_db
def test_run_task_persists_analytics_and_validation(config):
    scenario = Scenario.objects.create(
        name='Analytics Scenario',
        slug='analytics-scenario',
        base_config=config,
    )
    version = ScenarioVersion.objects.create(
        scenario=scenario,
        config=config,
        config_snapshot={},
    )

    run = SimulationRun.objects.create(config=config, scenario_version=version)

    run_simulation_task.apply(args=[str(run.id)])
    run.refresh_from_db()

    analytics = SimulationAnalytics.objects.get(run=run)
    assert analytics.summary['total_ticks'] == run.total_ticks
    assert analytics.summary['max_active_cells'] == run.max_active_cells

    validation = ValidationResult.objects.get(run=run, scenario_version=version)
    assert validation.status == ValidationResult.Status.COMPLETED
    assert validation.metrics['final_footprint'] == analytics.summary['final_footprint']
