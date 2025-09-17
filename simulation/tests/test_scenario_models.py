import uuid

import pytest
from django.utils import timezone

from simulation.models import (
    CalibrationSession,
    Checkpoint,
    Scenario,
    ScenarioVersion,
    SimulationConfig,
    SimulationRun,
)


@pytest.fixture
@pytest.mark.django_db
def user(django_user_model):
    return django_user_model.objects.create_user(username='tester', password='password')


@pytest.fixture
@pytest.mark.django_db
def config(user):
    return SimulationConfig.objects.create(
        name='Base Config',
        slug=f'base-config-{uuid.uuid4().hex[:8]}',
        description='Test config',
        grid_size=20,
        time_steps=10,
        stochastic_seed=42,
        temporal_parameters={},
        spatial_parameters={},
        resource_parameters=[],
        environment_parameters={},
        created_by=user,
    )


@pytest.mark.django_db
def test_scenario_version_autoincrements(user, config):
    scenario = Scenario.objects.create(
        name='Test Scenario',
        slug='test-scenario',
        description='Scenario for testing',
        base_config=config,
        created_by=user,
    )

    version_one = ScenarioVersion.objects.create(
        scenario=scenario,
        config=config,
        config_snapshot={'grid_size': config.grid_size},
        created_by=user,
    )
    version_two = ScenarioVersion.objects.create(
        scenario=scenario,
        config=config,
        config_snapshot={'grid_size': config.grid_size},
        created_by=user,
    )

    assert version_one.version == 1
    assert version_two.version == 2
    assert scenario.active_version_id == version_one.id


@pytest.mark.django_db
def test_calibration_session_defaults(user, config):
    scenario = Scenario.objects.create(
        name='Calibration Scenario',
        slug='calibration-scenario',
        base_config=config,
        created_by=user,
    )
    version = ScenarioVersion.objects.create(
        scenario=scenario,
        config=config,
        config_snapshot={'grid_size': config.grid_size},
        created_by=user,
    )

    session = CalibrationSession.objects.create(
        scenario_version=version,
        config=config,
        created_by=user,
    )

    assert session.status == CalibrationSession.Status.PENDING
    assert session.parameters == {}
    assert session.fitted_parameters == {}
    assert session.goodness_of_fit == {}


@pytest.mark.django_db
def test_checkpoint_enforces_unique_tick(user, config):
    run = SimulationRun.objects.create(
        config=config,
        status=SimulationRun.Status.RUNNING,
        started_at=timezone.now(),
    )
    Checkpoint.objects.create(
        run=run,
        tick_index=5,
        payload=b'checkpoint',
        metadata={'note': 'first'},
        created_by=user,
    )

    with pytest.raises(Exception):
        Checkpoint.objects.create(
            run=run,
            tick_index=5,
            payload=b'duplicate',
        )
