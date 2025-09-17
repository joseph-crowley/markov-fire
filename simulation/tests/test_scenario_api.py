import uuid

import pytest
from django.test import override_settings
from rest_framework.test import APIClient
from unittest.mock import patch

from simulation.models import Scenario, ScenarioVersion, SimulationConfig, ScenarioTag, SimulationRun, SimulationAnalytics
from simulation.tasks import run_simulation_task


@pytest.fixture
@pytest.mark.django_db
def user(django_user_model):
    user = django_user_model.objects.create_user(username='api-user', password='password')
    return user


@pytest.fixture
@pytest.mark.django_db
def client(user):
    client = APIClient()
    client.force_authenticate(user)
    return client


@pytest.fixture
@pytest.mark.django_db
def config(user):
    return SimulationConfig.objects.create(
        name='API Config',
        slug=f'api-config-{uuid.uuid4().hex[:8]}',
        description='Config for API tests',
        grid_size=25,
        time_steps=50,
        stochastic_seed=99,
        temporal_parameters={'spread_rate': 0.04},
        spatial_parameters={'ignition_density': 0.8, 'empty_density': 0.2},
        resource_parameters=[],
        environment_parameters={'slope': 0.1},
        created_by=user,
    )


@override_settings(CHANNEL_LAYERS={'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'}})
@pytest.mark.django_db
def test_create_scenario_and_version(client, user, config):
    tag = ScenarioTag.objects.create(name='High Risk', slug='high-risk')

    scenario_payload = {
        'name': 'Scenario Alpha',
        'slug': 'scenario-alpha',
        'description': 'Test scenario',
        'base_config_id': str(config.id),
        'metadata': {'region': 'north'},
        'tag_ids': [str(tag.id)],
    }
    response = client.post('/api/scenarios/', scenario_payload, format='json')
    assert response.status_code == 201, response.content
    scenario_id = response.data['id']
    scenario = Scenario.objects.get(id=scenario_id)
    assert scenario.tags.filter(id=tag.id).exists()
    assert response.data['active_version'] is None

    version_payload = {
        'scenario': str(scenario_id),
        'config': str(config.id),
        'label': 'Baseline',
        'notes': 'Initial version',
        'metadata': {'weather': 'dry'},
        'set_active': True,
    }
    response = client.post('/api/scenario-versions/', version_payload, format='json')
    assert response.status_code == 201, response.content
    assert response.data['version'] == 1

    scenario.refresh_from_db()
    assert str(scenario.active_version_id) == response.data['id']

    response = client.post('/api/scenario-versions/', version_payload, format='json')
    assert response.status_code == 201
    assert response.data['version'] == 2

    versions = ScenarioVersion.objects.filter(scenario=scenario).order_by('version')
    assert versions.count() == 2

    run_payload = {
        'config_id': str(config.id),
        'seed': 111,
        'scenario_version_id': response.data['id'],
    }
    with patch('simulation.api.views.run_simulation_task.delay') as mock_delay:
        run_response = client.post('/api/runs/', run_payload, format='json')
        mock_delay.assert_called_once()
    assert run_response.status_code == 201, run_response.content
    run_id = run_response.data['id']

    run_simulation_task.apply(args=[run_id])
    SimulationRun.objects.get(id=run_id)

    detail = client.get(f'/api/runs/{run_id}/')
    assert detail.status_code == 200
    assert detail.data['analytics']['summary']['total_ticks'] > 0
    assert detail.data['validation_results'][0]['status'] == 'completed'

    # create another run for diff
    run_payload_2 = {
        'config_id': str(config.id),
        'seed': 222,
        'scenario_version_id': response.data['id'],
    }
    with patch('simulation.api.views.run_simulation_task.delay') as mock_delay:
        run_response_2 = client.post('/api/runs/', run_payload_2, format='json')
        mock_delay.assert_called_once()
    run_id_2 = run_response_2.data['id']
    run_simulation_task.apply(args=[run_id_2])

    diff_response = client.post(
        f'/api/scenarios/{scenario_id}/analytics_diff/',
        {'run_a': run_id, 'run_b': run_id_2},
        format='json',
    )
    assert diff_response.status_code == 200
    delta = diff_response.data['summary_delta']
    assert 'total_ticks' in delta


@pytest.mark.django_db
def test_scenario_list_includes_nested_data(client, config):
    scenario = Scenario.objects.create(
        name='Scenario Bravo',
        slug='scenario-bravo',
        base_config=config,
        metadata={'region': 'south'},
    )
    ScenarioVersion.objects.create(scenario=scenario, config=config, config_snapshot={})

    response = client.get('/api/scenarios/')
    assert response.status_code == 200
    assert len(response.data) >= 1
    payload = next(item for item in response.data if item['id'] == str(scenario.id))
    assert payload['base_config']['id'] == str(config.id)
    assert payload['active_version']['version'] == 1
