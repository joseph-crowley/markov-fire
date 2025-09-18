import pytest
from django.core.management import call_command

from simulation.management.commands.seed_extreme_scenarios import EXTREME_SPECS
from simulation.models import Scenario, ScenarioTag, ScenarioVersion, SimulationRun, Checkpoint


@pytest.mark.django_db
def test_seed_extreme_scenarios_idempotent():
    call_command('seed_extreme_scenarios')

    slugs = [spec.slug for spec in EXTREME_SPECS]
    scenarios = Scenario.objects.filter(slug__in=slugs)
    assert scenarios.count() == len(slugs)

    for scenario in scenarios:
        assert scenario.tags.filter(slug='megafire').exists()
        config = scenario.base_config
        assert config.temporal_parameters['spread_rate'] >= 0.1
        assert config.spatial_parameters.get('physics', {}).get('enabled') is True

    version_counts = {
        scenario.slug: ScenarioVersion.objects.filter(scenario=scenario).count()
        for scenario in scenarios
    }

    call_command('seed_extreme_scenarios')

    for scenario in Scenario.objects.filter(slug__in=slugs):
        assert ScenarioVersion.objects.filter(scenario=scenario).count() == version_counts[scenario.slug]


@pytest.mark.django_db
def test_seed_extreme_scenarios_reset_rebuilds_versions():
    call_command('seed_extreme_scenarios')
    slug = EXTREME_SPECS[0].slug
    scenario = Scenario.objects.get(slug=slug)
    original_versions = ScenarioVersion.objects.filter(scenario=scenario).count()

    call_command('seed_extreme_scenarios', '--reset')

    scenario.refresh_from_db()
    assert ScenarioVersion.objects.filter(scenario=scenario).count() == original_versions
    latest = scenario.active_version
    assert latest is not None
    assert latest.metadata['theme'] == EXTREME_SPECS[0].metadata['theme']


@pytest.mark.django_db
def test_seed_demo_fire_generates_long_run():
    call_command('seed_demo_fire', '--reset')
    scenario = Scenario.objects.get(slug='demo-fire-corridor')
    run = SimulationRun.objects.filter(scenario_version__scenario=scenario).order_by('-created_at').first()
    assert run.total_ticks == 200
    assert Checkpoint.objects.filter(run=run).count() >= 1
    assert run.status == SimulationRun.Status.COMPLETED
