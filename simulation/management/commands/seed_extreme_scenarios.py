from __future__ import annotations

from dataclasses import dataclass

from django.core.management.base import BaseCommand
from django.db import transaction

from simulation.models import (
    SimulationConfig,
    Scenario,
    ScenarioVersion,
    ScenarioTag,
)


@dataclass(frozen=True)
class ExtremeScenarioSpec:
    slug: str
    name: str
    description: str
    label: str
    notes: str
    metadata: dict
    temporal: dict
    spatial: dict
    environment: dict
    resources: list
    grid_size: int
    time_steps: int
    stochastic_seed: int


EXTREME_SPECS: tuple[ExtremeScenarioSpec, ...] = (
    ExtremeScenarioSpec(
        slug="pyrocumulus-runaway",
        name="Pyrocumulus Runaway",
        description="Explosive crown fire with pyroconvective wind feedback and spotting across ridgelines.",
        label="Baseline",
        notes="Designed for training rapid ROS estimation under pyroconvective conditions.",
        metadata={"region": "Sierra crest", "theme": "pyrocumulus"},
        grid_size=96,
        time_steps=360,
        stochastic_seed=1001,
        temporal={
            "spread_rate": 0.12,
            "extinguish_rate": 0.018,
            "firefighting_rate": 0.025,
            "initial_population": 40,
        },
        spatial={
            "ignition_density": 0.95,
            "empty_density": 0.05,
            "variance": 2.5,
            "wind_bias": {"N": 0.9, "NE": 1.2, "E": 1.5, "SE": 1.8, "S": 1.6, "SW": 1.3, "W": 1.0, "NW": 0.85},
            "physics": {
                "enabled": True,
                "fuel_model": "TU5",
                "cell_size_m": 30.0,
                "timestep_minutes": 0.5,
                "wind_speed_ms": 18.0,
                "wind_direction_deg": 120.0,
                "slope_degrees": 28.0,
                "aspect_degrees": 135.0,
                "moisture_dead": 0.04,
                "moisture_live": 0.45,
                "wind_reduction_factor": 0.35,
                "fuel_moisture_adjustment": 0.9,
                "crown_fire": True,
            },
        },
        environment={
            "slope": 0.32,
            "vegetation_density": 0.96,
            "wind_vector": [2.8, 1.9],
            "moisture": 0.05,
            "fuel_type": "timber",
            "humidity": 0.12,
            "temperature": 39,
            "natural_barriers": 0.04,
            "weather_conditions": 0.08,
            "base_budget": 180,
        },
        resources=[
            {"type": "suppression", "strength": 0.14},
            {"type": "firebreak", "strength": 0.12},
            {"type": "thinning", "strength": 0.06},
        ],
    ),
    ExtremeScenarioSpec(
        slug="ember-storm-wildland-urban",
        name="Ember Storm Wildland-Urban",
        description="Dense ember spotting across WUI interface with wind-driven ROS and resource depletion.",
        label="WUI Stress",
        notes="Models ember cast across defensive space with constrained suppression budget.",
        metadata={"region": "Front range", "theme": "ember-storm"},
        grid_size=80,
        time_steps=420,
        stochastic_seed=2025,
        temporal={
            "spread_rate": 0.1,
            "extinguish_rate": 0.02,
            "firefighting_rate": 0.03,
            "initial_population": 30,
        },
        spatial={
            "ignition_density": 0.93,
            "empty_density": 0.07,
            "variance": 3.0,
            "wind_bias": {"N": 0.85, "NE": 1.1, "E": 1.55, "SE": 1.7, "S": 1.4, "SW": 1.1, "W": 0.8, "NW": 0.75},
            "physics": {
                "enabled": True,
                "fuel_model": "GS2",
                "cell_size_m": 20.0,
                "timestep_minutes": 0.75,
                "wind_speed_ms": 16.0,
                "wind_direction_deg": 90.0,
                "slope_degrees": 18.0,
                "aspect_degrees": 110.0,
                "moisture_dead": 0.05,
                "moisture_live": 0.5,
                "wind_reduction_factor": 0.4,
                "fuel_moisture_adjustment": 0.85,
                "crown_fire": False,
            },
        },
        environment={
            "slope": 0.22,
            "vegetation_density": 0.88,
            "wind_vector": [2.3, 1.6],
            "moisture": 0.07,
            "fuel_type": "brush",
            "humidity": 0.2,
            "temperature": 37,
            "natural_barriers": 0.1,
            "weather_conditions": 0.18,
            "base_budget": 140,
        },
        resources=[
            {"type": "suppression", "strength": 0.1},
            {"type": "firebreak", "strength": 0.08},
            {"type": "defensible_space", "strength": 0.05},
        ],
    ),
    ExtremeScenarioSpec(
        slug="megafire-complex",
        name="Cross-Basin Megafire Complex",
        description="Multi-head megafire across basins with limited moisture recovery and escalating ROS.",
        label="Complex",
        notes="Useful for nightly regression on extreme ROS calibration limits.",
        metadata={"region": "Great Basin", "theme": "megafire"},
        grid_size=140,
        time_steps=540,
        stochastic_seed=4096,
        temporal={
            "spread_rate": 0.11,
            "extinguish_rate": 0.017,
            "firefighting_rate": 0.028,
            "initial_population": 55,
        },
        spatial={
            "ignition_density": 0.97,
            "empty_density": 0.03,
            "variance": 2.2,
            "wind_bias": {"N": 1.05, "NE": 1.2, "E": 1.35, "SE": 1.5, "S": 1.4, "SW": 1.25, "W": 1.1, "NW": 1.0},
            "physics": {
                "enabled": True,
                "fuel_model": "TL3",
                "cell_size_m": 40.0,
                "timestep_minutes": 1.0,
                "wind_speed_ms": 14.0,
                "wind_direction_deg": 150.0,
                "slope_degrees": 22.0,
                "aspect_degrees": 160.0,
                "moisture_dead": 0.06,
                "moisture_live": 0.55,
                "wind_reduction_factor": 0.45,
                "fuel_moisture_adjustment": 0.8,
                "crown_fire": True,
            },
        },
        environment={
            "slope": 0.27,
            "vegetation_density": 0.94,
            "wind_vector": [2.0, 1.7],
            "moisture": 0.09,
            "fuel_type": "mixed",
            "humidity": 0.16,
            "temperature": 38,
            "natural_barriers": 0.06,
            "weather_conditions": 0.1,
            "base_budget": 220,
        },
        resources=[
            {"type": "suppression", "strength": 0.18},
            {"type": "firebreak", "strength": 0.14},
            {"type": "thinning", "strength": 0.1},
        ],
    ),
)


class Command(BaseCommand):
    help = "Seed a set of extreme wildfire scenarios with high spread characteristics."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Recreate scenarios even if they already exist (will update existing records).",
        )

    @transaction.atomic
    def handle(self, *args, **options):
        reset = options["reset"]
        tag, _ = ScenarioTag.objects.get_or_create(
            slug="megafire", defaults={"name": "MegaFire", "description": "Extreme rate-of-spread scenarios"}
        )

        created, updated = 0, 0

        for spec in EXTREME_SPECS:
            config_defaults = {
                "name": spec.name,
                "description": spec.description,
                "grid_size": spec.grid_size,
                "time_steps": spec.time_steps,
                "stochastic_seed": spec.stochastic_seed,
                "temporal_parameters": spec.temporal,
                "spatial_parameters": spec.spatial,
                "environment_parameters": spec.environment,
                "resource_parameters": spec.resources,
            }

            config, config_created = SimulationConfig.objects.update_or_create(
                slug=f"{spec.slug}-config",
                defaults=config_defaults,
            )

            scenario_defaults = {
                "name": spec.name,
                "description": spec.description,
                "base_config": config,
                "metadata": spec.metadata,
                "is_active": True,
            }

            scenario, scenario_created = Scenario.objects.update_or_create(
                slug=spec.slug,
                defaults=scenario_defaults,
            )
            scenario.tags.add(tag)

            if reset:
                ScenarioVersion.objects.filter(scenario=scenario).delete()

            if not scenario.versions.exists():
                version = ScenarioVersion.objects.create(
                    scenario=scenario,
                    config=config,
                    label=spec.label,
                    notes=spec.notes,
                    metadata=spec.metadata,
                )
                version_created = True
            else:
                if reset:
                    version = ScenarioVersion.objects.create(
                        scenario=scenario,
                        config=config,
                        label=spec.label,
                        notes=spec.notes,
                        metadata=spec.metadata,
                    )
                    version_created = True
                else:
                    version = scenario.active_version or scenario.versions.order_by('-version').first()
                    version.label = spec.label
                    version.notes = spec.notes
                    version.metadata = spec.metadata
                    version.config = config
                    version.save(update_fields=["label", "notes", "metadata", "config", "updated_at"])
                    version_created = False

            if version_created or scenario.active_version_id is None:
                scenario.active_version = version
                scenario.save(update_fields=["active_version", "updated_at"])

            if scenario_created or config_created or version_created:
                created += 1
            else:
                updated += 1

            self.stdout.write(
                self.style.SUCCESS(
                    f"Seeded scenario '{scenario.slug}' (version v{version.version}) with config '{config.slug}'."
                )
            )

        self.stdout.write(self.style.MIGRATE_HEADING(f"Completed: {created} created, {updated} updated."))
