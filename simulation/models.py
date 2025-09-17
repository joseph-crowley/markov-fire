import uuid
from django.conf import settings
from django.db import models
from django.utils import timezone


class TimestampedModel(models.Model):
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class EnvironmentPreset(TimestampedModel):
    name = models.CharField(max_length=128, unique=True)
    description = models.TextField(blank=True)
    parameters = models.JSONField(default=dict)

    def __str__(self) -> str:
        return self.name


class ResourcePreset(TimestampedModel):
    name = models.CharField(max_length=128, unique=True)
    description = models.TextField(blank=True)
    resources = models.JSONField(default=list)

    def __str__(self) -> str:
        return self.name


class SimulationConfig(TimestampedModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=150)
    slug = models.SlugField(max_length=160, unique=True)
    description = models.TextField(blank=True)

    grid_size = models.PositiveIntegerField(default=50)
    time_steps = models.PositiveIntegerField(default=200)
    stochastic_seed = models.PositiveIntegerField(null=True, blank=True)

    temporal_parameters = models.JSONField(default=dict)
    spatial_parameters = models.JSONField(default=dict)
    resource_parameters = models.JSONField(default=list)
    environment_parameters = models.JSONField(default=dict)

    environment_preset = models.ForeignKey(
        EnvironmentPreset, null=True, blank=True, on_delete=models.SET_NULL, related_name='simulation_configs'
    )
    resource_preset = models.ForeignKey(
        ResourcePreset, null=True, blank=True, on_delete=models.SET_NULL, related_name='simulation_configs'
    )

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name='simulation_configs'
    )

    class Meta:
        ordering = ['name']
        unique_together = ('name', 'created_by')

    def __str__(self) -> str:
        return self.name


class SimulationRun(TimestampedModel):
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        RUNNING = 'running', 'Running'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'
        CANCELLED = 'cancelled', 'Cancelled'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    config = models.ForeignKey(SimulationConfig, on_delete=models.CASCADE, related_name='runs')
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.PENDING)
    seed = models.PositiveIntegerField(null=True, blank=True)
    celery_task_id = models.CharField(max_length=255, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True)

    total_ticks = models.PositiveIntegerField(default=0)
    extinguishment_step = models.PositiveIntegerField(null=True, blank=True)
    max_active_cells = models.PositiveIntegerField(default=0)

    scenario_version = models.ForeignKey(
        'ScenarioVersion', null=True, blank=True, on_delete=models.SET_NULL, related_name='runs'
    )
    resume_from = models.ForeignKey(
        'Checkpoint', null=True, blank=True, on_delete=models.SET_NULL, related_name='resumed_runs'
    )

    class Meta:
        ordering = ['-created_at']

    def __str__(self) -> str:
        return f'Run {self.id} ({self.status})'


class SimulationTick(models.Model):
    run = models.ForeignKey(SimulationRun, on_delete=models.CASCADE, related_name='ticks')
    tick_index = models.PositiveIntegerField()
    active_cells = models.PositiveIntegerField(default=0)
    burned_cells = models.PositiveIntegerField(default=0)
    suppressed_cells = models.PositiveIntegerField(default=0)
    footprint = models.PositiveIntegerField(default=0)
    grid_payload = models.BinaryField()
    created_at = models.DateTimeField(default=timezone.now, editable=False)

    class Meta:
        unique_together = ('run', 'tick_index')
        ordering = ['tick_index']

    def __str__(self) -> str:
        return f'Tick {self.tick_index} for {self.run_id}'


class LiveMetric(TimestampedModel):
    run = models.OneToOneField(SimulationRun, on_delete=models.CASCADE, related_name='live_metrics')
    metrics = models.JSONField(default=dict)

    def __str__(self) -> str:
        return f'Live metrics for {self.run_id}'


class ScenarioTag(TimestampedModel):
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=120, unique=True)
    description = models.TextField(blank=True)

    class Meta:
        ordering = ['name']

    def __str__(self) -> str:
        return self.name


class FuelLayer(TimestampedModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=150)
    slug = models.SlugField(max_length=160, unique=True)
    description = models.TextField(blank=True)
    storage_uri = models.CharField(max_length=512)
    resolution_m = models.FloatField(default=30.0)
    bounds = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    checksum = models.CharField(max_length=128, blank=True)

    class Meta:
        ordering = ['name']

    def __str__(self) -> str:
        return self.name


class MoistureLayer(TimestampedModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=150)
    slug = models.SlugField(max_length=160, unique=True)
    description = models.TextField(blank=True)
    storage_uri = models.CharField(max_length=512)
    resolution_m = models.FloatField(default=30.0)
    bounds = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    checksum = models.CharField(max_length=128, blank=True)

    class Meta:
        ordering = ['name']

    def __str__(self) -> str:
        return self.name


class Scenario(TimestampedModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=150)
    slug = models.SlugField(max_length=160, unique=True)
    description = models.TextField(blank=True)
    base_config = models.ForeignKey(
        SimulationConfig, on_delete=models.PROTECT, related_name='scenarios'
    )
    tags = models.ManyToManyField(ScenarioTag, blank=True, related_name='scenarios')
    metadata = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name='scenarios'
    )
    active_version = models.ForeignKey(
        'ScenarioVersion', null=True, blank=True, on_delete=models.SET_NULL, related_name='+'
    )

    class Meta:
        ordering = ['name']

    def __str__(self) -> str:
        return self.name

    def next_version_number(self) -> int:
        latest = self.versions.order_by('-version').first()
        return (latest.version if latest else 0) + 1


class ScenarioVersion(TimestampedModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    scenario = models.ForeignKey(Scenario, on_delete=models.CASCADE, related_name='versions')
    version = models.PositiveIntegerField(null=True, blank=True)
    label = models.CharField(max_length=150, blank=True)
    notes = models.TextField(blank=True)
    config = models.ForeignKey(
        SimulationConfig, on_delete=models.PROTECT, related_name='scenario_versions'
    )
    config_snapshot = models.JSONField(default=dict, blank=True)
    fuel_layer = models.ForeignKey(
        FuelLayer, null=True, blank=True, on_delete=models.SET_NULL, related_name='scenario_versions'
    )
    moisture_layer = models.ForeignKey(
        MoistureLayer, null=True, blank=True, on_delete=models.SET_NULL, related_name='scenario_versions'
    )
    metadata = models.JSONField(default=dict, blank=True)
    is_locked = models.BooleanField(default=False)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name='scenario_versions'
    )

    class Meta:
        ordering = ['scenario', '-version']
        unique_together = ('scenario', 'version')

    def __str__(self) -> str:
        base = f"{self.scenario.name} v{self.version}"
        return f"{base} ({self.label})" if self.label else base

    def save(self, *args, **kwargs):
        if self.version is None:
            self.version = self.scenario.next_version_number()
        super().save(*args, **kwargs)
        scenario = self.scenario
        if scenario.active_version_id is None:
            scenario.active_version = self
            scenario.save(update_fields=['active_version', 'updated_at'])


class TemporalProfile(TimestampedModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    scenario_version = models.ForeignKey(
        ScenarioVersion, on_delete=models.CASCADE, related_name='temporal_profiles'
    )
    name = models.CharField(max_length=150)
    description = models.TextField(blank=True)
    phase_rules = models.JSONField(default=dict, blank=True)
    weather_rules = models.JSONField(default=dict, blank=True)
    resource_rules = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['name']

    def __str__(self) -> str:
        return f"{self.name} ({self.scenario_version})"


class CalibrationSession(TimestampedModel):
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        RUNNING = 'running', 'Running'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'
        CANCELLED = 'cancelled', 'Cancelled'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    scenario_version = models.ForeignKey(
        ScenarioVersion, null=True, blank=True, on_delete=models.SET_NULL, related_name='calibrations'
    )
    config = models.ForeignKey(
        SimulationConfig, on_delete=models.PROTECT, related_name='calibration_sessions'
    )
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.PENDING)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    parameters = models.JSONField(default=dict, blank=True)
    fitted_parameters = models.JSONField(default=dict, blank=True)
    goodness_of_fit = models.JSONField(default=dict, blank=True)
    notes = models.TextField(blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name='calibration_sessions'
    )

    class Meta:
        ordering = ['-created_at']

    def __str__(self) -> str:
        return f'Calibration {self.id} ({self.status})'


class SimulationAnalytics(TimestampedModel):
    run = models.OneToOneField(SimulationRun, on_delete=models.CASCADE, related_name='analytics')
    summary = models.JSONField(default=dict, blank=True)
    distributions = models.JSONField(default=dict, blank=True)
    computed_at = models.DateTimeField(default=timezone.now)

    def __str__(self) -> str:
        return f'Analytics for {self.run_id}'


class ValidationResult(TimestampedModel):
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        RUNNING = 'running', 'Running'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    scenario_version = models.ForeignKey(
        ScenarioVersion, null=True, blank=True, on_delete=models.SET_NULL, related_name='validation_results'
    )
    run = models.ForeignKey(
        SimulationRun, null=True, blank=True, on_delete=models.SET_NULL, related_name='validation_results'
    )
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.PENDING)
    metrics = models.JSONField(default=dict, blank=True)
    comparison_geometry = models.JSONField(default=dict, blank=True)
    notes = models.TextField(blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name='validation_results'
    )

    class Meta:
        ordering = ['-created_at']

    def __str__(self) -> str:
        return f'Validation {self.id} ({self.status})'


class Checkpoint(TimestampedModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(SimulationRun, on_delete=models.CASCADE, related_name='checkpoints')
    tick_index = models.PositiveIntegerField()
    payload = models.BinaryField()
    metadata = models.JSONField(default=dict, blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name='checkpoints'
    )

    class Meta:
        unique_together = ('run', 'tick_index')
        ordering = ['run', 'tick_index']

    def __str__(self) -> str:
        return f'Checkpoint {self.tick_index} for {self.run_id}'
