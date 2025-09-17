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
