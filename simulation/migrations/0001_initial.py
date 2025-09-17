from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import uuid
from django.conf import settings


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='EnvironmentPreset',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now, editable=False)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('name', models.CharField(max_length=128, unique=True)),
                ('description', models.TextField(blank=True)),
                ('parameters', models.JSONField(default=dict)),
            ],
            options={'abstract': False},
        ),
        migrations.CreateModel(
            name='ResourcePreset',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now, editable=False)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('name', models.CharField(max_length=128, unique=True)),
                ('description', models.TextField(blank=True)),
                ('resources', models.JSONField(default=list)),
            ],
            options={'abstract': False},
        ),
        migrations.CreateModel(
            name='SimulationConfig',
            fields=[
                ('created_at', models.DateTimeField(default=django.utils.timezone.now, editable=False)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=150)),
                ('slug', models.SlugField(max_length=160, unique=True)),
                ('description', models.TextField(blank=True)),
                ('grid_size', models.PositiveIntegerField(default=50)),
                ('time_steps', models.PositiveIntegerField(default=200)),
                ('stochastic_seed', models.PositiveIntegerField(blank=True, null=True)),
                ('temporal_parameters', models.JSONField(default=dict)),
                ('spatial_parameters', models.JSONField(default=dict)),
                ('resource_parameters', models.JSONField(default=list)),
                ('environment_parameters', models.JSONField(default=dict)),
                ('created_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='simulation_configs', to=settings.AUTH_USER_MODEL)),
                ('environment_preset', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='simulation_configs', to='simulation.environmentpreset')),
                ('resource_preset', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='simulation_configs', to='simulation.resourcepreset')),
            ],
            options={'ordering': ['name']},
        ),
        migrations.CreateModel(
            name='SimulationRun',
            fields=[
                ('created_at', models.DateTimeField(default=django.utils.timezone.now, editable=False)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('running', 'Running'), ('completed', 'Completed'), ('failed', 'Failed'), ('cancelled', 'Cancelled')], default='pending', max_length=12)),
                ('seed', models.PositiveIntegerField(blank=True, null=True)),
                ('celery_task_id', models.CharField(blank=True, max_length=255)),
                ('started_at', models.DateTimeField(blank=True, null=True)),
                ('finished_at', models.DateTimeField(blank=True, null=True)),
                ('notes', models.TextField(blank=True)),
                ('total_ticks', models.PositiveIntegerField(default=0)),
                ('extinguishment_step', models.PositiveIntegerField(blank=True, null=True)),
                ('max_active_cells', models.PositiveIntegerField(default=0)),
                ('config', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='runs', to='simulation.simulationconfig')),
            ],
            options={'ordering': ['-created_at']},
        ),
        migrations.CreateModel(
            name='SimulationTick',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tick_index', models.PositiveIntegerField()),
                ('active_cells', models.PositiveIntegerField(default=0)),
                ('burned_cells', models.PositiveIntegerField(default=0)),
                ('suppressed_cells', models.PositiveIntegerField(default=0)),
                ('footprint', models.PositiveIntegerField(default=0)),
                ('grid_payload', models.BinaryField()),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now, editable=False)),
                ('run', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='ticks', to='simulation.simulationrun')),
            ],
            options={'ordering': ['tick_index']},
        ),
        migrations.CreateModel(
            name='LiveMetric',
            fields=[
                ('created_at', models.DateTimeField(default=django.utils.timezone.now, editable=False)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('metrics', models.JSONField(default=dict)),
                ('run', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='live_metrics', to='simulation.simulationrun')),
            ],
            options={'abstract': False},
        ),
        migrations.AlterUniqueTogether(
            name='simulationtick',
            unique_together={('run', 'tick_index')},
        ),
        migrations.AlterUniqueTogether(
            name='simulationconfig',
            unique_together={('name', 'created_by')},
        ),
    ]
