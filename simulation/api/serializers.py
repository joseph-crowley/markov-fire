from rest_framework import serializers

from simulation.models import SimulationConfig, SimulationRun, SimulationTick


class SimulationConfigSerializer(serializers.ModelSerializer):
    created_by = serializers.StringRelatedField(read_only=True)

    class Meta:
        model = SimulationConfig
        fields = [
            'id',
            'name',
            'slug',
            'description',
            'grid_size',
            'time_steps',
            'stochastic_seed',
            'temporal_parameters',
            'spatial_parameters',
            'resource_parameters',
            'environment_parameters',
            'environment_preset',
            'resource_preset',
            'created_by',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at', 'created_by']


class SimulationRunSerializer(serializers.ModelSerializer):
    config = SimulationConfigSerializer(read_only=True)
    config_id = serializers.PrimaryKeyRelatedField(
        queryset=SimulationConfig.objects.all(), write_only=True, source='config'
    )

    class Meta:
        model = SimulationRun
        fields = [
            'id',
            'config',
            'config_id',
            'status',
            'seed',
            'total_ticks',
            'extinguishment_step',
            'max_active_cells',
            'started_at',
            'finished_at',
            'created_at',
            'notes',
        ]
        read_only_fields = [
            'status', 'total_ticks', 'extinguishment_step', 'max_active_cells',
            'started_at', 'finished_at', 'created_at', 'notes'
        ]


class SimulationTickSerializer(serializers.ModelSerializer):
    class Meta:
        model = SimulationTick
        fields = [
            'tick_index',
            'active_cells',
            'burned_cells',
            'suppressed_cells',
            'footprint',
            'created_at',
        ]
