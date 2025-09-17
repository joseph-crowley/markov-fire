from rest_framework import serializers

from simulation.models import (
    SimulationConfig,
    SimulationRun,
    SimulationTick,
    ScenarioVersion,
    Scenario,
    ScenarioTag,
    FuelLayer,
    MoistureLayer,
    Checkpoint,
    SimulationAnalytics,
    ValidationResult,
    LiveMetric,
)


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


class CheckpointSerializer(serializers.ModelSerializer):
    class Meta:
        model = Checkpoint
        fields = ['id', 'tick_index', 'metadata', 'created_at']


class SimulationAnalyticsSerializer(serializers.ModelSerializer):
    class Meta:
        model = SimulationAnalytics
        fields = ['summary', 'distributions', 'computed_at']


class ValidationResultSerializer(serializers.ModelSerializer):
    scenario_version = serializers.PrimaryKeyRelatedField(read_only=True)

    class Meta:
        model = ValidationResult
        fields = ['id', 'scenario_version', 'status', 'metrics', 'created_at']


class AnalyticsDiffSerializer(serializers.Serializer):
    run_a = serializers.UUIDField()
    run_b = serializers.UUIDField()
    summary_delta = serializers.DictField(child=serializers.FloatField(), read_only=True)
    summary_a = serializers.DictField(read_only=True)
    summary_b = serializers.DictField(read_only=True)


class SimulationRunSerializer(serializers.ModelSerializer):
    config = SimulationConfigSerializer(read_only=True)
    config_id = serializers.PrimaryKeyRelatedField(
        queryset=SimulationConfig.objects.all(), write_only=True, source='config'
    )
    scenario_version = serializers.UUIDField(source='scenario_version_id', read_only=True)
    scenario_version_id = serializers.PrimaryKeyRelatedField(
        queryset=ScenarioVersion.objects.all(), write_only=True, allow_null=True, required=False, source='scenario_version'
    )
    resume_from = serializers.UUIDField(source='resume_from_id', read_only=True)
    resume_from_id = serializers.PrimaryKeyRelatedField(
        queryset=Checkpoint.objects.all(), write_only=True, allow_null=True, required=False, source='resume_from'
    )
    analytics = SimulationAnalyticsSerializer(read_only=True)
    validation_results = ValidationResultSerializer(many=True, read_only=True)
    live_metrics = serializers.SerializerMethodField()
    checkpoints = CheckpointSerializer(many=True, read_only=True)

    class Meta:
        model = SimulationRun
        fields = [
            'id',
            'config',
            'config_id',
            'scenario_version',
            'scenario_version_id',
            'resume_from',
            'resume_from_id',
            'status',
            'seed',
            'total_ticks',
            'extinguishment_step',
            'max_active_cells',
            'started_at',
            'finished_at',
            'created_at',
            'notes',
            'analytics',
            'validation_results',
            'live_metrics',
            'checkpoints',
        ]
        read_only_fields = [
            'status', 'total_ticks', 'extinguishment_step', 'max_active_cells',
            'started_at', 'finished_at', 'created_at', 'notes',
            'scenario_version', 'resume_from'
        ]

    def get_live_metrics(self, obj):
        try:
            return obj.live_metrics.metrics
        except LiveMetric.DoesNotExist:
            return None


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


class ScenarioTagSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScenarioTag
        fields = ['id', 'name', 'slug', 'description', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at']


class FuelLayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = FuelLayer
        fields = [
            'id',
            'name',
            'slug',
            'description',
            'storage_uri',
            'resolution_m',
            'bounds',
            'metadata',
            'checksum',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at']


class MoistureLayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = MoistureLayer
        fields = [
            'id',
            'name',
            'slug',
            'description',
            'storage_uri',
            'resolution_m',
            'bounds',
            'metadata',
            'checksum',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at']


class ScenarioVersionSerializer(serializers.ModelSerializer):
    scenario = serializers.PrimaryKeyRelatedField(queryset=Scenario.objects.all())
    config = serializers.PrimaryKeyRelatedField(queryset=SimulationConfig.objects.all())
    fuel_layer = serializers.PrimaryKeyRelatedField(
        queryset=FuelLayer.objects.all(), allow_null=True, required=False
    )
    moisture_layer = serializers.PrimaryKeyRelatedField(
        queryset=MoistureLayer.objects.all(), allow_null=True, required=False
    )
    set_active = serializers.BooleanField(write_only=True, required=False, default=False)

    class Meta:
        model = ScenarioVersion
        fields = [
            'id',
            'scenario',
            'version',
            'label',
            'notes',
            'config',
            'config_snapshot',
            'fuel_layer',
            'moisture_layer',
            'metadata',
            'is_locked',
            'set_active',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['version', 'config_snapshot', 'created_at', 'updated_at']

    def create(self, validated_data):
        scenario = validated_data['scenario']
        config = validated_data['config']
        set_active = validated_data.pop('set_active', False)
        snapshot_fields = [
            'grid_size',
            'time_steps',
            'stochastic_seed',
            'temporal_parameters',
            'spatial_parameters',
            'resource_parameters',
            'environment_parameters',
        ]
        config_snapshot = {field: getattr(config, field) for field in snapshot_fields}
        validated_data.setdefault('config_snapshot', config_snapshot)
        instance = super().create(validated_data)
        if set_active or scenario.active_version_id is None:
            scenario.active_version = instance
            scenario.save(update_fields=['active_version', 'updated_at'])
        return instance


class ScenarioSerializer(serializers.ModelSerializer):
    base_config = SimulationConfigSerializer(read_only=True)
    base_config_id = serializers.PrimaryKeyRelatedField(
        queryset=SimulationConfig.objects.all(), write_only=True, source='base_config'
    )
    tags = ScenarioTagSerializer(many=True, read_only=True)
    tag_ids = serializers.PrimaryKeyRelatedField(
        queryset=ScenarioTag.objects.all(), many=True, write_only=True, required=False
    )
    active_version = ScenarioVersionSerializer(read_only=True)

    class Meta:
        model = Scenario
        fields = [
            'id',
            'name',
            'slug',
            'description',
            'base_config',
            'base_config_id',
            'metadata',
            'is_active',
            'tags',
            'tag_ids',
            'active_version',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at', 'tags', 'active_version']

    def create(self, validated_data):
        tags = validated_data.pop('tag_ids', [])
        scenario = super().create(validated_data)
        if tags:
            scenario.tags.set(tags)
        return scenario

    def update(self, instance, validated_data):
        tags = validated_data.pop('tag_ids', None)
        scenario = super().update(instance, validated_data)
        if tags is not None:
            scenario.tags.set(tags)
        return scenario
