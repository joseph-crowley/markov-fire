from django.contrib import admin
from .models import (
    SimulationConfig,
    SimulationRun,
    SimulationTick,
    EnvironmentPreset,
    ResourcePreset,
    LiveMetric,
    Scenario,
    ScenarioVersion,
    ScenarioTag,
    FuelLayer,
    MoistureLayer,
    CalibrationSession,
    SimulationAnalytics,
    ValidationResult,
    Checkpoint,
)


@admin.register(SimulationConfig)
class SimulationConfigAdmin(admin.ModelAdmin):
    list_display = ('name', 'grid_size', 'time_steps', 'created_by', 'created_at')
    prepopulated_fields = {'slug': ('name',)}
    search_fields = ('name', 'description')
    list_filter = ('created_by',)


@admin.register(SimulationRun)
class SimulationRunAdmin(admin.ModelAdmin):
    list_display = ('id', 'config', 'status', 'started_at', 'finished_at', 'total_ticks')
    list_filter = ('status', 'created_at')
    search_fields = ('id', 'config__name')


@admin.register(SimulationTick)
class SimulationTickAdmin(admin.ModelAdmin):
    list_display = ('run', 'tick_index', 'active_cells', 'burned_cells', 'suppressed_cells')
    list_filter = ('run',)


@admin.register(EnvironmentPreset)
class EnvironmentPresetAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at')
    search_fields = ('name',)


@admin.register(ResourcePreset)
class ResourcePresetAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at')
    search_fields = ('name',)


@admin.register(LiveMetric)
class LiveMetricAdmin(admin.ModelAdmin):
    list_display = ('run', 'updated_at')


class ScenarioVersionInline(admin.TabularInline):
    model = ScenarioVersion
    extra = 0
    fields = ('version', 'label', 'config', 'is_locked', 'created_at')
    readonly_fields = ('version', 'created_at')


@admin.register(Scenario)
class ScenarioAdmin(admin.ModelAdmin):
    list_display = ('name', 'slug', 'base_config', 'is_active', 'active_version_display', 'created_by', 'created_at')
    search_fields = ('name', 'slug', 'description')
    list_filter = ('is_active', 'created_by')
    prepopulated_fields = {'slug': ('name',)}
    filter_horizontal = ('tags',)
    inlines = [ScenarioVersionInline]

    @admin.display(description='Active Version')
    def active_version_display(self, obj):
        if obj.active_version:
            return f'v{obj.active_version.version} ({obj.active_version.label or ""})'.strip()
        return '-'


@admin.register(ScenarioVersion)
class ScenarioVersionAdmin(admin.ModelAdmin):
    list_display = ('scenario', 'version', 'label', 'config', 'fuel_layer', 'moisture_layer', 'is_locked', 'created_at')
    list_filter = ('scenario', 'is_locked')
    search_fields = ('scenario__name', 'label')
    autocomplete_fields = ('scenario', 'config', 'fuel_layer', 'moisture_layer')


@admin.register(ScenarioTag)
class ScenarioTagAdmin(admin.ModelAdmin):
    list_display = ('name', 'slug', 'created_at')
    search_fields = ('name', 'slug')
    prepopulated_fields = {'slug': ('name',)}


@admin.register(FuelLayer)
class FuelLayerAdmin(admin.ModelAdmin):
    list_display = ('name', 'slug', 'storage_uri', 'resolution_m', 'created_at')
    search_fields = ('name', 'slug', 'storage_uri')
    prepopulated_fields = {'slug': ('name',)}


@admin.register(MoistureLayer)
class MoistureLayerAdmin(admin.ModelAdmin):
    list_display = ('name', 'slug', 'storage_uri', 'resolution_m', 'created_at')
    search_fields = ('name', 'slug', 'storage_uri')
    prepopulated_fields = {'slug': ('name',)}


@admin.register(CalibrationSession)
class CalibrationSessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'scenario_version', 'config', 'status', 'started_at', 'finished_at')
    list_filter = ('status',)
    search_fields = ('id', 'scenario_version__scenario__name')
    autocomplete_fields = ('scenario_version', 'config', 'created_by')


@admin.register(SimulationAnalytics)
class SimulationAnalyticsAdmin(admin.ModelAdmin):
    list_display = ('run', 'computed_at')
    search_fields = ('run__id',)
    autocomplete_fields = ('run',)


@admin.register(ValidationResult)
class ValidationResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'scenario_version', 'run', 'status', 'created_at')
    list_filter = ('status',)
    search_fields = ('id', 'scenario_version__scenario__name')
    autocomplete_fields = ('scenario_version', 'run', 'created_by')


@admin.register(Checkpoint)
class CheckpointAdmin(admin.ModelAdmin):
    list_display = ('run', 'tick_index', 'created_at')
    list_filter = ('run',)
    search_fields = ('run__id',)
    autocomplete_fields = ('run', 'created_by')
