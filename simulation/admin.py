from django.contrib import admin
from .models import SimulationConfig, SimulationRun, SimulationTick, EnvironmentPreset, ResourcePreset, LiveMetric


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
