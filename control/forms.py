from django import forms

from simulation.models import (
    SimulationConfig,
    SimulationRun,
    Scenario,
    ScenarioVersion,
    ScenarioTag,
    FuelLayer,
    MoistureLayer,
    Checkpoint,
)


class JSONFieldWidget(forms.Textarea):
    def __init__(self, *args, **kwargs):
        attrs = kwargs.setdefault('attrs', {})
        attrs.setdefault('class', 'form-control font-monospace')
        attrs.setdefault('rows', 6)
        attrs.setdefault('placeholder', '{\n  "key": "value"\n}')
        super().__init__(*args, **kwargs)

    def format_value(self, value):
        if value in (None, ''):
            return ''
        if isinstance(value, (dict, list)):
            import json
            return json.dumps(value, indent=2)
        return value


class SimulationConfigForm(forms.ModelForm):
    temporal_parameters = forms.JSONField(widget=JSONFieldWidget, required=False)
    spatial_parameters = forms.JSONField(widget=JSONFieldWidget, required=False)
    resource_parameters = forms.JSONField(widget=JSONFieldWidget, required=False)
    environment_parameters = forms.JSONField(widget=JSONFieldWidget, required=False)

    class Meta:
        model = SimulationConfig
        fields = [
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
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            if isinstance(field.widget, JSONFieldWidget):
                continue
            existing = field.widget.attrs.get('class', '')
            field.widget.attrs['class'] = f"form-control {existing}".strip()
            field.widget.attrs.setdefault('placeholder', field.label)


class ScenarioVersionChoiceField(forms.ModelChoiceField):
    def label_from_instance(self, obj):
        label = f"{obj.scenario.name} • v{obj.version}"
        if obj.label:
            label += f" ({obj.label})"
        return label


class CheckpointChoiceField(forms.ModelChoiceField):
    def label_from_instance(self, obj):
        base = f"Run {obj.run_id} – Tick {obj.tick_index}"
        if obj.metadata.get('final'):
            base += " (final)"
        return base


class SimulationRunForm(forms.ModelForm):
    class Meta:
        model = SimulationRun
        fields = ['config', 'scenario_version', 'resume_from', 'seed']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['config'].queryset = SimulationConfig.objects.order_by('name')
        self.fields['config'].widget.attrs['class'] = 'form-select'

        self.fields['scenario_version'] = ScenarioVersionChoiceField(
            queryset=ScenarioVersion.objects.select_related('scenario').order_by('scenario__name', '-version'),
            required=False,
            empty_label='— Select scenario version —',
        )
        self.fields['scenario_version'].widget.attrs['class'] = 'form-select'

        self.fields['resume_from'] = CheckpointChoiceField(
            queryset=Checkpoint.objects.select_related('run').order_by('-created_at')[:100],
            required=False,
            empty_label='— Select checkpoint —',
        )
        self.fields['resume_from'].widget.attrs['class'] = 'form-select'

        self.fields['seed'].widget.attrs['class'] = 'form-control'
        self.fields['seed'].widget.attrs.setdefault('placeholder', 'Seed (optional)')


class ScenarioForm(forms.ModelForm):
    metadata = forms.JSONField(widget=JSONFieldWidget, required=False)
    tag_ids = forms.ModelMultipleChoiceField(
        queryset=ScenarioTag.objects.all().order_by('name'),
        required=False,
        widget=forms.SelectMultiple(attrs={'class': 'form-select', 'size': 6})
    )

    class Meta:
        model = Scenario
        fields = ['name', 'slug', 'description', 'base_config', 'metadata', 'is_active']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['base_config'].queryset = SimulationConfig.objects.order_by('name')
        self.fields['base_config'].widget.attrs['class'] = 'form-select'
        self.fields['name'].widget.attrs['class'] = 'form-control'
        self.fields['slug'].widget.attrs['class'] = 'form-control'
        self.fields['description'].widget.attrs['class'] = 'form-control'
        self.fields['is_active'].widget.attrs['class'] = 'form-check-input'
        if self.instance.pk:
            self.fields['tag_ids'].initial = self.instance.tags.values_list('pk', flat=True)

    def save(self, commit=True):
        tags = self.cleaned_data.pop('tag_ids', [])
        scenario = super().save(commit)
        if commit:
            scenario.tags.set(tags)
        else:
            self._pending_tags = tags
        return scenario


class ScenarioVersionForm(forms.ModelForm):
    metadata = forms.JSONField(widget=JSONFieldWidget, required=False)
    set_active = forms.BooleanField(required=False, initial=True)

    class Meta:
        model = ScenarioVersion
        fields = [
            'scenario',
            'config',
            'label',
            'notes',
            'metadata',
            'fuel_layer',
            'moisture_layer',
            'is_locked',
            'set_active',
        ]

    def __init__(self, *args, scenario: Scenario | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if scenario is not None:
            self.fields['scenario'].initial = scenario
            self.fields['scenario'].queryset = Scenario.objects.filter(pk=scenario.pk)
            self.fields['scenario'].widget = forms.HiddenInput()
        else:
            self.fields['scenario'].widget.attrs['class'] = 'form-select'
        self.fields['config'].queryset = SimulationConfig.objects.order_by('name')
        self.fields['config'].widget.attrs['class'] = 'form-select'
        self.fields['label'].widget.attrs['class'] = 'form-control'
        self.fields['notes'].widget.attrs['class'] = 'form-control'
        self.fields['fuel_layer'].queryset = FuelLayer.objects.order_by('name')
        self.fields['fuel_layer'].widget.attrs['class'] = 'form-select'
        self.fields['moisture_layer'].queryset = MoistureLayer.objects.order_by('name')
        self.fields['moisture_layer'].widget.attrs['class'] = 'form-select'
        self.fields['is_locked'].widget.attrs['class'] = 'form-check-input'
        self.fields['set_active'].widget.attrs['class'] = 'form-check-input'

        if scenario is not None and not self.initial.get('config'):
            self.fields['config'].initial = scenario.base_config

    def clean(self):
        cleaned = super().clean()
        return cleaned

    def save(self, commit=True):
        instance = super().save(commit=False)
        snapshot_fields = [
            'grid_size',
            'time_steps',
            'stochastic_seed',
            'temporal_parameters',
            'spatial_parameters',
            'resource_parameters',
            'environment_parameters',
        ]
        config = instance.config
        instance.config_snapshot = {field: getattr(config, field) for field in snapshot_fields}
        if commit:
            instance.save()
            self.save_m2m()
        return instance
