from django import forms

from simulation.models import SimulationConfig, SimulationRun


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


class SimulationRunForm(forms.ModelForm):
    class Meta:
        model = SimulationRun
        fields = ['config', 'seed']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['config'].widget.attrs['class'] = 'form-select'
        self.fields['seed'].widget.attrs['class'] = 'form-control'
        self.fields['seed'].widget.attrs.setdefault('placeholder', 'Seed (optional)')
