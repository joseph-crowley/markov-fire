from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse_lazy
from django.utils.decorators import method_decorator
from django.views import View
from django.views.generic import ListView, DetailView, CreateView, TemplateView
from django.db.models import Count

from simulation.models import (
    SimulationConfig,
    SimulationRun,
    Scenario,
    ScenarioVersion,
    ScenarioTag,
    CalibrationSession,
    ValidationResult,
    SimulationAnalytics,
    Checkpoint,
)
from simulation.tasks import run_simulation_task

from .forms import (
    SimulationConfigForm,
    SimulationRunForm,
    ScenarioForm,
    ScenarioVersionForm,
)


class HomeView(TemplateView):
    template_name = 'control/home.html'


class SimulationConfigListView(LoginRequiredMixin, ListView):
    model = SimulationConfig
    template_name = 'control/config_list.html'
    context_object_name = 'configs'


class SimulationConfigCreateView(LoginRequiredMixin, CreateView):
    model = SimulationConfig
    form_class = SimulationConfigForm
    template_name = 'control/config_form.html'
    success_url = reverse_lazy('control:config-list')

    def form_valid(self, form):
        form.instance.created_by = self.request.user
        messages.success(self.request, 'Configuration created successfully.')
        return super().form_valid(form)


class SimulationRunListView(LoginRequiredMixin, ListView):
    model = SimulationRun
    template_name = 'control/run_list.html'
    context_object_name = 'runs'
    paginate_by = 20

    def get_queryset(self):
        return SimulationRun.objects.select_related('config').order_by('-created_at')


@method_decorator(login_required, name='dispatch')
class SimulationRunCreateView(View):
    template_name = 'control/run_form.html'

    def get(self, request):
        initial = {}
        if config_id := request.GET.get('config'):
            initial['config'] = get_object_or_404(SimulationConfig, pk=config_id)
        if version_id := request.GET.get('scenario_version'):
            initial['scenario_version'] = get_object_or_404(ScenarioVersion, pk=version_id)
        if checkpoint_id := request.GET.get('checkpoint'):
            initial['resume_from'] = get_object_or_404(Checkpoint, pk=checkpoint_id)
        form = SimulationRunForm(initial=initial)
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = SimulationRunForm(request.POST)
        if form.is_valid():
            run = form.save(commit=False)
            run.status = SimulationRun.Status.PENDING
            run.save()
            form.save_m2m()
            run_simulation_task.delay(str(run.id))
            messages.success(request, 'Simulation run created and started.')
            return redirect('control:run-detail', pk=run.id)
        return render(request, self.template_name, {'form': form})


class SimulationRunDetailView(LoginRequiredMixin, DetailView):
    model = SimulationRun
    template_name = 'control/run_detail.html'
    context_object_name = 'run'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['config'] = self.object.config
        context['scenario_version'] = self.object.scenario_version
        context['resume_from'] = self.object.resume_from
        context['checkpoints'] = self.object.checkpoints.order_by('-tick_index')
        context['analytics'] = getattr(self.object, 'analytics', None)
        return context


class ScenarioListView(LoginRequiredMixin, ListView):
    model = Scenario
    template_name = 'control/scenario_list.html'
    context_object_name = 'scenarios'

    def get_queryset(self):
        return (
            Scenario.objects.select_related('base_config', 'active_version')
            .prefetch_related('tags')
            .order_by('name')
        )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['tag_summary'] = ScenarioTag.objects.annotate(scenario_count=Count('scenarios')).order_by('name')
        return context


class ScenarioCreateView(LoginRequiredMixin, CreateView):
    model = Scenario
    form_class = ScenarioForm
    template_name = 'control/scenario_form.html'

    def form_valid(self, form):
        form.instance.created_by = self.request.user
        messages.success(self.request, 'Scenario created.')
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy('control:scenario-detail', kwargs={'slug': self.object.slug})


class ScenarioDetailView(LoginRequiredMixin, DetailView):
    model = Scenario
    slug_field = 'slug'
    slug_url_kwarg = 'slug'
    template_name = 'control/scenario_detail.html'
    context_object_name = 'scenario'

    def get_queryset(self):
        return Scenario.objects.select_related('base_config', 'active_version').prefetch_related('tags')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        versions = (
            self.object.versions.select_related('config', 'fuel_layer', 'moisture_layer', 'created_by')
            .order_by('-version')
        )
        context['versions'] = versions
        context['calibrations'] = (
            CalibrationSession.objects.filter(scenario_version__scenario=self.object)
            .select_related('scenario_version', 'config')
            .order_by('-created_at')[:10]
        )
        context['validations'] = (
            ValidationResult.objects.filter(scenario_version__scenario=self.object)
            .select_related('scenario_version', 'run')
            .order_by('-created_at')[:10]
        )
        context['analytics'] = (
            SimulationAnalytics.objects.filter(run__scenario_version__scenario=self.object)
            .select_related('run', 'run__config', 'run__scenario_version')
            .order_by('-computed_at')[:10]
        )
        context['analytics_payload'] = []
        for record in context['analytics']:
            scenario_version = record.run.scenario_version
            version_label = f"v{scenario_version.version}" if scenario_version else 'No version'
            context['analytics_payload'].append({
                'run_id': str(record.run_id),
                'label': f"Run {record.run_id} · {version_label}",
                'summary': record.summary or {},
                'distributions': record.distributions or {},
            })
        context['is_demo_scenario'] = (self.object.metadata or {}).get('theme') == 'demo-corridor'
        return context


class ScenarioVersionCreateView(LoginRequiredMixin, CreateView):
    model = ScenarioVersion
    form_class = ScenarioVersionForm
    template_name = 'control/scenario_version_form.html'

    def dispatch(self, request, *args, **kwargs):
        self.scenario = get_object_or_404(Scenario, slug=kwargs['slug'])
        return super().dispatch(request, *args, **kwargs)

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['scenario'] = self.scenario
        return kwargs

    def form_valid(self, form):
        form.instance.created_by = self.request.user
        response = super().form_valid(form)
        if form.cleaned_data.get('set_active'):
            scenario = self.object.scenario
            scenario.active_version = self.object
            scenario.save(update_fields=['active_version', 'updated_at'])
        messages.success(self.request, f'Scenario version v{self.object.version} created.')
        return response

    def get_success_url(self):
        return reverse_lazy('control:scenario-detail', kwargs={'slug': self.scenario.slug})

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['scenario'] = self.scenario
        return context
