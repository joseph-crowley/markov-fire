from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse_lazy
from django.utils.decorators import method_decorator
from django.views import View
from django.views.generic import ListView, DetailView, CreateView, TemplateView

from simulation.models import SimulationConfig, SimulationRun
from simulation.tasks import run_simulation_task

from .forms import SimulationConfigForm, SimulationRunForm


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
        form = SimulationRunForm(initial=initial)
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = SimulationRunForm(request.POST)
        if form.is_valid():
            run = form.save(commit=False)
            run.status = SimulationRun.Status.PENDING
            run.save()
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
        return context
