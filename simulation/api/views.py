from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from simulation.models import (
    SimulationConfig,
    SimulationRun,
    SimulationTick,
    Scenario,
    ScenarioVersion,
    ScenarioTag,
    FuelLayer,
    MoistureLayer,
    SimulationAnalytics,
)
from simulation.tasks import run_simulation_task
from simulation.services.demo import generate_demo_run

from .serializers import (
    SimulationConfigSerializer,
    SimulationRunSerializer,
    SimulationTickSerializer,
    ScenarioSerializer,
    ScenarioVersionSerializer,
    ScenarioTagSerializer,
    FuelLayerSerializer,
    MoistureLayerSerializer,
    AnalyticsDiffSerializer,
)


class SimulationConfigViewSet(viewsets.ModelViewSet):
    queryset = SimulationConfig.objects.all().select_related('created_by')
    serializer_class = SimulationConfigSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)


class SimulationRunViewSet(viewsets.GenericViewSet, mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.CreateModelMixin):
    queryset = SimulationRun.objects.all().select_related(
        'config',
        'config__created_by',
        'scenario_version',
        'analytics',
        'live_metrics',
    ).prefetch_related('validation_results', 'checkpoints')
    serializer_class = SimulationRunSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        run = serializer.save(status=SimulationRun.Status.PENDING)
        run_simulation_task.delay(str(run.id))

    @action(detail=True, methods=['post'])
    def force_checkpoint(self, request, pk=None):
        run = self.get_object()
        if run.status not in {SimulationRun.Status.RUNNING, SimulationRun.Status.PAUSED}:
            return Response({'detail': 'Run is not active.'}, status=status.HTTP_400_BAD_REQUEST)
        SimulationRun.objects.filter(pk=run.pk).update(checkpoint_requested=True)
        return Response({'detail': 'Checkpoint request queued.'})

    @action(detail=True, methods=['post'])
    def pause(self, request, pk=None):
        run = self.get_object()
        if run.status != SimulationRun.Status.RUNNING:
            return Response({'detail': 'Run is not currently running.'}, status=status.HTTP_400_BAD_REQUEST)
        SimulationRun.objects.filter(pk=run.pk).update(pause_requested=True)
        return Response({'detail': 'Pause request queued.'})

    @action(detail=True, methods=['post'])
    def start(self, request, pk=None):
        run = self.get_object()
        if run.status == SimulationRun.Status.RUNNING:
            return Response({'detail': 'Run already in progress.'}, status=status.HTTP_400_BAD_REQUEST)
        run.status = SimulationRun.Status.PENDING
        run.started_at = None
        run.finished_at = None
        run.total_ticks = 0
        run.save(update_fields=['status', 'started_at', 'finished_at', 'total_ticks'])
        run_simulation_task.delay(str(run.id))
        return Response({'detail': 'Simulation started.'})

    @action(detail=True, methods=['get'])
    def ticks(self, request, pk=None):
        run = self.get_object()
        start = int(request.query_params.get('from', 0))
        limit = int(request.query_params.get('limit', 200))
        ticks = run.ticks.filter(tick_index__gte=start).order_by('tick_index')[:limit]
        serializer = SimulationTickSerializer(ticks, many=True)
        return Response(serializer.data)


class ScenarioViewSet(viewsets.ModelViewSet):
    queryset = Scenario.objects.all().select_related('base_config', 'active_version').prefetch_related('tags')
    serializer_class = ScenarioSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

    @action(detail=True, methods=['post'])
    def analytics_diff(self, request, pk=None):
        scenario = self.get_object()
        serializer = AnalyticsDiffSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        run_a_id = serializer.validated_data['run_a']
        run_b_id = serializer.validated_data['run_b']

        analytics_qs = SimulationAnalytics.objects.filter(run__scenario_version__scenario=scenario)
        try:
            analytics_a = analytics_qs.select_related('run').get(run__id=run_a_id)
            analytics_b = analytics_qs.select_related('run').get(run__id=run_b_id)
        except SimulationAnalytics.DoesNotExist:
            return Response({'detail': 'Analytics not found for one or both runs.'}, status=status.HTTP_404_NOT_FOUND)

        summary_a = analytics_a.summary or {}
        summary_b = analytics_b.summary or {}
        keys = set(summary_a.keys()) | set(summary_b.keys())
        delta = {key: float(summary_b.get(key, 0) - summary_a.get(key, 0)) for key in keys}

        return Response({
            'run_a': str(run_a_id),
            'run_b': str(run_b_id),
            'summary_a': summary_a,
            'summary_b': summary_b,
            'summary_delta': delta,
        })

    @action(detail=True, methods=['post'])
    def seed_demo_run(self, request, pk=None):
        scenario = self.get_object()
        if scenario.slug != 'demo-fire-corridor':
            return Response({'detail': 'Demo seeding available only for the demo scenario.'}, status=status.HTTP_400_BAD_REQUEST)
        result = generate_demo_run(reset=bool(request.data.get('reset', False)))
        run = result.run
        serializer = SimulationRunSerializer(run, context={'request': request})
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class ScenarioVersionViewSet(viewsets.ModelViewSet):
    queryset = ScenarioVersion.objects.all().select_related('scenario', 'config', 'fuel_layer', 'moisture_layer')
    serializer_class = ScenarioVersionSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)


class ScenarioTagViewSet(viewsets.ModelViewSet):
    queryset = ScenarioTag.objects.all()
    serializer_class = ScenarioTagSerializer
    permission_classes = [IsAuthenticated]


class FuelLayerViewSet(viewsets.ModelViewSet):
    queryset = FuelLayer.objects.all()
    serializer_class = FuelLayerSerializer
    permission_classes = [IsAuthenticated]


class MoistureLayerViewSet(viewsets.ModelViewSet):
    queryset = MoistureLayer.objects.all()
    serializer_class = MoistureLayerSerializer
    permission_classes = [IsAuthenticated]
