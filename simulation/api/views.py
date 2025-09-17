from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from simulation.models import SimulationConfig, SimulationRun, SimulationTick
from simulation.tasks import run_simulation_task

from .serializers import SimulationConfigSerializer, SimulationRunSerializer, SimulationTickSerializer


class SimulationConfigViewSet(viewsets.ModelViewSet):
    queryset = SimulationConfig.objects.all().select_related('created_by')
    serializer_class = SimulationConfigSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)


class SimulationRunViewSet(viewsets.GenericViewSet, mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.CreateModelMixin):
    queryset = SimulationRun.objects.all().select_related('config', 'config__created_by')
    serializer_class = SimulationRunSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        run = serializer.save(status=SimulationRun.Status.PENDING)
        run_simulation_task.delay(str(run.id))

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
