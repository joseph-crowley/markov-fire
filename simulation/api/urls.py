from rest_framework.routers import DefaultRouter

from .views import (
    SimulationConfigViewSet,
    SimulationRunViewSet,
    ScenarioViewSet,
    ScenarioVersionViewSet,
    ScenarioTagViewSet,
    FuelLayerViewSet,
    MoistureLayerViewSet,
)

router = DefaultRouter()
router.register(r'configs', SimulationConfigViewSet)
router.register(r'runs', SimulationRunViewSet)
router.register(r'scenarios', ScenarioViewSet)
router.register(r'scenario-versions', ScenarioVersionViewSet)
router.register(r'scenario-tags', ScenarioTagViewSet)
router.register(r'fuel-layers', FuelLayerViewSet)
router.register(r'moisture-layers', MoistureLayerViewSet)

urlpatterns = router.urls
