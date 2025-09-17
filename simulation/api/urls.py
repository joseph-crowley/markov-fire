from rest_framework.routers import DefaultRouter

from .views import SimulationConfigViewSet, SimulationRunViewSet

router = DefaultRouter()
router.register(r'configs', SimulationConfigViewSet)
router.register(r'runs', SimulationRunViewSet)

urlpatterns = router.urls
