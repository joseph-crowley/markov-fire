from django.urls import path

from . import views

app_name = 'control'

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('configs/', views.SimulationConfigListView.as_view(), name='config-list'),
    path('configs/new/', views.SimulationConfigCreateView.as_view(), name='config-create'),
    path('runs/', views.SimulationRunListView.as_view(), name='run-list'),
    path('runs/new/', views.SimulationRunCreateView.as_view(), name='run-create'),
    path('runs/<uuid:pk>/', views.SimulationRunDetailView.as_view(), name='run-detail'),
    path('scenarios/', views.ScenarioListView.as_view(), name='scenario-list'),
    path('scenarios/new/', views.ScenarioCreateView.as_view(), name='scenario-create'),
    path('scenarios/<slug:slug>/', views.ScenarioDetailView.as_view(), name='scenario-detail'),
    path('scenarios/<slug:slug>/versions/new/', views.ScenarioVersionCreateView.as_view(), name='scenario-version-create'),
]
