from django.contrib import admin
from django.urls import path, include

from control.views import RegisterView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('control.urls')),
    path('api/', include('simulation.api.urls')),
    path('accounts/register/', RegisterView.as_view(), name='register'),
    path('accounts/', include('django.contrib.auth.urls')),
]
