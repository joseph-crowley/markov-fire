import pytest
from django.urls import reverse
from django.contrib.auth import get_user_model


@pytest.mark.django_db
def test_register_creates_user_and_logs_in(client):
    response = client.post(reverse('register'), {
        'username': 'demo_user',
        'email': 'demo@example.com',
        'password1': 'SuperSecure123!',
        'password2': 'SuperSecure123!',
    })
    assert response.status_code == 302
    assert response.url == reverse('control:home')
    user = get_user_model().objects.get(username='demo_user')
    # Ensure user is authenticated for subsequent requests
    dashboard = client.get(reverse('control:home'))
    assert dashboard.status_code == 200
    assert str(user.pk) == str(dashboard.wsgi_request.user.pk)


@pytest.mark.django_db
def test_change_password_flow_requires_auth(client):
    user = get_user_model().objects.create_user(username='change_me', password='oldpass123')
    client.login(username='change_me', password='oldpass123')
    response = client.post(reverse('password_change'), {
        'old_password': 'oldpass123',
        'new_password1': 'NewSecure123!',
        'new_password2': 'NewSecure123!',
    })
    assert response.status_code == 302
    assert response.url == reverse('password_change_done')
    client.logout()
    assert client.login(username='change_me', password='NewSecure123!')
