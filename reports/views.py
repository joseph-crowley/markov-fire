from django.shortcuts import render


def placeholder(request):
    return render(request, 'control/base.html', {'message': 'Reports coming soon'})
