#!/bin/sh
set -e

python manage.py makemigrations --noinput
python manage.py migrate --noinput
python manage.py collectstatic --noinput --clear

APP_PORT="${APP_PORT:-8000}"

exec daphne -b 0.0.0.0 -p "$APP_PORT" markov_fire_portal.asgi:application
