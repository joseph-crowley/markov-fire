#!/usr/bin/env bash
set -euo pipefail

python manage.py seed_extreme_scenarios
python manage.py seed_demo_fire
