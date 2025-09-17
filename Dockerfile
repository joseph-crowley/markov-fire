FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ARG APP_UID=1000
ARG APP_GID=1000
RUN if ! getent group ${APP_GID} >/dev/null; then groupadd -g ${APP_GID} appgroup; fi \
    && if ! getent passwd ${APP_UID} >/dev/null; then useradd -m -u ${APP_UID} -g ${APP_GID} appuser; fi

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R ${APP_UID}:${APP_GID} /app

ENV DJANGO_SETTINGS_MODULE=markov_fire_portal.settings

CMD ["sh", "./entrypoint.sh"]
