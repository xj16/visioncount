# VisionCount — slim, CPU-only container for the Flask dashboard.
#
# Uses opencv-python-headless (no GUI/X11 libs) so the image stays small and runs
# anywhere, including an arm64 Raspberry Pi. Served by waitress (a production WSGI
# server) rather than Flask's dev server.
#
#   docker build -t visioncount .
#   docker run --rm -p 5000:5000 visioncount
#   # then open http://127.0.0.1:5000  (streams the synthetic feed by default)

FROM python:3.12-slim AS base

# libGL + glib are the only system libs opencv-python-headless needs at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VISIONCOUNT_SOURCE=synthetic \
    VISIONCOUNT_AUTOSTART=1 \
    HOST=0.0.0.0 \
    PORT=5000

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt waitress

# Then the package.
COPY pyproject.toml README.md ./
COPY visioncount ./visioncount
RUN pip install --no-deps -e .

# Persist the optional SQLite event log to a mounted volume if VISIONCOUNT_DB is set.
VOLUME ["/data"]

EXPOSE 5000

# Non-root for safety.
RUN useradd --create-home --uid 10001 appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=4s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,os,sys,json; \
u='http://127.0.0.1:'+os.environ.get('PORT','5000')+'/api/health'; \
d=json.load(urllib.request.urlopen(u,timeout=3)); \
sys.exit(0 if d['status'] in ('ok','starting') else 1)"

# waitress serves the module-level `app` (WSGI callable) from visioncount.app.
CMD ["sh", "-c", "waitress-serve --host=$HOST --port=$PORT --threads=8 visioncount.app:app"]
