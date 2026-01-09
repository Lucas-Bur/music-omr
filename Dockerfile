FROM python:3.12-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml uv.lock* ./
RUN uv pip install --system -r pyproject.toml

FROM python:3.12-slim

WORKDIR /app

# System-Dependencies für OpenCV installieren
RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg \
  libsm6 \
  libxext6 \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Kopiere nur die installierten Packages
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Scripts und Code kopieren
COPY scripts/ ./scripts/
COPY src/ ./src/

# Modelle herunterladen (einmalig beim Build)
RUN python scripts/initial_setup.py

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.main"]