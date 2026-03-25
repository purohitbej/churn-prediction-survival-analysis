# ============================================================
# Stage 1 — Builder
# Install all Python dependencies into isolated layer
# ============================================================
FROM python:3.12-slim AS builder

# Prevents Python from writing .pyc files to disk and from
# buffering stdout/stderr (to see logs immediately)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /install

# Copy only the requirements file first
COPY requirements.txt .

# Install dependencies into a custom directory (/deps) to be cleanly copied into the final stage
RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt


# ============================================================
# Stage 2 — Runtime image
# Production-ready image that only contains what the API needs
# ============================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tell Python where to find the packages installed in Stage 1
    PYTHONPATH=/deps/lib/python3.12/site-packages \
    # Tell the shell where to find executables like gunicorn
    PATH="/deps/bin:$PATH" \
    # Default port; can be overridden with -e PORT=8080 at docker run
    PORT=5000 \
    FLASK_ENV=production

# Creating non-root user for security
# LightGBM requires libgomp (OpenMP) — present on Windows/local Python
# Stripped out of the slim image. Must be installed explicitly.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security.
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --create-home --home-dir /home/appuser appuser

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /deps /deps

# Copy only the source files the API actually needs at runtime
COPY api/        ./api/
COPY src/        ./src/
COPY models/     ./models/
COPY wsgi.py          .
COPY gunicorn.conf.py .

# Hand ownership of the working directory to the non-root user
RUN chown -R appuser:appgroup /app
USER appuser

# Tell Docker that the container listens on this port
# Needs -p 5000:5000 in your docker run command
EXPOSE 5000

# Health check: Docker will curl /health every 30 s.
# After 3 consecutive failures the container is marked unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Production entrypoint using Gunicorn (WSGI server)
# -w 2            → 2 worker processes
# -b 0.0.0.0:5000 → listen on all interfaces at port 5000
# wsgi:app        → module wsgi, variable app (Flask factory)
CMD ["gunicorn", "--config", "gunicorn.conf.py", "wsgi:app"]