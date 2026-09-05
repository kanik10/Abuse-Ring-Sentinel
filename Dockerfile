FROM python:3.12-slim

WORKDIR /app

# curl is only needed for the HEALTHCHECK below -- everything in requirements.txt
# installs from prebuilt wheels (verified), so no build-essential/gcc is needed here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root
RUN useradd --create-home --uid 1000 sentinel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/app_interactive.py", "--server.port=8501", "--server.address=0.0.0.0"]
