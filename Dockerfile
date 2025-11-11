FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
RUN mkdir -p uploads models logs

EXPOSE 8080
ENV PORT=8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "app.py"]
