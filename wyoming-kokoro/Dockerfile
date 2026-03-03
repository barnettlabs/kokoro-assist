FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir wyoming aiohttp

COPY server.py /app/server.py
CMD ["python", "-u", "server.py"]
