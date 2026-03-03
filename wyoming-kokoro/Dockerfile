FROM python:3.11-slim

WORKDIR /app

# Pin wyoming; no serve_forever needed in code
RUN pip install --no-cache-dir "wyoming==1.8.0" "aiohttp>=3.9.0"

COPY server.py /app/server.py

EXPOSE 10200
CMD ["python", "-u", "/app/server.py"]
