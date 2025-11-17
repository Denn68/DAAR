FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.115.5 \
    "uvicorn[standard]==0.32.0" \
    psycopg2-binary==2.9.10 \
    pydantic==2.9.2

COPY server.py /app/server.py

COPY wait-for-db.sh /usr/local/bin/wait-for-db.sh
RUN sed -i 's/\r$//' /usr/local/bin/wait-for-db.sh && \
    chmod +x /usr/local/bin/wait-for-db.sh

ENTRYPOINT ["/bin/sh", "/usr/local/bin/wait-for-db.sh"]
CMD ["python", "server.py"]
