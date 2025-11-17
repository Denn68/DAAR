#!/bin/sh
set -e

HOST="${DB_HOST:-db}"
PORT="${DB_PORT:-5432}"
TIMEOUT="${WAIT_TIMEOUT:-60}"

echo "[wait-for-db] Waiting for $HOST:$PORT (timeout ${TIMEOUT}s)..."

end=$((SECONDS + TIMEOUT))
while ! nc -z "$HOST" "$PORT" 2>/dev/null; do
  if [ $SECONDS -ge $end ]; then
    echo "[wait-for-db] Timeout after ${TIMEOUT}s; continuing anyway."
    break
  fi
  sleep 1
done

echo "[wait-for-db] Proceeding to start: $*"
exec "$@"
