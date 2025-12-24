FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install backend dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend source code (including static files)
COPY backend ./backend

WORKDIR /app/backend

# Default port (overridden by platform via PORT env var)
ENV PORT=8000
EXPOSE 8000

# Start FastAPI with uvicorn, binding to the injected PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]

