FROM python:3.11-slim
WORKDIR /app
# Use the lockfile — keeps transitive deps (Starlette, Pydantic, httpcore, ...)
# pinned to what was tested. requirements.txt has only the top-level pins.
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
