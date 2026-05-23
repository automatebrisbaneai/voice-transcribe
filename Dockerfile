FROM python:3.11-slim
WORKDIR /app
# Use the pip-compile-generated lockfile with hash verification.
# Every dep + transitive dep is pinned to a specific sha256, so a compromised
# PyPI mirror or tampered package fails the install rather than silently
# substituting a malicious build.
COPY requirements.lock.txt .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
