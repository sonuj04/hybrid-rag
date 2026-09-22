FROM python:3.12-slim

WORKDIR /app

# pymupdf needs a C build toolchain at install time
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only PyTorch first: the wheel sentence-transformers would pull in by default
# targets a GPU, which this laptop doesn't have, and is much larger.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app app
COPY scripts scripts

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]