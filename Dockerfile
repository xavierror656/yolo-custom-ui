FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models so the container is ready out of the box
RUN python - <<'EOF'
from ultralytics import YOLO, YOLOE
print("Downloading YOLO26 models...")
YOLO("yolo26n.pt")
YOLO("yolo26s.pt")
YOLO("yolo26m.pt")
print("Downloading YOLO11 models...")
YOLO("yolo11n.pt")
YOLO("yolo11s.pt")
print("Downloading YOLOE models...")
YOLOE("yoloe-26s-seg.pt")
YOLOE("yoloe-26l-seg.pt")
YOLOE("yoloe-26l-seg-pf.pt")
print("All models downloaded.")
EOF

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--server.enableWebsocketCompression=false", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
