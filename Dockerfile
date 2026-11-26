# NORMEX — CPU-only production image (offline inference + dashboard).
#
# Build:
#   docker build -t normex:latest .
#
# Run (dashboard on http://127.0.0.1:8765):
#   docker run --rm -p 8765:8765 \
#     -v "$PWD/datasets:/app/datasets" \
#     -v "$PWD/profiles:/app/profiles" \
#     -v "$PWD/results:/app/results" \
#     -e NORMEX_CAMERA_URL="http://192.168.1.100:8080/video" \
#     normex:latest
#
# Train inside the container:
#   docker run --rm -v "$PWD/datasets:/app/datasets" -v "$PWD/profiles:/app/profiles" \
#     normex:latest train --product screw --images datasets/mvtec_ad/screw/train/good

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    NORMEX_HOST=0.0.0.0 \
    NORMEX_PORT=8765

# OpenCV runtime libraries (headless wheel still needs libgl/libglib on slim)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.15.0+cpu torchvision==0.29.2+cpu \
 && pip install -r requirements.txt

COPY . .

# Small entrypoint so both `train ...` and `serve` work from one image.
COPY docker-entrypoint.sh /usr/local/bin/normex-entrypoint
RUN chmod +x /usr/local/bin/normex-entrypoint

EXPOSE 8765
ENTRYPOINT ["normex-entrypoint"]
CMD ["serve"]
