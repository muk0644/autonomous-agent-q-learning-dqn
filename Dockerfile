FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps required by some packages (pygame may need SDL libs)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libsdl2-2.0-0 \
       libsdl2-dev \
       libsdl2-image-2.0-0 \
       libsdl2-mixer-2.0-0 \
       libsdl2-ttf-2.0-0 \
       libsmpeg0 \
       libportmidi0 \
       libavformat58 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Default: show help; users should run specific scripts/Notebook interactively
CMD ["bash", "-lc", "echo 'Container ready. Run scripts inside /app, e.g. python 2-q-learning-agent/main.py' && bash"]
