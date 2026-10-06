# Desk Jarvis - Streamlit demo (works locally and on Hugging Face Spaces)
FROM python:3.11-slim

# OpenCV / MediaPipe need these shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
RUN pip install --no-cache-dir --user --no-deps -e . && \
    python scripts/download_models.py

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true"]
