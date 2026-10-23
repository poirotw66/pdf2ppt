# syntax=docker/dockerfile:1
# pdf2ppt: CLI + Review UI API image.
#
# GEMINI_API_KEY / OPENAI_API_KEY are never baked in — pass them at runtime
# via docker compose / `-e` only when using the optional layout planner or
# generative inpaint backends.
#
# LibreOffice is optional (needed only for `--refine` / slide-quality). Build with:
#   docker build --build-arg INSTALL_LIBREOFFICE=1 -t pdf2ppt .

FROM python:3.12-slim-bookworm

# PDF2PPT_ROOT=/data so relative model paths (./model) resolve to the
# /data/model volume, not /app/model under site-packages installs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PDF2PPT_JOB_ROOT=/data/jobs \
    PDF2PPT_ROOT=/data \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

WORKDIR /app

# CJK fonts for PPTX font metrics / optional LibreOffice refine.
# OCR runtime libs (libgomp, libglib) for PaddlePaddle CPU wheels.
ARG INSTALL_LIBREOFFICE=0
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-noto-cjk \
        libgomp1 \
        libglib2.0-0 \
        curl \
    && if [ "$INSTALL_LIBREOFFICE" = "1" ]; then \
         apt-get install -y --no-install-recommends libreoffice-impress; \
       fi \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# OCR + Review UI API. Pin set lives in pyproject ([ocr] / [server]).
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[ocr,server]"

RUN mkdir -p /data/jobs /data/output /data/model \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin pdf2ppt \
    && chown -R pdf2ppt:pdf2ppt /data /app

USER pdf2ppt

VOLUME ["/data/jobs", "/data/output", "/data/model"]
EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8008/health || exit 1

# Default: Review UI backend. Override command for one-shot CLI conversions.
CMD ["pdf2ppt-api"]
