# ============================================================
#  PaddleOCR-VL-1.6 untuk Runpod
#
#  Base: runpod/base (bukan runpod/pytorch) supaya image kecil.
#  runpod/base tetap punya start script Runpod, sehingga flag
#  startJupyter / startSsh di template berfungsi.
#
#  Kenapa PaddleOCR-VL-1.6 dan bukan PP-StructureV3:
#  Pada OmniDocBench v1.5, PP-StructureV3 = 86,73 sedangkan
#  PaddleOCR-VL = 92,86. Selisih terbesar justru di tabel
#  (Table TEDS 81,68 -> 90,89) dan formula (CDM 85,79 -> 91,22),
#  yaitu persis yang dibutuhkan buku kuliah scan.
#
#  CATATAN soal model:
#  Model TIDAK di-bake ke image. libpaddle.so ter-link ke
#  libcuda.so.1 (library driver NVIDIA), dan runner GitHub tidak
#  punya GPU maupun driver, sehingga `import paddle` selalu gagal
#  di CI. Model diunduh saat boot pertama pod.
# ============================================================
FROM runpod/base:1.0.2-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive

# libgl1 + libglib2.0-0 -> OpenCV
# poppler-utils         -> pdfimages, untuk cek DPI scan
# tmux                  -> job panjang tetap hidup walau SSH putus
# font-*                -> dipakai saat merender Markdown (ikut Dockerfile resmi)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      libgl1 libglib2.0-0 poppler-utils tmux curl ca-certificates \
      fontconfig fonts-dejavu-core fonts-liberation \
      fonts-noto-cjk fonts-wqy-microhei fonts-freefont-ttf \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# PaddlePaddle GPU 3.2.1 — PaddleOCR-VL mensyaratkan >= 3.2.1
# (sebelumnya kita pakai 3.2.0 untuk PP-StructureV3)
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir paddlepaddle-gpu==3.2.1 \
      -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Verifikasi dependensi CUDA TANPA meng-import paddle.
# `import paddle` tidak mungkin di CI (lihat catatan di atas),
# jadi kita periksa paketnya lewat metadata saja.
RUN python3 -c "import importlib.metadata as m; \
    v = m.version('nvidia-cuda-runtime-cu12'); \
    assert v.startswith('12.6'), f'versi CUDA tidak sesuai: {v}'; \
    m.version('nvidia-cudnn-cu12'); \
    print('OK - dependensi CUDA runtime terpasang:', v)"

# PaddleOCR + dukungan doc-parser (PaddleOCR-VL, layout, tabel, formula)
RUN python3 -m pip install --no-cache-dir "paddleocr[doc-parser]" pymupdf

# Model diunduh saat boot pertama ke sini.
# Cache PaddleOCR-VL jauh lebih kecil dari PP-StructureV3 (~2,2 GB vs ~2,8 GB)
# karena tidak memakai UniMERNet (1,5 GB) maupun SLANeXt (351 MB x2).
ENV PADDLE_PDX_CACHE_HOME=/workspace/.paddlex
ENV PADDLE_PDX_MODEL_SOURCE=huggingface

COPY run_ocr.py /opt/paddleocr/run_ocr.py

WORKDIR /workspace
