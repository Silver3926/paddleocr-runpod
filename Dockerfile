# ============================================================
#  PaddleOCR PP-StructureV3 untuk Runpod
#
#  Base: runpod/base (bukan runpod/pytorch) supaya image kecil.
#  runpod/base tetap punya start script Runpod, sehingga flag
#  startJupyter / startSsh di template berfungsi.
#
#  CATATAN PENTING soal model:
#  Model TIDAK di-bake ke image, dan itu bukan pilihan gaya.
#  libpaddle.so ter-link ke libcuda.so.1 (library driver NVIDIA).
#  Runner GitHub Actions tidak punya GPU maupun driver, jadi
#  `import paddle` selalu gagal di CI. Karena itu warm-up model
#  tidak mungkin dilakukan di sini; model diunduh saat boot
#  pertama pod, ke Network Volume supaya persisten.
# ============================================================
FROM runpod/base:1.0.2-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive

# libgl1 + libglib2.0-0 -> OpenCV
# poppler-utils         -> pdfimages, untuk cek DPI scan
# tmux                  -> job panjang tetap hidup walau SSH putus
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      libgl1 libglib2.0-0 poppler-utils tmux curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# PaddlePaddle GPU — cu126 (butuh driver host >= 550.54.14)
# Wheel ini self-contained: 1,85 GB wheel + ~2 GB paket nvidia-*.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir paddlepaddle-gpu==3.2.0 \
      -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Verifikasi dependensi CUDA TANPA meng-import paddle.
# `import paddle` tidak mungkin di CI (lihat catatan di atas),
# jadi kita periksa paketnya lewat metadata saja.
RUN python3 -c "import importlib.metadata as m; \
    v = m.version('nvidia-cuda-runtime-cu12'); \
    assert v.startswith('12.6'), f'versi CUDA tidak sesuai: {v}'; \
    m.version('nvidia-cudnn-cu12'); \
    print('OK - dependensi CUDA runtime terpasang:', v)"

# PaddleOCR + dukungan PP-StructureV3 (layout, tabel, formula)
RUN python3 -m pip install --no-cache-dir "paddleocr[doc-parser]" pymupdf

# Model diunduh saat boot pertama ke Network Volume (persisten),
# sehingga sesi berikutnya tidak mengunduh ulang.
# Kalau volume tidak ter-mount, /workspace ada di container disk dan
# model akan hilang saat pod di-stop (sekadar mengunduh ulang).
ENV PADDLE_PDX_CACHE_HOME=/workspace/.paddlex
ENV PADDLE_PDX_MODEL_SOURCE=huggingface

COPY run_ocr.py /opt/paddleocr/run_ocr.py

WORKDIR /workspace
