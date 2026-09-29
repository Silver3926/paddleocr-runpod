# ============================================================
#  PaddleOCR-VL-1.6 untuk Runpod
#
#  Base: runpod/base (bukan runpod/pytorch) supaya image kecil.
#  runpod/base tetap punya start script Runpod, sehingga flag
#  startJupyter / startSsh di template berfungsi.
#
#  Terbukti jalan: paddlepaddle-gpu 3.2.1 cu126 mendeteksi RTX A6000
#  (CC 8.6) pada host dengan Driver API 13.2 / Runtime API 12.6.
#  Wheel-nya self-contained, tidak butuh CUDA toolkit sistem.
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

# jupyterlab WAJIB ada di interpreter yang sama dengan `python3`.
#
# /start.sh milik Runpod menyalakan JupyterLab lewat:
#     nohup python3 -m jupyter lab ... &> /jupyter.log &
#
# Di image ini `python3` adalah 3.10.12 (tempat Paddle terpasang),
# sementara entrypoint `jupyter` menunjuk ke Python 3.12. Akibatnya
# `python3 -m jupyter` gagal seketika, tapi /start.sh tetap mencetak
# "Jupyter Lab started", dan pesan errornya dibuang ke /jupyter.log
# sehingga tidak terlihat di log container sama sekali.
#
# Karena itu jupyterlab dipasang ke interpreter yang sama dengan
# `python3`, bukan ke 3.12. Efek sampingnya bagus: notebook ikut
# berjalan di 3.10, jadi `import paddle` di dalam notebook juga bekerja.
RUN python3 -m pip install --no-cache-dir "paddleocr[doc-parser]" pymupdf jupyterlab

# Penjaga: pastikan modul jupyter benar-benar bisa dipanggil lewat
# `python3`. Ini persis yang dulu gagal tanpa suara. Perintah ini tidak
# butuh GPU, jadi bisa dijalankan di runner CI dan membatalkan build
# lebih awal kalau suatu saat ada yang mengubahnya lagi.
RUN python3 -m jupyter --version

# Model diunduh saat boot pertama ke sini.
# Cache PaddleOCR-VL sekitar 2,2 GB — lebih kecil dari PP-StructureV3
# karena tidak memakai UniMERNet (1,5 GB) maupun SLANeXt (351 MB x2).
ENV PADDLE_PDX_CACHE_HOME=/workspace/.paddlex
ENV PADDLE_PDX_MODEL_SOURCE=huggingface

COPY run_ocr.py /opt/paddleocr/run_ocr.py

WORKDIR /workspace
