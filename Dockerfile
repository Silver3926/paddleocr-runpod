# ============================================================
#  PaddleOCR PP-StructureV3 untuk Runpod
#
#  Base: runpod/base (bukan runpod/pytorch) supaya image kecil.
#  runpod/base tetap punya start script Runpod, sehingga flag
#  startJupyter / startSsh di template berfungsi.
#
#  PaddlePaddle GPU wheel bersifat self-contained: cukup driver
#  host yang cocok, tidak perlu install CUDA toolkit terpisah.
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
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir paddlepaddle-gpu==3.2.0 \
      -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Gagalkan build lebih awal kalau wheel ternyata bukan versi GPU.
# paddle.version.cuda() tidak butuh GPU fisik, jadi bisa jalan di runner CI.
RUN python3 -c "import paddle; c = paddle.version.cuda(); \
    assert c, 'FATAL: wheel yang terinstall bukan GPU build'; \
    print('OK - Paddle CUDA build:', c)"

# PaddleOCR + dukungan PP-StructureV3 (layout, tabel, formula)
RUN python3 -m pip install --no-cache-dir "paddleocr[doc-parser]" pymupdf

# ------------------------------------------------------------------
#  Bake model ke dalam image.
#  Tujuannya bukan kecepatan, tapi PORTABILITAS: kapasitas GPU Runpod
#  sedang ketat dan Network Volume terikat region. Dengan model di
#  dalam image, pod bisa jalan di region mana pun tanpa unduhan.
#  Kalau ingin image lebih kecil, hapus blok RUN di bawah.
# ------------------------------------------------------------------
ENV PADDLE_PDX_CACHE_HOME=/opt/paddlex
ENV PADDLE_PDX_MODEL_SOURCE=huggingface

RUN python3 -c "\
from paddleocr import PPStructureV3; \
PPStructureV3(use_chart_recognition=False, use_seal_recognition=False)" \
    || echo 'WARN: warm-up dilewati - model akan diunduh saat boot pertama'

COPY run_ocr.py /opt/paddleocr/run_ocr.py

WORKDIR /workspace
