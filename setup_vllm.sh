#!/bin/bash
# ============================================================
#  Pasang dependensi server vLLM untuk PaddleOCR-VL-1.6.
#
#  Dijalankan DI DALAM pod. Tujuan awalnya untuk mencari perintah
#  yang benar-benar bekerja; setelah terbukti, perintahnya dibekukan
#  ke Dockerfile supaya tidak perlu dipasang lagi tiap pod.
#
#  Pakai:
#      bash setup_vllm.sh
# ============================================================
set -uo pipefail

# Wheel FlashAttention siap pakai (disebut di dokumentasi PaddleOCR).
# Dibangun untuk cp310 + torch 2.8 + cu128 — cocok dengan Python 3.10
# di image kita.
FLASH_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl"

echo "=== Langkah 1: dependensi server vLLM (perintah resmi) ==="
if paddleocr install_genai_server_deps vllm; then
    echo "=== Berhasil lewat jalur resmi ==="
else
    echo
    echo "=== Gagal. Kemungkinan FlashAttention butuh nvcc untuk kompilasi. ==="
    echo "=== Mencoba wheel siap pakai... ==="

    # vLLM menarik torch. Wheel FlashAttention di atas dibangun untuk
    # torch 2.8 + cu128, jadi torch-nya disamakan lebih dulu.
    python3 -m pip install --no-cache-dir "torch==2.8.*" \
        --index-url https://download.pytorch.org/whl/cu128 || true

    python3 -m pip install --no-cache-dir "$FLASH_URL" || {
        echo "!!! Gagal memasang FlashAttention juga."
        echo "!!! Kirim seluruh output di atas."
        exit 1
    }

    echo "=== Mengulangi pemasangan dependensi server... ==="
    paddleocr install_genai_server_deps vllm || {
        echo "!!! Masih gagal. Kirim seluruh output di atas."
        exit 1
    }
fi

echo
echo "=== Langkah 2: verifikasi ==="
python3 -c "import vllm; print('vLLM versi', vllm.__version__)" || exit 1

echo
echo "=== Selesai. Sekarang jalankan server: ==="
echo "    nohup paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B \\"
echo "        --host 0.0.0.0 --port 8118 --backend vllm > /workspace/vllm.log 2>&1 &"
