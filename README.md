# paddleocr-runpod

Image PaddleOCR PP-StructureV3 siap pakai untuk Runpod, di-build otomatis
ke GitHub Container Registry. Dipakai sebagai *one-click template*.

## Isi

| File | Fungsi |
|---|---|
| `Dockerfile` | PaddlePaddle GPU cu126 + `paddleocr[doc-parser]` + model ter-bake |
| `run_ocr.py` | OCR buku scan per rentang halaman, output Markdown + JSON, dengan checkpoint |
| `.github/workflows/build.yml` | Build & push ke GHCR setiap kali `main` berubah |

## Cara pakai

1. Setiap push ke `main` otomatis membangun image dan mendorongnya ke:
   `ghcr.io/silver3926/paddleocr-runpod:latest`
2. Buka halaman *Packages* repo ini di GitHub, lalu ubah visibility paket
   menjadi **Public** — kalau tidak, Runpod tidak bisa menarik image tanpa kredensial.
3. Di Runpod, buat template dengan image di atas.

## Menjalankan OCR di dalam pod

```bash
tmux new -s ocr

# cek seberapa tajam scan-nya (penting untuk formula)
pdfimages -list /workspace/data/buku.pdf | head -30

# kalibrasi 20 halaman tersulit dulu
python3 /opt/paddleocr/run_ocr.py 100 119

# lanjut kalau kualitas sudah oke
python3 /opt/paddleocr/run_ocr.py 1 45
python3 /opt/paddleocr/run_ocr.py 46 90
```

Hasil ada di `/workspace/out/buku-kuliah/` sebagai `pXXXX.md` dan `pXXXX.json`.
JSON-nya menyimpan bounding box dan skor keyakinan per baris — berguna untuk
menandai halaman yang perlu ditinjau manual.

## Catatan

- `SRC_PDF` dan `OUT_DIR` bisa dioverride lewat env var.
- Format `paddleocr` di pin ke `3.2.0` (cu126). Jika driver host lebih lama
dari 550.54.14, ganti ke wheel `cu118`.
- Progres disimpan di `checkpoint.json`, jadi job boleh dihentikan dan dilanjutkan.
