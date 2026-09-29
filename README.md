# paddleocr-runpod

Image PaddleOCR PP-StructureV3 siap pakai untuk Runpod, di-build otomatis
ke GitHub Container Registry. Dipakai sebagai *one-click template*.

## Isi

| File | Fungsi |
|---|---|
| `Dockerfile` | PaddlePaddle GPU cu126 + `paddleocr[doc-parser]` |
| `run_ocr.py` | OCR buku scan per rentang halaman, output Markdown + JSON, dengan checkpoint |
| `.github/workflows/build.yml` | Build & push ke GHCR setiap kali `main` berubah |

## Model tidak di-bake ke image — dan kenapa

`libpaddle.so` ter-link ke `libcuda.so.1`, yaitu library **driver** NVIDIA.
Runner GitHub Actions tidak punya GPU maupun driver, sehingga `import paddle`
selalu gagal di CI. Karena `from paddleocr import PPStructureV3` juga meng-import
paddle, **warm-up model tidak mungkin dijalankan saat build**.

Maka model diunduh saat boot pertama pod, ke `PADDLE_PDX_CACHE_HOME`.
Mount Network Volume di `/workspace` supaya unduhan itu hanya terjadi sekali.

Konsekuensi: pod **wajib** memakai Network Volume, karena model tersimpan di sana.
Alternatifnya, model akan diunduh ulang setiap pod baru (sekitar 3-5 menit).

## Cara pakai

1. Setiap push ke `main` otomatis membangun image ke:
   `ghcr.io/silver3926/paddleocr-runpod:latest`
2. Buka *Packages* repo ini di GitHub, ubah visibility paket menjadi **Public** —
   kalau tidak, Runpod tidak bisa menarik image tanpa kredensial.
3. Di Runpod, buat template dengan image di atas, mount Network Volume ke
   `/workspace`, dan set container disk 20 GB (image sekitar 7 GB, jadi 20 GB lega).
4. Boot pertama: biarkan model terunduh. Cek dulu bahwa GPU terdeteksi:

```bash
python3 -c "import paddle; print(paddle.__version__); paddle.utils.run_check()"
```

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
- Format `paddlepaddle-gpu` di pin ke `3.2.0` (cu126). Jika driver host lebih
  lama dari 550.54.14, ganti ke indeks `cu118`.
- Progres disimpan di `checkpoint.json`, jadi job boleh dihentikan dan dilanjutkan.
- Ukuran komponen terukur: base ~0,75 GB, apt +18 MB,
  `paddlepaddle-gpu` + `nvidia-*` ~3,93 GB. Total image sekitar 7 GB.
