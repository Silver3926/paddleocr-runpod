# paddleocr-runpod

Image **PaddleOCR-VL-1.6** siap pakai untuk Runpod, di-build otomatis ke
GitHub Container Registry. Dipakai sebagai *one-click template*.

## Kenapa PaddleOCR-VL-1.6, bukan PP-StructureV3

Pada OmniDocBench v1.5 (benchmark yang sama, jadi bisa dibandingkan langsung):

| Metrik | PP-StructureV3 | PaddleOCR-VL |
|---|---|---|
| Overall | 86,73 | **92,86** |
| Table TEDS | 81,68 | **90,89** |
| Formula CDM | 85,79 | **91,22** |
| Text Edit (makin kecil makin baik) | 0,073 | **0,035** |

Selisih terbesar ada di tabel dan formula — persis yang dibutuhkan buku
kuliah hasil scan.

Dua hal yang dimiliki PaddleOCR-VL dan tidak dimiliki PP-StructureV3:

1. **Penggabungan antar-halaman** — `restructure_pages(merge_tables=True,
   relevel_titles=True)` menggabungkan tabel yang terpotong antar halaman
   dan menyusun ulang hierarki heading bab/sub-bab.
2. **Pemrosesan PDF secara batch** — `use_queues=True` menjalankan render
   halaman, layout analysis, dan inferensi VLM secara asinkron. Loop per
   halaman justru mematikan pipelining ini.

## Isi

| File | Fungsi |
|---|---|
| `Dockerfile` | PaddlePaddle GPU 3.2.1 cu126 + `paddleocr[doc-parser]` + jupyterlab + font |
| `run_ocr.py` | OCR per bagian dengan checkpoint, output Markdown + JSON |
| `.github/workflows/build.yml` | Build & push ke GHCR saat `main` berubah |

## Model tidak di-bake ke image — dan kenapa

`libpaddle.so` ter-link ke `libcuda.so.1`, library **driver** NVIDIA. Runner
GitHub Actions tidak punya GPU maupun driver, sehingga `import paddle` selalu
gagal di CI. Karena `PaddleOCRVL` juga meng-import paddle, **warm-up model tidak
mungkin dijalankan saat build**. Model diunduh saat boot pertama pod.

## JupyterLab: kenapa sempat gagal

`/start.sh` milik Runpod menyalakan JupyterLab lewat:

```bash
nohup python3 -m jupyter lab ... &> /jupyter.log &
```

Di base image ini `python3` adalah **3.10.12** (tempat Paddle dipasang),
sementara entrypoint `jupyter` menunjuk ke **Python 3.12**. Jadi
`python3 -m jupyter` gagal seketika — tapi `/start.sh` tetap mencetak
`Jupyter Lab started` 0,3 ms kemudian, dan pesan errornya dibuang ke
`/jupyter.log` sehingga tidak terlihat di log container.

Perbaikannya: `jupyterlab` dipasang ke interpreter yang sama dengan `python3`.
Efek sampingnya menguntungkan — notebook ikut berjalan di 3.10, sehingga
`import paddle` di dalam notebook juga bekerja.

Ada juga penjaga di build: `RUN python3 -m jupyter --version`, yang langsung
menggagalkan build kalau suatu saat pemasangan ini lepas lagi.

> Kalau proxy menampilkan *"Waiting for service to respond"*, periksa
> `/jupyter.log` — di situlah pesan sebenarnya berada.

## Network Volume TIDAK diperlukan

Cache model PaddleOCR-VL sekitar **2,2 GB** (PP-DocLayoutV3 + preprocessor +
model VLM 0.9B). Tidak ada UniMERNet (1,5 GB) maupun SLANeXt (351 MB x2).

```
Network volume (minimum 10 GB)  = $0.70/bulan, dibayar terus
Unduh ulang model tiap sesi     = ~1-2 menit, ~$0.006/sesi
Break-even                      = ~100 sesi/bulan
```

Pasang Network Volume hanya kalau kamu butuh menyimpan PDF dan hasil
antar-sesi. Saat itu cache model bisa ikut ke volume tanpa biaya tambahan —
SSH. Kalau perlu `scp`/`rsync` untuk file besar, daftarkan SSH public key di
Settings Runpod lebih dulu (tanpa itu env `PUBLIC_KEY` berisi `null` dan SSH
tidak bisa dipakai).

## Menjalankan OCR di dalam pod

```bash
tmux new -s ocr

# cek seberapa tajam scan-nya (penting untuk formula)
pdfimages -list /workspace/data/buku.pdf | head -30

# kalibrasi 20 halaman tersulit dulu — sekaligus ukur kecepatan
python3 /opt/paddleocr/run_ocr.py 100 119

# lanjut kalau kualitas sudah oke
python3 /opt/paddleocr/run_ocr.py 1 45
```

Hasil per bagian ada di `/workspace/out/buku-kuliah/bagian-xxxx-yyyy/` sebagai
`.md` dan `.json`. Setelah semua bagian selesai, script otomatis menggabungkannya
menjadi `/workspace/out/buku-kuliah/buku-lengkap.md`.

**Unduh hasilnya sebelum terminate pod** — container disk hilang saat pod berhenti.

## Catatan penting

- **Kecepatan inferensi lokal belum terukur.** Dokumentasi PaddleOCR memperingatkan
  bahwa inferensi lokal tanpa server vLLM ditujukan untuk validasi cepat, dan
  kecepatannya belum tentu memadai untuk produksi. Ukur dulu pada 20 halaman.
  Kalau terlalu lambat, langkah berikutnya adalah menjalankan server vLLM
  (backend `vllm-server`) sebagai proses kedua di pod yang sama.
- `CHUNK_PAGES` (default 50) menentukan ukuran bagian. Penggabungan tabel dan
  heading hanya berlaku di dalam satu bagian. Set `CHUNK_PAGES=0` untuk
  memproses seluruh PDF sekaligus (penggabungan penuh, tanpa checkpoint).
- `PADDLE_PDX_CACHE_HOME` default `/workspace/.paddlex`.
- `SRC_PDF`, `OUT_DIR`, dan `CHUNK_PAGES` bisa dioverride lewat env var.
- `paddlepaddle-gpu` di pin ke `3.2.1` (cu126). Driver host perlu >= 550.54.14.
  Terbukti jalan di host CUDA 13.2.
- Progres disimpan di `checkpoint.json` per bagian.
- Ukuran komponen terukur: base ~0,75 GB, apt termasuk font beberapa ratus MB,
  `paddlepaddle-gpu` + `nvidia-*` ~3,93 GB. Total image sekitar 7-8 GB.
