# paddleocr-runpod

Image PaddleOCR PP-StructureV3 siap pakai untuk Runpod, di-build otomatis
ke GitHub Container Registry. Dipakai sebagai *one-click template*.

## Isi

| File | Fungsi |
|---|---|
| `Dockerfile` | PaddlePaddle GPU cu126 + `paddleocr[doc-parser]` |
| `run_ocr.py` | OCR buku scan per rentang halaman, output Markdown + JSON, dengan checkpoint |
| `.github/workflows/build.yml` | Build & push ke GHCR setiap kali `main` berubah |

## Model tidak di-bake ke image - dan kenapa

`libpaddle.so` ter-link ke `libcuda.so.1`, yaitu library **driver** NVIDIA.
Runner GitHub Actions tidak punya GPU maupun driver, sehingga `import paddle`
selalu gagal di CI. Karena `from paddleocr import PPStructureV3` juga meng-import
paddle, **warm-up model tidak mungkin dijalankan saat build**.

Maka model diunduh saat boot pertama pod, ke `PADDLE_PDX_CACHE_HOME`
(`/workspace/.paddlex`).

## Network Volume TIDAK diperlukan

Cache model PP-StructureV3 berukuran **~2,8 GB** (UniMERNet sendiri 1,5 GB).
Ini bukan data yang perlu dipersist - bisa dibuat ulang kapan saja.

```
Network volume (minimum 10 GB)  = $0.70/bulan, dibayar terus
Unduh ulang model tiap sesi     = ~1,5 menit, ~$0.006/sesi
Break-even                      = ~108 sesi/bulan
```

Untuk pemakaian sporadis, mengunduh ulang jauh lebih murah. **Jangan sewa
volume hanya untuk cache model.**

Pasang Network Volume hanya kalau kamu butuh menyimpan PDF dan hasil antar-sesi
(mis. mengerjakan buku per bab selama beberapa hari). Saat itu cache model bisa
ikut ke volume **tanpa biaya tambahan**, karena volumenya sudah dibayar untuk
alasan lain. Untuk itu, mount volume ke `/workspace` - path cache sudah menunjuk
ke sana.

## Optimasi yang belum diuji: ganti model formula

Model formula default (UniMERNet, 1530 MB, 1312 ms/formula) tampak kalah dari
alternatif yang lebih kecil, menurut tabel benchmark resmi:

| Model | Ukuran | GPU | En-BLEU | Zh-BLEU |
|---|---|---|---|---|
| UniMERNet | 1530 MB | 1312 ms | 85,91 | 43,50 |
| PP-FormulaNet-S | 224 MB | 182 ms | 87,00 | 45,71 |
| PP-FormulaNet_plus-M | 592 MB | 1040 ms | 91,45 | 89,76 |

Cek dulu apakah `PPStructureV3` menerima parameter pemilihan model:

```bash
python3 -c "from paddleocr import PPStructureV3; help(PPStructureV3)" | grep -i formula
```

Kalau ya, ganti ke `PP-FormulaNet_plus-M` akan memotong cache model dari
~2,8 GB ke ~1,5 GB dan mempercepat inferensi secara signifikan.

## Cara pakai

1. Setiap push ke `main` (yang menyentuh Dockerfile / run_ocr.py / workflow)
   otomatis membangun image ke:
   `ghcr.io/silver3926/paddleocr-runpod:latest`
2. Buka *Packages* repo ini di GitHub, ubah visibility paket menjadi **Public** -
   kalau tidak, Runpod tidak bisa menarik image tanpa kredensial.
3. Di Runpod, buat template dengan image di atas. Container disk **20 GB**
   (image sekitar 7 GB, ditambah cache model 2,8 GB -> 20 GB lega).
4. Boot pertama: biarkan model terunduh (~1-2 menit). Cek dulu GPU terdeteksi:

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
JSON-nya menyimpan bounding box dan skor keyakinan per baris - berguna untuk
menandai halaman yang perlu ditinjau manual.

**Unduh hasilnya sebelum terminate pod** - container disk hilang saat pod berhenti.

## Catatan

- `SRC_PDF` dan `OUT_DIR` bisa dioverride lewat env var.
- `paddlepaddle-gpu` di pin ke `3.2.0` (cu126). Jika driver host lebih lama
  dari 550.54.14, ganti ke indeks `cu118`.
- Progres disimpan di `checkpoint.json`, jadi job boleh dihentikan dan dilanjutkan.
- Ukuran komponen terukur: base ~0,75 GB, apt +18 MB,
  `paddlepaddle-gpu` + `nvidia-*` ~3,93 GB. Total image sekitar 7 GB.
