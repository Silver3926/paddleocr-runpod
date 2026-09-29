"""
OCR buku scan -> Markdown, memakai PaddleOCR-VL-1.6.

Pakai:
    python3 /opt/paddleocr/run_ocr.py              # semua halaman, per 50 halaman
    python3 /opt/paddleocr/run_ocr.py 1 45         # halaman 1-45 saja
    python3 /opt/paddleocr/run_ocr.py 1 45 10      # halaman 1-45, per 10 halaman

Env var:
    SRC_PDF      path PDF     (default /workspace/data/buku.pdf)
    OUT_DIR      folder hasil (default /workspace/out/buku-kuliah)
    CHUNK_PAGES  halaman per bagian (default 50; 0 = seluruh PDF sekali jalan)

Kenapa dipecah per bagian (chunk)?
  - Ada checkpoint: crash di bagian 8 tidak mengulang bagian 1-7.
  - Tapi penggabungan tabel antar-halaman (merge_tables) dan
    penyusunan ulang heading (relevel_titles) hanya berlaku DI DALAM
    satu bagian. Set CHUNK_PAGES=0 kalau kamu mau penggabungan
    seluruh buku dan tidak keberatan kehilangan checkpoint.

Catatan: pemanggilan PERTAMA mengunduh model (~2,2 GB, beberapa menit).
Model disimpan di PADDLE_PDX_CACHE_HOME, jadi kalau Network Volume
ter-mount, unduhan itu hanya sekali.
"""
import json
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from paddleocr import PaddleOCRVL

SRC = os.environ.get("SRC_PDF", "/workspace/data/buku.pdf")
OUT = Path(os.environ.get("OUT_DIR", "/workspace/out/buku-kuliah"))
CKPT = OUT / "checkpoint.json"
CHUNK = int(os.environ.get("CHUNK_PAGES", "50"))
TMP = Path("/tmp/paddleocr-chunks")

OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

print(f"Model cache : {os.environ.get('PADDLE_PDX_CACHE_HOME', '(default)')}")
print("Menyiapkan pipeline (unduhan model pertama kali bisa beberapa menit)...", flush=True)

# use_doc_unwarping penting untuk buku scan: menghilangkan distorsi
# lengkung di area punggung buku. PaddleOCR-VL tidak punya modul formula
# terpisah — teks, tabel, dan formula ditangani model VLM yang sama.
pipeline = PaddleOCRVL(
    pipeline_version="v1.6",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_chart_recognition=False,
    use_seal_recognition=False,
)
print("Pipeline siap.\n", flush=True)

done = json.loads(CKPT.read_text()) if CKPT.exists() else {}
doc = fitz.open(SRC)
TOTAL = doc.page_count

START = int(sys.argv[1]) if len(sys.argv) > 1 else 1
END = int(sys.argv[2]) if len(sys.argv) > 2 else TOTAL
if len(sys.argv) > 3:
    CHUNK = int(sys.argv[3])

ranges = (
    [(s, min(s + CHUNK - 1, END)) for s in range(START, END + 1, CHUNK)]
    if CHUNK > 0
    else [(START, END)]
)

print(f"PDF     : {SRC} ({TOTAL} halaman)")
print(f"Rentang : {START}-{END}  ({len(ranges)} bagian)", flush=True)

started = time.time()
for cs, ce in ranges:
    key = f"{cs}-{ce}"
    if key in done:
        print(f"  [{key}] sudah selesai, dilewati", flush=True)
        continue

    # Potong PDF menjadi satu bagian. Pipeline memproses PDF secara batch
    # dengan use_queues=True, sehingga render halaman, layout analysis, dan
    # inferensi VLM berjalan asinkron — jauh lebih efisien daripada
    # memanggil predict() satu halaman demi satu halaman.
    sub = fitz.open()
    sub.insert_pdf(doc, from_page=cs - 1, to_page=ce - 1)
    sub_path = TMP / f"chunk-{cs:04d}-{ce:04d}.pdf"
    sub.save(sub_path)
    sub.close()

    t0 = time.time()
    pages = list(pipeline.predict(str(sub_path)))
    merged = list(
        pipeline.restructure_pages(
            pages,
            merge_tables=True,      # gabungkan tabel yang terpotong antar halaman
            relevel_titles=True,    # susun ulang hierarki heading multi-level
            concatenate_pages=True, # jadikan satu dokumen
        )
    )

    bagian = OUT / f"bagian-{cs:04d}-{ce:04d}"
    bagian.mkdir(exist_ok=True)
    for res in merged:
        res.save_to_markdown(save_path=str(bagian))
        res.save_to_json(save_path=str(bagian))

    done[key] = round(time.time() - t0, 1)
    CKPT.write_text(json.dumps(done))
    sub_path.unlink(missing_ok=True)
    print(f"  [{key}] {done[key]:.0f}s", flush=True)

# Gabungkan semua bagian jadi satu dokumen kalau semuanya sudah selesai.
if all(f"{cs}-{ce}" in done for cs, ce in ranges):
    parts = []
    for cs, ce in ranges:
        md = sorted((OUT / f"bagian-{cs:04d}-{ce:04d}").glob("*.md"))
        if md:
            parts.append(md[0].read_text(encoding="utf-8"))
    if parts:
        target = OUT / "buku-lengkap.md"
        target.write_text("\n\n".join(parts), encoding="utf-8")
        print(f"\nDigabung -> {target}")

print(f"\nSELESAI. {len(done)} bagian, {(time.time() - started) / 60:.1f} menit.", flush=True)
