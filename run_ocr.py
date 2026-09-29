"""
OCR buku scan -> Markdown, memakai PP-StructureV3.

Pakai:
    python3 /opt/paddleocr/run_ocr.py 1 45     # halaman 1-45 saja
    python3 /opt/paddleocr/run_ocr.py          # semua halaman

Env var:
    SRC_PDF   path PDF     (default /workspace/data/buku.pdf)
    OUT_DIR   folder hasil (default /workspace/out/buku-kuliah)

Catatan: pemanggilan PERTAMA akan mengunduh model PP-StructureV3
(beberapa menit). Model disimpan di PADDLE_PDX_CACHE_HOME, jadi
kalau Network Volume ter-mount, unduhan itu hanya sekali.

Aman dihentikan kapan saja: progres dicatat per halaman di
checkpoint.json, jadi menjalankan ulang hanya memproses halaman
yang belum selesai.
"""
import json
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from paddleocr import PPStructureV3

SRC = os.environ.get("SRC_PDF", "/workspace/data/buku.pdf")
OUT = Path(os.environ.get("OUT_DIR", "/workspace/out/buku-kuliah"))
CKPT = OUT / "checkpoint.json"
OUT.mkdir(parents=True, exist_ok=True)

cache = os.environ.get("PADDLE_PDX_CACHE_HOME", "(default)")
print(f"Model cache : {cache}", flush=True)
print("Menyiapkan pipeline (unduhan model pertama kali bisa beberapa menit)...", flush=True)

# use_doc_unwarping sangat penting untuk buku scan: menghilangkan
# distorsi lengkung di area punggung buku.
# use_chart_recognition dimatikan: model VLM, berat, jarang diperlukan.
pipeline = PPStructureV3(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_formula_recognition=True,
    use_table_recognition=True,
    use_chart_recognition=False,
    use_seal_recognition=False,
    engine="paddle",  # wajib "paddle"; engine transformers tidak dukung formula
)
print("Pipeline siap.\n", flush=True)

done = json.loads(CKPT.read_text()) if CKPT.exists() else {}
doc = fitz.open(SRC)

START = int(sys.argv[1]) if len(sys.argv) > 1 else 1
END = int(sys.argv[2]) if len(sys.argv) > 2 else doc.page_count

print(f"PDF     : {SRC} ({doc.page_count} halaman)", flush=True)
print(f"Rentang : {START}-{END}", flush=True)
print(f"Output  : {OUT}", flush=True)
print(f"Sudah   : {len(done)} halaman", flush=True)

started = time.time()
for pno in range(START, END + 1):
    if str(pno) in done:
        continue

    tmp = f"/tmp/p{pno}.png"
    doc[pno - 1].get_pixmap(dpi=300).save(tmp)  # 300 DPI: minimum untuk formula

    t0 = time.time()
    for res in pipeline.predict(tmp):
        res.save_to_markdown(save_path=str(OUT / f"p{pno:04d}"))
        res.save_to_json(save_path=str(OUT / f"p{pno:04d}"))
    done[str(pno)] = round(time.time() - t0, 2)

    CKPT.write_text(json.dumps(done))
    os.remove(tmp)
    print(f"  p{pno:04d}  {done[str(pno)]:>6.1f}s", flush=True)

elapsed = time.time() - started
print(f"\nSELESAI. {len(done)} halaman tersimpan, {elapsed / 60:.1f} menit.", flush=True)
