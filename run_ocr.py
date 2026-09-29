#!/usr/bin/env python3
"""
OCR PDF buku scan -> Markdown, memakai PaddleOCR-VL-1.6.

Contoh pemakaian:
    # inferensi lokal (VLM berjalan di proses ini)
    python3 /opt/paddleocr/run_ocr.py --pdf /workspace/data/buku.pdf --pages 1 20

    # VLM dilayani server vLLM terpisah (dianjurkan)
    python3 /opt/paddleocr/run_ocr.py --pdf buku.pdf --pages 1 20 \\
        --vl-rec-backend vllm-server --vl-rec-server-url http://127.0.0.1:8118/v1

--pdf WAJIB dan tidak punya nilai default. Seluruh argumen divalidasi
sebelum pipeline dimuat, jadi path yang salah gagal seketika — bukan
setelah 1,9 GB model terlanjur diunduh.

Kenapa dipecah per bagian (--chunk)?
  - Ada checkpoint: crash di bagian 8 tidak mengulang bagian 1-7.
  - Tapi penggabungan tabel antar-halaman (merge_tables) dan penyusunan
    ulang heading (relevel_titles) hanya berlaku DI DALAM satu bagian.
    Pakai --chunk 0 kalau mau penggabungan seluruh buku tanpa checkpoint.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from paddleocr import PaddleOCRVL


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="OCR PDF buku scan menjadi Markdown (PaddleOCR-VL-1.6).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--pdf", required=True, metavar="PATH",
        help="berkas PDF yang diproses (WAJIB, tidak ada default)",
    )
    ap.add_argument(
        "--pages", nargs=2, type=int, metavar=("START", "END"),
        help="rentang halaman, 1-indexed dan inklusif (default: semua halaman)",
    )
    ap.add_argument(
        "--chunk", type=int, default=50, metavar="N",
        help="halaman per bagian (default: 50). 0 = seluruh PDF sekali jalan",
    )
    ap.add_argument(
        "--out", metavar="DIR",
        help="folder hasil (default: <nama-pdf>-ocr di sebelah PDF-nya)",
    )
    ap.add_argument(
        "--vl-rec-backend", default=None, metavar="NAMA",
        help="backend VLM: vllm-server | sglang-server | fastdeploy-server. "
             "Kosong = inferensi lokal",
    )
    ap.add_argument(
        "--vl-rec-server-url", default=None, metavar="URL",
        help="URL server VLM, mis. http://127.0.0.1:8118/v1",
    )
    ap.add_argument(
        "--no-queues", action="store_true",
        help="matikan use_queues (jalur paralel). Pakai kalau pipeline menggantung",
    )
    return ap.parse_args(argv)


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    #  Validasi masukan SEBELUM memuat pipeline.
    # ------------------------------------------------------------------
    src = Path(args.pdf).expanduser().resolve()
    if not src.exists():
        sys.exit(f"ERROR: PDF tidak ditemukan: {src}")
    if not src.is_file():
        sys.exit(f"ERROR: bukan berkas: {src}")
    if src.suffix.lower() != ".pdf":
        sys.exit(f"ERROR: bukan berkas PDF: {src}")

    try:
        doc = fitz.open(str(src))
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"ERROR: gagal membuka PDF: {exc}")

    total = doc.page_count
    start, end = args.pages if args.pages else (1, total)
    if not (1 <= start <= end <= total):
        sys.exit(f"ERROR: rentang {start}-{end} di luar batas 1-{total}")

    out = (
        Path(args.out).expanduser().resolve()
        if args.out
        else src.parent / f"{src.stem}-ocr"
    )
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint.json"
    tmp = Path("/tmp/paddleocr-chunks")
    tmp.mkdir(parents=True, exist_ok=True)

    chunk = args.chunk
    ranges = (
        [(s, min(s + chunk - 1, end)) for s in range(start, end + 1, chunk)]
        if chunk > 0
        else [(start, end)]
    )

    print(f"PDF     : {src}  ({total} halaman)")
    print(f"Rentang : {start}-{end}  -> {len(ranges)} bagian")
    print(f"Output  : {out}")
    print(f"Cache   : {os.environ.get('PADDLE_PDX_CACHE_HOME', '(default)')}")
    print(f"VLM     : {args.vl_rec_backend or 'lokal'}"
          + (f" @ {args.vl_rec_server_url}" if args.vl_rec_server_url else ""))
    print("\nMenyiapkan pipeline (unduhan model pertama kali beberapa menit)...", flush=True)

    # use_doc_unwarping penting untuk buku scan: menghilangkan distorsi
    # lengkung di area punggung buku.
    kwargs = dict(
        pipeline_version="v1.6",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_chart_recognition=False,
        use_seal_recognition=False,
    )
    if args.vl_rec_backend:
        kwargs["vl_rec_backend"] = args.vl_rec_backend
    if args.vl_rec_server_url:
        kwargs["vl_rec_server_url"] = args.vl_rec_server_url
    if args.no_queues:
        kwargs["use_queues"] = False

    pipeline = PaddleOCRVL(**kwargs)
    print("Pipeline siap.\n", flush=True)

    done = json.loads(ckpt.read_text()) if ckpt.exists() else {}
    started = time.time()

    for cs, ce in ranges:
        key = f"{cs}-{ce}"
        n = ce - cs + 1
        if key in done:
            print(f"  [{key}] sudah selesai, dilewati", flush=True)
            continue

        sub = fitz.open()
        sub.insert_pdf(doc, from_page=cs - 1, to_page=ce - 1)
        sub_path = tmp / f"chunk-{cs:04d}-{ce:04d}.pdf"
        sub.save(sub_path)
        sub.close()

        print(f"  [{key}] mulai, {n} halaman...", flush=True)
        t0 = time.time()

        # predict_iter() mengembalikan generator sehingga progres bisa
        # dicetak per halaman. predict() mengembalikan list yang baru
        # tersedia setelah SELURUH bagian selesai.
        if hasattr(pipeline, "predict_iter"):
            stream = pipeline.predict_iter(str(sub_path))
        else:
            stream = pipeline.predict(str(sub_path))

        pages = []
        for i, res in enumerate(stream, 1):
            pages.append(res)
            el = time.time() - t0
            print(
                f"      {i}/{n}  {el:6.0f}s  (~{el / i:4.1f}s/hal,"
                f" sisa ~{(el / i) * (n - i) / 60:4.1f} mnt)",
                flush=True,
            )

        merged = list(
            pipeline.restructure_pages(
                pages,
                merge_tables=True,       # gabungkan tabel antar halaman
                relevel_titles=True,     # susun ulang heading multi-level
                concatenate_pages=True,  # jadikan satu dokumen
            )
        )

        bagian = out / f"bagian-{cs:04d}-{ce:04d}"
        bagian.mkdir(exist_ok=True)
        for res in merged:
            res.save_to_markdown(save_path=str(bagian))
            res.save_to_json(save_path=str(bagian))

        done[key] = round(time.time() - t0, 1)
        ckpt.write_text(json.dumps(done))
        sub_path.unlink(missing_ok=True)

        print(
            f"  [{key}] selesai {done[key]:.0f}s  ({done[key] / n:.1f}s/halaman)"
            f"  -> {bagian}",
            flush=True,
        )

    if all(f"{cs}-{ce}" in done for cs, ce in ranges):
        parts = []
        for cs, ce in ranges:
            md = sorted((out / f"bagian-{cs:04d}-{ce:04d}").glob("*.md"))
            if md:
                parts.append(md[0].read_text(encoding="utf-8"))
        if parts:
            target = out / f"{src.stem}-lengkap.md"
            target.write_text("\n\n".join(parts), encoding="utf-8")
            print(f"\nDigabung -> {target}")

    elapsed = time.time() - started
    pages_done = sum(ce - cs + 1 for cs, ce in ranges if f"{cs}-{ce}" in done)
    rate = elapsed / pages_done if pages_done else 0
    print(
        f"\nSELESAI. {len(done)} bagian, {pages_done} halaman, "
        f"{elapsed / 60:.1f} menit ({rate:.1f}s/halaman).",
        flush=True,
    )


if __name__ == "__main__":
    main()
