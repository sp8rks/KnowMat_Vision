from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def _read_xy_rows(classification_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with classification_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("classification") or "").strip() == "xy_plots":
                rows.append(row)
    return rows


def _write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "queue_image",
                "source_pdf",
                "page_index",
                "source_image_index",
                "classified_image_file",
                "accepted_subplot_count",
                "context_excerpt",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_instructions(out_dir: Path, n_images: int) -> None:
    text = f"""# WebPlotDigitizer Queue

Prepared {n_images} xy-plot figure images for manual/assisted digitization.

## 1) Start a static server from `week2`
Example:
```bash
npx serve .
```

## 2) Open WebPlotDigitizer wrapper
In browser, open:
`http://localhost:3000/node_modules/@insilicall/img2data/index.html`

## 3) Load queue images
Use files from:
`{out_dir.name}/images/`

## 4) Track provenance
Use:
`{out_dir.name}/manifest.csv`
to map each queued image back to PDF/page/image metadata.
"""
    (out_dir / "README_webplotdigitizer.md").write_text(text, encoding="utf-8")


def build_wpd_queue(run_dir: Path, out_dir: Path, max_images: int | None = None) -> dict[str, str | int]:
    classification_csv = run_dir / "csv" / "figure_classification.csv"
    if not classification_csv.exists():
        raise FileNotFoundError(f"Missing classification CSV: {classification_csv}")

    rows = _read_xy_rows(classification_csv)
    if max_images is not None:
        rows = rows[: max(0, int(max_images))]

    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    copied = 0

    for idx, row in enumerate(rows):
        src = Path(str(row.get("classified_image_file") or ""))
        if not src.exists():
            continue
        name = f"xy_{idx:05d}{src.suffix.lower() or '.png'}"
        dst = images_dir / name
        shutil.copy2(src, dst)
        manifest_rows.append(
            {
                "queue_image": str(dst),
                "source_pdf": str(row.get("pdf_file") or ""),
                "page_index": str(row.get("page_index") or ""),
                "source_image_index": str(row.get("source_image_index") or ""),
                "classified_image_file": str(src),
                "accepted_subplot_count": str(row.get("accepted_subplot_count") or ""),
                "context_excerpt": str(row.get("context_excerpt") or ""),
            }
        )
        copied += 1

    manifest_path = out_dir / "manifest.csv"
    _write_manifest(manifest_path, manifest_rows)
    _write_instructions(out_dir, copied)

    return {
        "run_dir": str(run_dir),
        "classification_csv": str(classification_csv),
        "queue_dir": str(out_dir),
        "queue_images_dir": str(images_dir),
        "queue_images": copied,
        "manifest_csv": str(manifest_path),
        "instructions_md": str(out_dir / "README_webplotdigitizer.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WebPlotDigitizer queue from classified xy figures")
    parser.add_argument("--run-dir", required=True, help="Pipeline run output directory")
    parser.add_argument("--out", required=True, help="Output directory for WebPlotDigitizer queue")
    parser.add_argument("--max-images", type=int, default=None, help="Optional max number of queued images")
    args = parser.parse_args()

    report = build_wpd_queue(run_dir=Path(args.run_dir), out_dir=Path(args.out), max_images=args.max_images)
    print(report)


if __name__ == "__main__":
    main()
