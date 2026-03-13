from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="week2 x-y extraction and synthetic augmentation pipeline")
    parser.add_argument("--pdf-dir", default="../100 articles", help="Directory containing source PDFs")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--max-pdfs", type=int, default=None, help="Optional limit on number of PDFs")
    parser.add_argument(
        "--max-figures-per-pdf",
        type=int,
        default=None,
        help="Optional limit on extracted plot figures per PDF",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    args = parser.parse_args()

    report = run_pipeline(
        pdf_dir=Path(args.pdf_dir),
        out_dir=Path(args.out),
        max_pdfs=args.max_pdfs,
        max_figures_per_pdf=args.max_figures_per_pdf,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
