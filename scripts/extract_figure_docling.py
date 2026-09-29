#!/usr/bin/env python3
"""
Extract figures and their captions from scientific PDFs with Docling.

Docling runs a layout model over every page, so figures are found as figures
(raster or vector, single or multi-panel) and captions are linked to them by the
document model rather than by guessing from geometry.

Only figure/caption pairs are output: a picture with no caption (logo, icon,
decoration) and a caption with no picture are both dropped. On top of Docling's
own linking this script:

1. **Recovers unlinked captions** — a ``Fig. N`` caption Docling detected but did
   not attach to any picture is given to the nearest caption-less picture on the
   same page (same column, close by, no body text in between).
2. **Merges split panels** — a caption-less picture directly next to a captioned
   one (no text or caption in between) is merged into it, so a multi-panel figure
   Docling split into pieces comes out as one image.
3. **Drops non-figures** — pictures nested inside another picture, and pictures
   whose caption is a ``Table N`` caption (tables embedded as images).

Output matches ``extract_figure.py`` so the two can be compared directly:
``<out_dir>/<pdf_stem>/figures.json`` (figure number, caption, 0-based page,
bbox in PDF points with top-left origin, image file) and one PNG per figure.

Image quality: Docling is used only to *find* figures and captions. Each PNG is
rendered straight from the PDF with pypdfium2 (installed with Docling) at
``--dpi`` (default 300), so vector plots stay sharp and raster figures keep their
detail instead of being cut out of a low-resolution layout image.

Setup (separate environment recommended; pulls in PyTorch and layout models):
    pip install docling

Usage:
    python extract_figure_docling.py paper.pdf [more.pdf ...] [-o out_dir] [--dpi 300]
    python extract_figure_docling.py papers/*.pdf --device cuda
    python extract_figure_docling.py papers/*.pdf --no-ocr     # born-digital PDFs only
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel

try:  # docling >= 2.30 moved the accelerator options
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
except ImportError:  # pragma: no cover - older docling
    from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions


# "Fig. 1.", "Figure 2:", "FIG. 3 |", "Fig 1. ...", "Figure S4 -", "Fig. 10a", plus
# Russian "Рисунок 3." / "Рис. 3.". Docling already classified the block as a
# caption, so no delimiter is required after the number (label-only "Fig. 1" is fine).
FIGURE_RE = re.compile(
    r"^\s*(?:Fig\.?|Figure|FIG\.?|FIGURE|Рисунок|Рис\.?)\s*(S?\d+[A-Za-z]?)\b"
)
TABLE_RE = re.compile(r"^\s*(?:Table|TABLE|Таблица)\s*(S?\d+|[IVXLCDM]+)\b")

MAX_CAPTION_DIST = 60.0  # pt; an unlinked caption this close to a picture is its caption
MERGE_GAP = 20.0         # pt; caption-less panels this close to a captioned figure merge
CROP_PAD = 2.0           # pt added around the figure when cropping
NON_BLOCKING = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}


@dataclass
class Box:
    """Rectangle in PDF points, top-left origin."""

    x0: float
    y0: float
    x1: float
    y1: float

    def union(self, o: Box) -> Box:
        return Box(min(self.x0, o.x0), min(self.y0, o.y0),
                   max(self.x1, o.x1), max(self.y1, o.y1))

    def h_overlap(self, o: Box) -> float:
        return max(0.0, min(self.x1, o.x1) - max(self.x0, o.x0))

    def v_overlap(self, o: Box) -> float:
        return max(0.0, min(self.y1, o.y1) - max(self.y0, o.y0))

    def intersects(self, o: Box) -> bool:
        return self.h_overlap(o) > 0 and self.v_overlap(o) > 0

    def contains(self, o: Box, tol: float = 1.0) -> bool:
        return (o.x0 >= self.x0 - tol and o.y0 >= self.y0 - tol
                and o.x1 <= self.x1 + tol and o.y1 <= self.y1 + tol)

    def gap(self, o: Box) -> float:
        """Distance between the two boxes along the axis they are separated on."""
        dx = max(0.0, max(self.x0, o.x0) - min(self.x1, o.x1))
        dy = max(0.0, max(self.y0, o.y0) - min(self.y1, o.y1))
        return max(dx, dy)


@dataclass
class BoundingBox:
    """Figure location on the page, in PDF points (top-left origin)."""

    x0: float
    y0: float
    x1: float
    y1: float
    page: int


@dataclass
class Figure:
    """One extracted figure with its caption."""

    caption: str
    image_data: bytes
    page: int                            # 0-based, same as extract_figure.py
    bbox: Optional[BoundingBox] = None
    figure_id: Optional[str] = None
    figure_number: Optional[str] = None  # e.g. "Figure 1"; None if caption has no label
    image_file: Optional[str] = None     # set when the PNG is written


@dataclass
class _Pic:
    item: object                         # docling PictureItem
    page_no: int                         # docling page number (1-based)
    box: Box
    caption: str = ""


# --------------------------------------------------------------------------- #
# Converter
# --------------------------------------------------------------------------- #
def build_converter(ocr: bool = True, device: str = "auto",
                    threads: int = 8) -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = ocr                     # needed for scanned pages / captions in bitmaps
    opts.do_table_structure = False       # figures only; layout still labels tables
    opts.generate_page_images = False     # layout only; PNGs are rendered by _PageRenderer
    opts.generate_picture_images = False
    opts.accelerator_options = AcceleratorOptions(
        num_threads=threads, device=getattr(AcceleratorDevice, device.upper())
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _box(prov, doc) -> Box:
    page = doc.pages[prov.page_no]
    bb = prov.bbox.to_top_left_origin(page_height=page.size.height)
    return Box(bb.l, bb.t, bb.r, bb.b)


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _blocked(a: Box, b: Box, blockers: List[Box]) -> bool:
    """True if any blocker sits in the gap between ``a`` and ``b``."""
    if a.v_overlap(b) > 0:     # side by side: gap is horizontal
        left, right = (a, b) if a.x1 <= b.x0 else (b, a)
        gap = Box(left.x1, max(a.y0, b.y0), right.x0, min(a.y1, b.y1))
    else:                      # stacked: gap is vertical
        top, bottom = (a, b) if a.y1 <= b.y0 else (b, a)
        gap = Box(max(a.x0, b.x0), top.y1, min(a.x1, b.x1), bottom.y0)
    if gap.x1 <= gap.x0 or gap.y1 <= gap.y0:
        return False
    return any(t.intersects(gap) and not t.intersects(a) and not t.intersects(b)
               for t in blockers)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _collect_pictures(doc) -> List[_Pic]:
    pics = []
    for item in doc.pictures:
        if not item.prov:
            continue
        prov = item.prov[0]
        pics.append(_Pic(item, prov.page_no, _box(prov, doc), _clean(item.caption_text(doc))))
    # Pictures nested inside another picture are panels of it, not separate figures.
    return [
        p for p in pics
        if not any(o is not p and o.page_no == p.page_no and o.box.contains(p.box)
                   and not p.box.contains(o.box) for o in pics)
    ]


def _page_text_boxes(doc) -> dict:
    """{page_no: [(label, Box, self_ref, text)]} for every text item."""
    out: dict = {}
    for t in doc.texts:
        if not t.prov:
            continue
        prov = t.prov[0]
        out.setdefault(prov.page_no, []).append(
            (t.label, _box(prov, doc), t.self_ref, _clean(t.text))
        )
    return out


def _attach_unlinked_captions(doc, pics: List[_Pic], texts: dict) -> None:
    linked = {ref.cref for p in doc.pictures for ref in p.captions}
    linked |= {ref.cref for t in doc.tables for ref in t.captions}
    for page_no, items in texts.items():
        blockers = [b for lab, b, _r, _t in items if lab not in NON_BLOCKING]
        for label, cbox, ref, text in items:
            if label != DocItemLabel.CAPTION or ref in linked or not FIGURE_RE.match(text):
                continue
            best, best_d = None, MAX_CAPTION_DIST
            for p in pics:
                if p.page_no != page_no or p.caption:
                    continue
                if p.box.h_overlap(cbox) < 0.3 * min(p.box.x1 - p.box.x0, cbox.x1 - cbox.x0):
                    continue  # different column
                d = p.box.gap(cbox)
                if d <= best_d and not _blocked(p.box, cbox, blockers):
                    best, best_d = p, d
            if best is not None:
                best.caption = text
                linked.add(ref)


def _merge_split_panels(pics: List[_Pic], texts: dict) -> List[_Pic]:
    captioned = [p for p in pics if p.caption]
    for p in [p for p in pics if not p.caption]:
        items = texts.get(p.page_no, [])
        blockers = [b for lab, b, _r, _t in items if lab not in NON_BLOCKING]
        best, best_d = None, MERGE_GAP
        for c in captioned:
            if c.page_no != p.page_no:
                continue
            if c.box.h_overlap(p.box) <= 0 and c.box.v_overlap(p.box) <= 0:
                continue  # diagonal neighbours are not panels of one figure
            d = c.box.gap(p.box)
            if d <= best_d and not _blocked(c.box, p.box, blockers):
                best, best_d = c, d
        if best is not None:
            best.box = best.box.union(p.box)
    return captioned


class _PageRenderer:
    """Renders PDF pages at the output DPI directly from the PDF (one page cached)."""

    def __init__(self, pdf_path: str | Path, dpi: int):
        self.pdf = pdfium.PdfDocument(str(pdf_path))
        self.dpi = dpi
        self._page_no: Optional[int] = None
        self._image = None

    def page(self, page_no: int):
        """PIL image of a 1-based page."""
        if page_no != self._page_no:
            page = self.pdf[page_no - 1]
            self._image = page.render(scale=self.dpi / 72.0).to_pil().convert("RGB")
            page.close()
            self._page_no = page_no
        return self._image

    def close(self) -> None:
        self._image = None
        self.pdf.close()


def _crop(doc, pic: _Pic, renderer: _PageRenderer) -> bytes:
    pil = renderer.page(pic.page_no)
    s = pil.width / doc.pages[pic.page_no].size.width
    b = pic.box
    pil = pil.crop((
        max(0, int((b.x0 - CROP_PAD) * s)), max(0, int((b.y0 - CROP_PAD) * s)),
        min(pil.width, int((b.x1 + CROP_PAD) * s + 0.5)),
        min(pil.height, int((b.y1 + CROP_PAD) * s + 0.5)),
    ))
    if pil.width < 2 or pil.height < 2:
        return b""
    buf = io.BytesIO()
    pil.save(buf, format="PNG", dpi=(renderer.dpi, renderer.dpi))
    return buf.getvalue()


def extract_figures(converter: DocumentConverter, pdf_path: str | Path,
                    dpi: int = 300) -> List[Figure]:
    """Return one Figure per picture that has a figure caption."""
    doc = converter.convert(str(pdf_path)).document
    texts = _page_text_boxes(doc)
    pics = _collect_pictures(doc)
    _attach_unlinked_captions(doc, pics, texts)
    pics = _merge_split_panels(pics, texts)
    pics = [p for p in pics if not TABLE_RE.match(p.caption)]  # tables stored as images
    pics.sort(key=lambda p: (p.page_no, p.box.y0, p.box.x0))

    figures: List[Figure] = []
    renderer = _PageRenderer(pdf_path, dpi)
    try:
        images = [_crop(doc, p, renderer) for p in pics]
    finally:
        renderer.close()
    for p, image in zip(pics, images):
        if not image:
            continue
        m = FIGURE_RE.match(p.caption)
        page0 = p.page_no - 1
        figures.append(Figure(
            caption=p.caption,
            image_data=image,
            page=page0,
            bbox=BoundingBox(p.box.x0, p.box.y0, p.box.x1, p.box.y1, page0),
            figure_id=f"fig_{len(figures)}",
            figure_number=f"Figure {m.group(1)}" if m else None,
        ))
    return figures


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_outputs(pdf_path: Path, figures: List[Figure], out_dir: Path,
                  save_images: bool) -> Path:
    target = out_dir / pdf_path.stem
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for f in figures:
        if save_images:
            num = (f.figure_number or "nolabel").replace("Figure ", "")
            f.image_file = f"{f.figure_id}_fig{num}_p{f.page}.png"
            (target / f.image_file).write_bytes(f.image_data)
        b = f.bbox
        records.append({
            "figure_id": f.figure_id,
            "figure_number": f.figure_number,
            "caption": f.caption,
            "page": f.page,
            "bbox": {"x0": round(b.x0, 1), "y0": round(b.y0, 1),
                     "x1": round(b.x1, 1), "y1": round(b.y1, 1)},
            "image_file": f.image_file,
        })
    out = target / "figures.json"
    out.write_text(json.dumps({"pdf": str(pdf_path), "figures": records},
                              indent=2, ensure_ascii=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("results/figures_docling"))
    ap.add_argument("--dpi", type=int, default=300,
                    help="resolution of the saved PNGs (rendered from the PDF)")
    ap.add_argument("--no-images", action="store_true", help="write figures.json only")
    ap.add_argument("--no-ocr", action="store_true",
                    help="skip OCR (faster; misses captions on scanned pages)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    converter = build_converter(ocr=not args.no_ocr,
                                device=args.device, threads=args.threads)
    failed = 0
    for pdf in args.pdfs:
        if not pdf.is_file():
            print(f"skip (not a file): {pdf}", file=sys.stderr)
            continue
        start = time.time()
        try:
            figures = extract_figures(converter, pdf, dpi=args.dpi)
            out = write_outputs(pdf, figures, args.out_dir, save_images=not args.no_images)
        except Exception as exc:  # one bad PDF must not stop the batch
            failed += 1
            print(f"\n=== {pdf.name}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        print(f"\n=== {pdf.name}: {len(figures)} figures in {time.time() - start:.1f}s -> {out}")
        for f in figures:
            print(f"  p{f.page:<3} {str(f.figure_number):<11} {f.caption[:80]!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
