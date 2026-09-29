# KnowMat Vision

Extract figures and captions from scientific PDFs using Docling.

---

## Quick Start

### 1. Drop your PDFs into `papers/`

```
papers/
├── p-1.pdf
├── p-4.pdf
└── p-9.pdf
```

### 2. Run the script

```bash
cd scripts
bash run_fig.sh
```

That's it. Figures are extracted from every PDF in `papers/` and saved to `figures_highRes/`.

---

## Directory Flow

```
KnowMat_Vision/
│
├── papers/                    ← put source PDFs here
│   ├── p-1.pdf
│   └── p-9.pdf
│
├── figures/                   ← auto-created; one subdir per paper
│   ├── p-1/
│   │   ├── figures.json       ← figure number, caption, page, bbox
│   │   ├── fig_0_fig1_p2.png
│   │   └── fig_1_fig2_p4.png
│   └── p-9/
│       ├── figures.json
│       └── fig_0_fig1_p3.png
│
└── scripts/
    ├── run_fig.sh             ← entry point
    └── extract_figure_docling.py
```

---

## Configuration

Edit the top of [`scripts/run_fig.sh`](scripts/run_fig.sh):

| Variable | Default | Description |
|---|---|---|
| `PAPERS_DIR` | `../papers` | Folder with source PDFs |
| `OUT_DIR` | `../figures` | Output root |
| `DEVICE` | `mps` | `mps` (Mac), `cuda` (NVIDIA), `cpu` |
| `EXTRA_ARGS` | _(empty)_ | e.g. `--no-ocr`, `--dpi 300` |

---

## Output

Each paper gets its own folder with:
- **`figures.json`** — metadata for every extracted figure (number, full caption, page, bounding box, image filename)
- **`fig_N_figX_pY.png`** — the cropped figure image at 300 DPI

---

## Requirements

See [`setup.md`](setup.md) for full installation instructions (`VLMtrain` conda env with `docling`).
