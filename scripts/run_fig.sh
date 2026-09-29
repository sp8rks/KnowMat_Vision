#!/usr/bin/env bash
# Extract figures + captions from every PDF in PAPERS_DIR using
# extract_figure_docling.py.  Results go to OUT_DIR/<pdf_stem>/.
#
# Usage:  bash run_fig.sh
set -euo pipefail

# ---- change these --------------------------------------------------------- #
PAPERS_DIR="../papers"   # directory containing the source PDFs
OUT_DIR="../figures"             # output root (created if absent)
DEVICE="mps"             # mps (Mac GPU) | cpu | cuda | auto
EXTRA_ARGS=""            # e.g. "--no-ocr" or "--no-images" or "--dpi 300"
PYTHON="$HOME/miniconda3/envs/VLMtrain/bin/python"  # conda env with all dependencies
# --------------------------------------------------------------------------- #

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACTOR="$SCRIPT_DIR/extract_figure_docling.py"

if [ ! -f "$EXTRACTOR" ]; then
    echo "extract_figure_docling.py not found in $SCRIPT_DIR" >&2
    exit 1
fi

ABS_PAPERS="$(cd "$PAPERS_DIR" 2>/dev/null && pwd || { echo "papers dir not found: $PAPERS_DIR" >&2; exit 1; })"
ABS_OUT="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

shopt -s nullglob
PDFS=("$ABS_PAPERS"/*.pdf)
shopt -u nullglob

if [ ${#PDFS[@]} -eq 0 ]; then
    echo "No PDF files found in $ABS_PAPERS" >&2
    exit 1
fi

echo "Found ${#PDFS[@]} PDF(s) in $ABS_PAPERS"
echo "Output root: $ABS_OUT"
echo

FAILED=0
for PDF in "${PDFS[@]}"; do
    STEM="$(basename "$PDF" .pdf)"
    TMP_DIR="$ABS_OUT/.tmp_$STEM"

    echo ">>> Processing: $STEM"
    rm -rf "$TMP_DIR"

    # shellcheck disable=SC2086
    if "$PYTHON" "$EXTRACTOR" "$PDF" --device "$DEVICE" -o "$TMP_DIR" $EXTRA_ARGS; then
        mkdir -p "$ABS_OUT/$STEM"
        mv "$TMP_DIR/$STEM"/* "$ABS_OUT/$STEM"/
        rm -rf "$TMP_DIR"
        echo "    Done -> $ABS_OUT/$STEM"
    else
        echo "    FAILED: $STEM" >&2
        rm -rf "$TMP_DIR"
        FAILED=$((FAILED + 1))
    fi
    echo
done

echo "=============================="
echo "Finished. Failed: $FAILED / ${#PDFS[@]}"
ls -1 "$ABS_OUT"
