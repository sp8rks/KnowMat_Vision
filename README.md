# Materials VLM training data generation

## Purpose

1. Extracts embedded images from PDFs.
2. Classifies all extracted figure images into `xy_plots`, `schematics`, or `arbitrary`.
3. Detects perpendicular x-y axis pairs (including multi-panel figures) and extracts x-y plot candidates.
4. Uses figure-local text/ticks around axes to infer labels, units, and scale (when available).
5. Digitizes curve series from plot images.
6. Exports ground-truth x-y points and figure metadata to CSV.
7. Generates synthetic x-y data and plots.
8. Applies 10 augmentation modes with 16 variations per plot.
9. Adds automatic stats labeling on generated plots.

## Run

```bash
cd week2
python3 -m src.init --pdf-dir "../100 articles" --out outputs
```

Optional flags:

```bash
python3 -m src.init --pdf-dir "../100 articles" --out outputs --max-pdfs 20 --max-figures-per-pdf 4 --seed 11
```

## Outputs

- `outputs/extracted_figures/` : extracted figure images from PDFs
- `outputs/classified_figures/xy_plots/` : all figure images classified as x-y plots
- `outputs/classified_figures/schematics/` : all figure images classified as schematics/diagrams
- `outputs/classified_figures/arbitrary/` : all figure images classified as other/arbitrary
- `outputs/csv/figure_classification.csv` : per-image classification table
- `outputs/csv/ground_truth_xy_points.csv` : point-level extracted x-y data
- `outputs/csv/ground_truth_figures.csv` : figure-level metadata/scalars
- `outputs/csv/synthetic_xy_points.csv` : synthetic + augmented x-y points
- `outputs/plots/raw_plots/` : standardized raw plots rendered from extracted x-y data with automatic labeling
- `outputs/plots/synthetic_base/` : base synthetic plots
- `outputs/plots/synthetic_augmented/` : 16 variations per plot
- `outputs/reports/run_report_<timestamp>.json` : run summary

## WebPlotDigitizer Bridge (Manual/Assisted)

Install wrapper once:

```bash
cd week2
npm install @insilicall/img2data
```

Prepare a queue of classified x-y figures for WebPlotDigitizer:

```bash
python3 -m src.wpd_bridge --run-dir outputs --out outputs/webplotdigitizer_queue
```

This creates:
- `outputs/webplotdigitizer_queue/images/`
- `outputs/webplotdigitizer_queue/manifest.csv`
- `outputs/webplotdigitizer_queue/README_webplotdigitizer.md`

## Augmentation Modes

Implemented modes:

1. over/underplotting (overplotting, neutral, underplotting)
2. x-shift
3. y-shift
4. curve misassignment
5. unit error
6. scale error
7. axis orientation error
8. visual density (high, medium, low)
9. resolution/blurriness (high, medium, low)
10. complex feature distortion
