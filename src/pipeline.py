from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import re
import tempfile
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyMuPDF (fitz) is required for this pipeline") from exc

from PIL import Image, ImageFilter


@dataclass
class SeriesData:
    series_id: str
    series_label: str
    x: list[float]
    y: list[float]


@dataclass
class FigureData:
    figure_id: str
    pdf_file: str
    page_index: int
    image_index: int
    image_file: str
    descriptor: str
    x_label: str
    x_unit: str
    y_label: str
    y_unit: str
    series: list[SeriesData]


@dataclass
class AxisPair:
    x_axis_row: int
    y_axis_col: int
    x_right: int
    y_top: int
    plot_bbox: tuple[int, int, int, int]
    score: float


@dataclass
class VariationConfig:
    variation_id: str
    name: str
    plotting_level: str = "neutral"
    x_shift: float = 0.0
    y_shift: float = 0.0
    curve_misassignment: bool = False
    unit_error: bool = False
    scale_error: float = 1.0
    axis_orientation_error: str = "none"
    visual_density: str = "medium"
    resolution_blurriness: str = "medium"
    complex_feature_distortion: str = "none"

    def as_mode_dict(self) -> dict[str, Any]:
        return {
            "over_underplotting": self.plotting_level,
            "x_shift": self.x_shift,
            "y_shift": self.y_shift,
            "curve_misassignment": self.curve_misassignment,
            "unit_error": self.unit_error,
            "scale_error": self.scale_error,
            "axis_orientation_error": self.axis_orientation_error,
            "visual_density": self.visual_density,
            "resolution_blurriness": self.resolution_blurriness,
            "complex_feature_distortion": self.complex_feature_distortion,
        }


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return s or "item"


def _float_list(values: list[float]) -> list[float]:
    return [float(v) for v in values]


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    classified_root = out_dir / "classified_figures"
    arbitrary_dir = classified_root / "arbitrary"
    dirs = {
        "root": out_dir,
        "figures": out_dir / "extracted_figures",
        "classified_root": classified_root,
        "classified_xy": classified_root / "xy_plots",
        # User preference: schematics are merged into arbitrary for output folders.
        "classified_schematic": arbitrary_dir,
        "classified_arbitrary": arbitrary_dir,
        "csv": out_dir / "csv",
        "plots_raw": out_dir / "plots" / "raw_plots",
        "plots_base": out_dir / "plots" / "synthetic_base",
        "plots_aug": out_dir / "plots" / "synthetic_augmented",
        "reports": out_dir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


STANDARD_UNITS_CSV = Path(__file__).with_name("material_parameter_units.csv")
STANDARD_UNITS_TXT = Path(__file__).resolve().parents[1] / "materials_parameters_units.txt"
STANDARD_UNITS_EXPANDED_TXT = Path(__file__).resolve().parents[1] / "materials_parameters_units_expanded.txt"


def _normalize_parameter_key(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("−", "-")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _normalize_unit_display(text: str) -> str:
    u = re.sub(r"\s+", " ", (text or "").strip())
    low = u.lower()
    if not u:
        return ""
    if "dimensionless" in low:
        return "-"
    if "or degree" in low or "degree" == low:
        return "deg"
    u = u.replace("μ", "µ").replace("Ω", "Ω")
    u = re.sub(r"(?i)\bohm\b", "Ω", u)
    u = u.replace(" ", "")
    u = re.sub(r"(?<![A-Za-z0-9])u(?=[mAsVFWHs](?:\b|[\^/\-]))", "µ", u)
    u = u.replace("^(", "^").replace(")-", "^-")
    u = u.replace("m^2/s", "m^2s^-1")
    return u


def _normalize_unit_key(text: str) -> str:
    u = _normalize_unit_display(text).lower()
    if not u:
        return ""
    u = u.replace(".", "")
    u = u.replace("μ", "u").replace("µ", "u")
    u = u.replace("ω", "ohm").replace("Ω", "ohm").replace("°", "deg")
    u = u.replace("vsrhe", "vvsrhe")
    u = re.sub(r"[^a-z0-9/\-^]+", "", u)
    return u


def _is_noise_unit_token(unit: str) -> bool:
    ul = _normalize_unit_key(unit)
    return ul in {"ip", "oop", "pfm", "xrd", "sem", "tem", "edx", "fig"}


def _rows_from_parameter_unit_text(txt_path: Path) -> list[tuple[str, str, str]]:
    if not txt_path.exists():
        return []
    try:
        text = txt_path.read_text(encoding="utf-8")
    except Exception:
        return []

    flat = re.sub(r"\s+", " ", text).strip()
    # Match repeated "Parameter (...) — unit" sequences in wrapped prose.
    pat = re.compile(
        r"([A-Za-z\u0370-\u03FF][A-Za-z0-9\u0370-\u03FF’'/_\-\s\(\),\.]+?)\s*[—-]\s*([A-Za-z0-9·\^\-\(\)\/\s\.\*µμΩΩ°√]+?)(?=(?:\s+[A-Z][A-Za-z0-9’'/_\-]*(?:\s+[A-Za-z0-9’'/_\-]+){0,6}\s*(?:\([^)]+\))?\s*[—-])|$)"
    )

    rows: list[tuple[str, str, str]] = []
    for m in pat.finditer(flat):
        param_raw = m.group(1).strip(" ,;:.")
        unit_raw = m.group(2).strip(" ,;:.")
        if not param_raw or not unit_raw:
            continue

        aliases: list[str] = []
        for token in re.findall(r"\(([^)]+)\)", param_raw):
            tok = token.strip()
            if tok and len(tok) <= 18:
                aliases.append(tok)
        param = re.sub(r"\([^)]*\)", "", param_raw).strip(" ,;:.")
        if not param:
            continue
        plow = param.lower()
        section_prefixes = (
            "basic physical properties",
            "mechanical properties",
            "thermal properties",
            "electrical properties",
            "magnetic properties",
            "optical properties",
            "transport properties",
            "surface and interface properties",
            "chemical and thermodynamic properties",
            "microstructural parameters",
        )
        for sfx in section_prefixes:
            if plow.startswith(sfx + " "):
                param = param[len(sfx) :].strip(" ,;:.")
                plow = param.lower()
                break
        if not param or " properties " in f" {plow} ":
            continue
        if len(param.split()) > 6 or len(param) > 56:
            continue

        unit = _normalize_unit_display(unit_raw)
        if not unit:
            continue
        if unit.lower() in {"varies", "various"}:
            continue
        rows.append((param, unit, "|".join(dict.fromkeys(aliases))))
    return rows


def _unit_symbol_score(unit: str) -> int:
    u = unit or ""
    score = 0
    if "µ" in u or "μ" in u:
        score += 4
    if "Ω" in u or "Ω" in u:
        score += 3
    if "°" in u:
        score += 2
    if "·" in u:
        score += 1
    if "√" in u:
        score += 1
    if "^" in u:
        score += 1
    return score


@lru_cache(maxsize=1)
def _load_canonical_unit_symbols() -> dict[str, str]:
    canonical: dict[str, str] = {}
    rows: list[tuple[str, str, str]] = []
    rows.extend(_rows_from_parameter_unit_text(STANDARD_UNITS_EXPANDED_TXT))
    rows.extend(_rows_from_parameter_unit_text(STANDARD_UNITS_TXT))
    rows.extend(_rows_from_parameter_unit_csv(STANDARD_UNITS_CSV))
    for _param, unit, _aliases in rows:
        disp = _normalize_unit_display(unit)
        key = _normalize_unit_key(disp)
        if not key:
            continue
        cur = canonical.get(key)
        if cur is None or _unit_symbol_score(disp) > _unit_symbol_score(cur):
            canonical[key] = disp

    # Fallback aliases for common OCR/ascii variants.
    canonical.update(
        {
            "um": "µm",
            "um^-1": "µm^-1",
            "um-1": "µm^-1",
            "um^2": "µm^2",
            "um2": "µm^2",
            "um^3": "µm^3",
            "ua": "µA",
            "uv": "µV",
            "us": "µs",
            "uf": "µF",
            "uh": "µH",
            "uw": "µW",
            "ohm": "Ω",
            "ohmm": "Ω·m",
            "degc": "°C",
        }
    )
    return canonical


def _canonicalize_unit_output(unit: str) -> str:
    disp = _normalize_unit_display(unit)
    if not disp:
        return "arb"
    if disp == "-":
        return "-"
    key = _normalize_unit_key(disp)
    if not key:
        return disp
    cmap = _load_canonical_unit_symbols()
    if key in {"cycle", "cycles"}:
        return "cycles"
    out = cmap.get(key, disp)
    out = out.replace("μ", "µ").replace("Ω", "Ω")
    return out


def _rows_from_parameter_unit_csv(csv_path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    if not csv_path.exists():
        return rows
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                unit = str(row.get("unit", "")).strip()
                param = str(row.get("parameter", "")).strip()
                aliases = str(row.get("aliases", "")).strip()
                if not unit or not param:
                    continue
                rows.append((param, unit, aliases))
    except Exception:
        return []
    return rows


def _is_generic_alias_key(key: str) -> bool:
    k = (key or "").strip()
    if not k:
        return True
    tokens = k.split()
    if all(len(t) == 1 for t in tokens):
        return True
    if k in {"x", "y", "z", "a", "b", "c", "r", "n", "m", "k", "e", "h", "t"}:
        return True
    return False


@lru_cache(maxsize=1)
def _load_standard_parameter_units() -> dict[str, str]:
    mapping: dict[str, str] = {}
    rows: list[tuple[str, str, str]] = []
    # User-provided text mapping is loaded first.
    rows.extend(_rows_from_parameter_unit_text(STANDARD_UNITS_EXPANDED_TXT))
    rows.extend(_rows_from_parameter_unit_text(STANDARD_UNITS_TXT))
    rows.extend(_rows_from_parameter_unit_csv(STANDARD_UNITS_CSV))
    if not rows:
        return mapping

    # Pass 1: add explicit parameter names.
    for param, unit, _aliases in rows:
        key = _normalize_parameter_key(param)
        if key:
            mapping[key] = _canonicalize_unit_output(unit)

    # Pass 2: add aliases only when they do not conflict with explicit names.
    for _param, unit, aliases in rows:
        if not aliases:
            continue
        for alias in aliases.split("|"):
            key = _normalize_parameter_key(alias)
            if not key or _is_generic_alias_key(key):
                continue
            if key not in mapping:
                mapping[key] = _canonicalize_unit_output(unit)

    return mapping


@lru_cache(maxsize=1)
def _load_unit_to_standard_labels() -> dict[str, list[str]]:
    label_to_unit = _load_standard_parameter_units()
    out: dict[str, list[str]] = {}
    for label_key, unit in label_to_unit.items():
        ukey = _normalize_unit_key(unit)
        if not ukey:
            continue
        out.setdefault(ukey, [])
        if label_key not in out[ukey]:
            out[ukey].append(label_key)
    return out


def _lookup_standard_unit(label: str) -> str | None:
    key = _normalize_parameter_key(label)
    if not key:
        return None
    mapping = _load_standard_parameter_units()
    if not mapping:
        return None

    if key in mapping:
        return _canonicalize_unit_output(mapping[key])

    best_key = ""
    best_unit: str | None = None
    for k, unit in mapping.items():
        if len(k) < 4:
            continue
        if k in key and len(k) > len(best_key):
            best_key = k
            best_unit = unit
    return _canonicalize_unit_output(best_unit) if best_unit else None


def _lookup_standard_label_from_unit(unit: str, axis_hint: str | None = None) -> str | None:
    unit_key = _normalize_unit_key(unit)
    if not unit_key:
        return None
    rev = _load_unit_to_standard_labels()
    labels = list(rev.get(unit_key, []))

    if not labels and len(unit_key) > 1:
        for pfx in ("k", "m", "g", "u", "n", "p"):
            if unit_key.startswith(pfx):
                labels = list(rev.get(unit_key[1:], []))
                if labels:
                    break
    if not labels:
        return None

    x_pref = ("time", "temperature", "frequency", "wavelength", "energy", "field", "strain", "distance", "angle")
    y_pref = ("stress", "strength", "modulus", "intensity", "current", "conductivity", "rate", "magnetization", "hardness")

    # Resolve highly ambiguous units with common axis defaults.
    if axis_hint == "x":
        if unit_key in {"s", "ms", "us", "ns", "min", "h"}:
            return "time"
        if unit_key in {"k"}:
            return "temperature"
    if axis_hint == "y":
        if unit_key in {"pa", "mpa", "gpa"}:
            return "stress"
        if unit_key in {"a", "ma", "ua"}:
            return "current"
        if unit_key in {"sm^-1", "sm-1", "sm^-1"}:
            return "conductivity"

    pref = x_pref if axis_hint == "x" else y_pref if axis_hint == "y" else ()

    best = ""
    best_score = -1
    for label_key in labels:
        score = 0
        if pref:
            if any(k in label_key for k in pref):
                score += 3
        if "coefficient" in label_key:
            score -= 1
        if len(label_key) <= 2:
            score -= 1
        if score > best_score:
            best_score = score
            best = label_key
    if not best:
        return None
    return best


def _axis_preference_for_label(label: str) -> str:
    low = (label or "").lower()
    x_like = ("time", "temperature", "frequency", "wavelength", "energy", "field", "strain", "length", "distance", "angle", "concentration")
    y_like = ("stress", "strength", "modulus", "intensity", "current", "conductivity", "rate", "voltage", "magnetization", "hardness", "phase")
    if any(k in low for k in x_like):
        return "x"
    if any(k in low for k in y_like):
        return "y"
    return ""


def _trim_panel_token_prefix(name: str) -> str:
    tokens = (name or "").split()
    if len(tokens) < 2:
        return name
    if all(re.fullmatch(r"[A-Za-z]", tok) for tok in tokens):
        # "N b" often comes from OCR-split element symbols; keep short joined forms.
        if len(tokens) <= 2:
            return "".join(tokens)
        # Longer pure single-letter sequences are usually panel markers (a b c ...).
        return ""
    idx = 0
    while idx < len(tokens) and re.fullmatch(r"[A-Za-z]", tokens[idx]):
        idx += 1
    if idx >= len(tokens):
        return ""
    if idx >= 2:
        return " ".join(tokens[idx:])
    return name


def _extract_label_before_delimiter(prefix: str) -> str:
    p = _strip_links_and_doi(re.sub(r"\s+", " ", (prefix or "").strip())).strip(" ,;:.-")
    if not p:
        return ""
    # Keep only the trailing clause before a delimiter, then take its trailing label-like span.
    p = re.split(r"[;,]", p)[-1].strip()
    m = re.search(
        r"([A-Za-z\u0394\u2206\u03ba\u03b4\u03bc\u00b5][A-Za-z0-9\u0394\u2206\u03ba\u03b4\u03bc\u00b5_\-]{0,24}(?:\s+[A-Za-z0-9\u0394\u2206\u03ba\u03b4\u03bc\u00b5_\-]{1,24}){0,3})$",
        p,
    )
    cand = m.group(1) if m else p
    out = _trim_panel_token_prefix(_clean_axis_name(cand))
    if out.lower() in {"org", "com", "net", "www", "http", "https", "doi"}:
        return ""
    return out


def _looks_like_link_or_doi(text: str) -> bool:
    low = (text or "").strip().lower()
    if not low:
        return False
    if "http://" in low or "https://" in low or "www." in low or "doi.org" in low:
        return True
    if re.search(r"\bdoi\b", low):
        return True
    if re.search(r"\b10\.\d{4,9}/\S+", low):
        return True
    if re.search(r"\b[a-z0-9-]+\.(org|com|net|edu|gov)\b", low):
        return True
    return False


def _strip_links_and_doi(text: str) -> str:
    t = text or ""
    t = re.sub(r"https?://\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bwww\.\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdoi\s*[:=]\s*\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b10\.\d{4,9}/\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _collect_embedded_param_candidates(text: str) -> list[tuple[str, str, float]]:
    t = _strip_links_and_doi(re.sub(r"\s+", " ", (text or "").strip()))
    if not t:
        return []

    out: list[tuple[str, str, float]] = []
    low = t.lower()

    for m in re.finditer(
        r"([A-Za-z][A-Za-z0-9\-/\s\u0394\u03bc\u00b5]{1,40})[\(\[]\s*([A-Za-z0-9%/\-\^\.\u00b5\u03a9]{1,16})\s*[\)\]]",
        t,
    ):
        name = _extract_label_before_delimiter(m.group(1))
        unit = _clean_unit(m.group(2))
        if (
            _is_plausible_axis_label(name)
            and _is_probable_unit(unit)
            and not _is_noise_unit_token(unit)
        ):
            out.append((name, unit, 9.0))

    for m in re.finditer(
        r"([A-Za-z][A-Za-z0-9\-/\s]{2,48})\s*[—:]\s*([A-Za-z0-9%/\-\^\.\u00b5\u03a9]{1,18})",
        t,
    ):
        name = _extract_label_before_delimiter(m.group(1))
        unit = _clean_unit(m.group(2))
        if (
            _is_plausible_axis_label(name)
            and _is_probable_unit(unit)
            and not _is_noise_unit_token(unit)
        ):
            out.append((name, unit, 7.0))

    # Explicit slash-delimited axis text such as "Nb/cycles" or "Stress/MPa".
    for m in re.finditer(
        r"([A-Za-z][A-Za-z0-9\-\s]{1,40})\s*/\s*([A-Za-z\u00b5\u03bc\u03a9%][A-Za-z0-9\u00b5\u03bc\u03a9%/\-\^\.]{0,18})",
        t,
    ):
        name = _extract_label_before_delimiter(m.group(1))
        unit = _clean_unit(m.group(2))
        if _is_plausible_axis_label(name) and _is_probable_unit(unit) and not _is_noise_unit_token(unit):
            out.append((name, unit, 8.0))

    norm = _normalize_parameter_key(t)
    mapping = _load_standard_parameter_units()
    for k, unit in mapping.items():
        if len(k) < 4:
            continue
        if re.search(rf"\b{re.escape(k)}\b", norm):
            out.append((k, unit, 5.0 + min(2.0, len(k) / 30.0)))

    for m in re.finditer(r"\b(μm|µm|um|nm|mm|cm|ms|us|ns|min|hz|khz|mhz|ghz|ev|pa|kpa|mpa|gpa|mv|ma|ua|ohm|deg|rad)\b", t, flags=re.IGNORECASE):
        u = _clean_unit(m.group(1))
        if not _is_probable_unit(u):
            continue
        label_guess = _lookup_standard_label_from_unit(u)
        if label_guess:
            out.append((label_guess, u, 3.0))

    # Single-letter units are accepted only when accompanied by a number.
    for m in re.finditer(r"\b[+-]?\d+(?:\.\d+)?\s*(v|a|k|s|%)\b", t, flags=re.IGNORECASE):
        u = _clean_unit(m.group(1))
        if not _is_probable_unit(u):
            continue
        label_guess = _lookup_standard_label_from_unit(u)
        if label_guess:
            out.append((label_guess, u, 3.0))

    # High-priority domain cues from embedded text.
    if "voltage" in low or re.search(r"\b[+-]?\d+(?:\.\d+)?\s*v\b", low):
        out.append(("voltage", "V", 8.0))
    if "current" in low or re.search(r"\b[+-]?\d+(?:\.\d+)?\s*(?:ma|ua|a)\b", low):
        out.append(("current", "A", 6.0))
    len_u = _extract_length_unit_from_text(t)
    if len_u and ("scale bar" in low or "bias" in low or re.search(r"\b\d+(?:\.\d+)?\s*[x×]\s*\d+", low)):
        out.append(("length", len_u, 8.0))

    best: dict[str, tuple[str, str, float]] = {}
    for label, unit, score in out:
        key = _normalize_parameter_key(label)
        cur = best.get(key)
        if cur is None or score > cur[2]:
            best[key] = (label, unit, score)
    return list(best.values())


def _pick_embedded_candidate(
    cands: list[tuple[str, str, float]],
    axis_hint: str,
    used_labels: set[str] | None = None,
) -> tuple[str, str, float] | None:
    used = used_labels or set()
    best: tuple[str, str, float] | None = None
    best_score = -1e9
    for label, unit, score in cands:
        lkey = _normalize_parameter_key(label)
        if lkey in used:
            continue
        s = float(score)
        pref = _axis_preference_for_label(label)
        if pref == axis_hint:
            s += 2.0
        elif pref and pref != axis_hint:
            s -= 1.0
        if s > best_score:
            best_score = s
            best = (label, unit, s)
    return best


def _label_keys_similar(a: str, b: str) -> bool:
    ak = _normalize_parameter_key(a)
    bk = _normalize_parameter_key(b)
    if not ak or not bk:
        return False
    return ak == bk or ak in bk or bk in ak


def _pick_embedded_unit_candidate(
    cands: list[tuple[str, str, float]],
    axis_hint: str,
    current_label: str,
) -> tuple[str, str, float] | None:
    best: tuple[str, str, float] | None = None
    best_score = -1e9
    for label, unit, score in cands:
        if not _is_probable_unit(unit):
            continue
        s = float(score)
        pref = _axis_preference_for_label(label)
        if pref == axis_hint:
            s += 1.5
        elif pref and pref != axis_hint:
            s -= 0.8
        if current_label not in {"x", "y"} and _label_keys_similar(label, current_label):
            s += 3.0
        if _lookup_standard_unit(label) is not None:
            s += 0.7
        if s > best_score:
            best_score = s
            best = (label, unit, s)
    return best


def discover_pdfs(pdf_dir: Path, max_pdfs: int | None = None) -> list[Path]:
    pdfs = sorted(p for p in pdf_dir.glob("*.pdf") if p.is_file())
    if max_pdfs is not None:
        pdfs = pdfs[: max(0, max_pdfs)]
    return pdfs


def _xy_text_score(page_text: str) -> int:
    if not page_text:
        return 0

    low = page_text.lower()
    score = 0

    if " vs " in low or "versus" in low:
        score += 2

    axis_terms = [
        "temperature",
        "time",
        "strain",
        "stress",
        "voltage",
        "current",
        "frequency",
        "wavelength",
        "energy",
        "intensity",
        "pressure",
        "modulus",
    ]
    hits = sum(1 for term in axis_terms if term in low)
    if hits >= 2:
        score += 1

    if len(re.findall(r"\([a-zA-Z0-9%/\-\^\.\s]{1,12}\)", page_text)) >= 2:
        score += 1

    numeric_hits = len(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", low))
    if numeric_hits >= 4:
        score += 1

    return score


def _longest_true_run(mask_1d: np.ndarray) -> int:
    if mask_1d.size == 0:
        return 0
    a = mask_1d.astype(np.int8)
    padded = np.pad(a, (1, 1), constant_values=0)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if starts.size == 0 or ends.size == 0:
        return 0
    return int(np.max(ends - starts))


def _run_bounds_1d(mask_1d: np.ndarray, idx: int) -> tuple[int, int] | None:
    n = int(mask_1d.size)
    if n == 0 or idx < 0 or idx >= n:
        return None
    if not bool(mask_1d[idx]):
        return None
    lo = int(idx)
    hi = int(idx)
    while lo > 0 and bool(mask_1d[lo - 1]):
        lo -= 1
    while hi + 1 < n and bool(mask_1d[hi + 1]):
        hi += 1
    return lo, hi


def _axis_continuity(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    plot = gray[y0:y1, x0:x1]
    if plot.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    h, w = plot.shape
    if h < 20 or w < 20:
        return 0.0, 0.0, 0.0, 0.0

    dark = plot < 165
    row_density = dark.mean(axis=1)
    col_density = dark.mean(axis=0)

    row_start = int(0.55 * h)
    col_end = max(1, int(0.45 * w))
    if row_start >= h:
        row_start = max(0, h - 1)

    x_axis_row = int(np.argmax(row_density[row_start:]) + row_start)
    y_axis_col = int(np.argmax(col_density[:col_end]))

    row_run = _longest_true_run(dark[x_axis_row, :]) / max(1.0, w)
    col_run = _longest_true_run(dark[:, y_axis_col]) / max(1.0, h)
    return float(row_density[x_axis_row]), float(col_density[y_axis_col]), float(row_run), float(col_run)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    a_area = float(max(1, ax1 - ax0) * max(1, ay1 - ay0))
    b_area = float(max(1, bx1 - bx0) * max(1, by1 - by0))
    return inter / max(1e-9, (a_area + b_area - inter))


def _bbox_overlap_on_smaller(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
    b_area = max(1e-9, (bx1 - bx0) * (by1 - by0))
    return float(inter / min(a_area, b_area))


def _axis_exactness_score(
    dark: np.ndarray,
    row: int,
    col: int,
    row_right: int,
    col_top: int,
) -> tuple[float, float]:
    h, w = dark.shape
    row = int(np.clip(row, 0, max(0, h - 1)))
    col = int(np.clip(col, 0, max(0, w - 1)))
    row_right = int(np.clip(row_right, col, max(0, w - 1)))
    col_top = int(np.clip(col_top, 0, row))

    row_slice = dark[row, col : row_right + 1]
    col_slice = dark[col_top : row + 1, col]
    if row_slice.size == 0 or col_slice.size == 0:
        return 0.0, 0.0

    horiz = float(row_slice.mean())
    vert = float(col_slice.mean())

    if row > 0:
        horiz -= 0.18 * float(dark[row - 1, col : row_right + 1].mean())
    if row + 1 < h:
        horiz -= 0.18 * float(dark[row + 1, col : row_right + 1].mean())
    if col > 0:
        vert -= 0.18 * float(dark[col_top : row + 1, col - 1].mean())
    if col + 1 < w:
        vert -= 0.18 * float(dark[col_top : row + 1, col + 1].mean())

    return max(0.0, horiz), max(0.0, vert)


def _is_composite_axis_pair(
    axis_pair: AxisPair,
    axis_pairs: list[AxisPair],
    *,
    img_w: int,
    img_h: int,
) -> bool:
    x0, y0, x1, y1 = axis_pair.plot_bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    if bw < int(0.52 * img_w) and bh < int(0.62 * img_h):
        return False

    inner_hits = 0
    saw_strong_inner = False
    for other in axis_pairs:
        if other is axis_pair:
            continue
        ox0, oy0, ox1, oy1 = other.plot_bbox
        obw = max(1, ox1 - ox0)
        obh = max(1, oy1 - oy0)
        if obw > int(0.48 * img_w):
            continue
        if ox0 < x0 - 0.04 * bw or ox1 > x1 + 0.04 * bw:
            continue
        if oy0 < y0 - 0.12 * bh or oy1 > y1 + 0.12 * bh:
            continue
        if abs(other.x_axis_row - axis_pair.x_axis_row) > max(18, int(0.12 * bh)):
            continue
        if other.score < axis_pair.score - 0.45:
            continue
        inner_hits += 1
        if obw <= int(0.45 * img_w):
            saw_strong_inner = True
        if inner_hits >= 2:
            return True
    if bw >= int(0.72 * img_w) and saw_strong_inner:
        return True
    return False


def _filter_redundant_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for asset in assets:
        key = (str(asset.get("pdf_file") or ""), int(asset.get("page_index") or 0))
        grouped.setdefault(key, []).append(asset)

    for group_assets in grouped.values():
        ranked = []
        for asset in group_assets:
            axis_meta = asset.get("axis_meta") or {}
            pb = _safe_float_bbox(axis_meta.get("plot_bbox_page"))
            src_idx = int(asset.get("source_image_index") or 0)
            source_kind = "page_level" if src_idx >= 900 else "image_object"
            if pb is None:
                ranked.append((asset, None, float("inf"), 0, source_kind))
                continue
            ax0, ay0, ax1, ay1 = pb
            area = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
            quality = 0
            if str(axis_meta.get("x_label") or "x") != "x":
                quality += 1
            if str(axis_meta.get("y_label") or "y") != "y":
                quality += 1
            if str(axis_meta.get("x_unit") or "arb") != "arb":
                quality += 1
            if str(axis_meta.get("y_unit") or "arb") != "arb":
                quality += 1
            ranked.append((asset, pb, area, quality, source_kind))

        drop_idx: set[int] = set()
        for i, (_, pbi, areai, quality_i, source_i) in enumerate(ranked):
            if pbi is None or i in drop_idx:
                continue
            for j, (_, pbj, areaj, quality_j, source_j) in enumerate(ranked):
                if i == j or pbj is None or j in drop_idx:
                    continue
                overlap_small = _bbox_overlap_on_smaller(pbi, pbj)
                if overlap_small < 0.82:
                    continue
                if (
                    source_i == "image_object"
                    and source_j == "page_level"
                    and areai > 1.20 * areaj
                    and quality_i <= quality_j
                ):
                    drop_idx.add(i)
                    break
                if areai > 1.55 * areaj and quality_i <= quality_j:
                    drop_idx.add(i)
                    break

        # Broader image-object crops can still survive when they sit adjacent to,
        # rather than directly on top of, tighter page-level subplot crops from the
        # same multi-panel row. Suppress those umbrella crops by row proximity.
        for i, (_, pbi, areai, quality_i, source_i) in enumerate(ranked):
            if pbi is None or i in drop_idx or source_i != "image_object":
                continue
            ax0, ay0, ax1, ay1 = pbi
            aw = max(1e-9, ax1 - ax0)
            ah = max(1e-9, ay1 - ay0)
            if aw < 0.42:
                continue
            nearby_small = 0
            best_quality = -1
            for j, (_, pbj, areaj, quality_j, source_j) in enumerate(ranked):
                if i == j or pbj is None or j in drop_idx or source_j != "page_level":
                    continue
                bx0, by0, bx1, by1 = pbj
                bw = max(1e-9, bx1 - bx0)
                bh = max(1e-9, by1 - by0)
                if bw > 0.70 * aw:
                    continue
                ayc = 0.5 * (ay0 + ay1)
                byc = 0.5 * (by0 + by1)
                if abs(ayc - byc) > 0.22 * max(ah, bh):
                    continue
                vertical_overlap = max(0.0, min(ay1, by1) - max(ay0, by0)) / max(1e-9, min(ah, bh))
                if vertical_overlap < 0.45:
                    continue
                horizontal_gap = max(0.0, max(ax0, bx0) - min(ax1, bx1))
                overlap_small = _bbox_overlap_on_smaller(pbi, pbj)
                if overlap_small < 0.20 and horizontal_gap > 0.20 * aw:
                    continue
                nearby_small += 1
                best_quality = max(best_quality, quality_j)
            if nearby_small >= 2 and quality_i <= best_quality:
                drop_idx.add(i)

        for idx, (asset, _, _, _, _) in enumerate(ranked):
            if idx in drop_idx:
                try:
                    Path(str(asset.get("image_path") or "")).unlink(missing_ok=True)
                except Exception:
                    pass
                continue
            kept.append(asset)
    return kept


def _detect_axis_pairs(gray: np.ndarray, max_pairs: int = 6) -> list[AxisPair]:
    h, w = gray.shape
    if h < 80 or w < 100:
        return []

    dark = gray < 165
    row_density = dark.mean(axis=1)
    col_density = dark.mean(axis=0)

    row_base = np.argsort(row_density)[::-1][: max(12, h // 18)]
    col_base = np.argsort(col_density)[::-1][: max(12, w // 18)]

    row_set = {int(r) for r in row_base.tolist()}
    col_set = {int(c) for c in col_base.tolist()}

    block_h = max(60, h // 6)
    block_w = max(80, w // 6)
    local_top = 2

    for y0 in range(0, h, block_h):
        y1 = min(h, y0 + block_h)
        band = row_density[y0:y1]
        if band.size == 0:
            continue
        local_rows = np.argsort(band)[::-1][: min(local_top, band.size)]
        for lr in local_rows:
            row_set.add(int(y0 + int(lr)))

    for x0 in range(0, w, block_w):
        x1 = min(w, x0 + block_w)
        band = col_density[x0:x1]
        if band.size == 0:
            continue
        local_cols = np.argsort(band)[::-1][: min(local_top, band.size)]
        for lc in local_cols:
            col_set.add(int(x0 + int(lc)))

    row_candidates = np.asarray(sorted(row_set, key=lambda r: row_density[int(r)], reverse=True)[:240], dtype=int)
    col_candidates = np.asarray(sorted(col_set, key=lambda c: col_density[int(c)], reverse=True)[:240], dtype=int)

    candidates: list[AxisPair] = []

    def _collect(
        *,
        row_min_frac: float | None,
        col_max_frac: float | None,
        row_run_min: float,
        col_run_min: float,
        inter_min: float,
        tr_min: float,
        score_min: float,
        embedded_mode: bool,
    ) -> None:
        for r in row_candidates:
            if row_min_frac is not None and r < int(row_min_frac * h):
                continue
            row_mask = dark[r, :]

            for c in col_candidates:
                if col_max_frac is not None and c > int(col_max_frac * w):
                    continue
                if not bool(row_mask[c]):
                    continue
                row_bounds = _run_bounds_1d(row_mask, int(c))
                if row_bounds is None:
                    continue
                row_left, row_right = row_bounds
                row_run = (row_right - row_left + 1) / max(1.0, w)
                if row_run < row_run_min:
                    continue
                left_len = int(c - row_left)
                right_len = int(row_right - c)
                if right_len < max(18, int(0.08 * w)):
                    continue
                if right_len < int(1.20 * max(1, left_len)):
                    continue

                col_mask = dark[:, c]
                if not bool(col_mask[r]):
                    continue
                col_bounds = _run_bounds_1d(col_mask, int(r))
                if col_bounds is None:
                    continue
                col_top, col_bottom = col_bounds
                col_run = (col_bottom - col_top + 1) / max(1.0, h)
                if col_run < col_run_min:
                    continue
                up_len = int(r - col_top)
                down_len = int(col_bottom - r)
                if up_len < max(14, int(0.06 * h)):
                    continue
                if down_len > max(4, int(0.22 * max(1, up_len))):
                    continue
                if r < int(col_top + 0.55 * max(1, col_bottom - col_top + 1)):
                    continue

                inter = dark[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]
                inter_strength = float(inter.mean()) if inter.size else 0.0
                if inter_strength < inter_min:
                    continue

                tr = dark[max(0, col_top) : max(0, r - 1), min(w - 1, c + 1) : min(w, row_right + 1)]
                tr_density = float(tr.mean()) if tr.size else 0.0
                if tr_density < tr_min:
                    continue

                horiz_exact, vert_exact = _axis_exactness_score(dark, int(r), int(c), int(row_right), int(col_top))
                if horiz_exact < 0.40 or vert_exact < 0.40:
                    continue

                x0 = max(0, c + 2)
                y1 = max(1, r - 1)
                if embedded_mode:
                    x1 = min(w - 1, row_right)
                    y0 = max(0, col_top)
                else:
                    x1 = min(w - 1, max(row_right, int(0.80 * w)))
                    y0 = max(0, min(col_top, int(0.10 * h)))

                bw = x1 - x0
                bh = y1 - y0
                if bw < 40 or bh < 35:
                    continue
                if embedded_mode:
                    area_ratio = float(bw * bh) / max(1.0, float(w * h))
                    if bw > int(0.40 * w) or bh > int(0.42 * h):
                        continue
                    if area_ratio < 0.003 or area_ratio > 0.11:
                        continue

                score = (
                    2.2 * row_run
                    + 2.1 * col_run
                    + 1.2 * inter_strength
                    + 2.0 * tr_density
                    + 1.5 * horiz_exact
                    + 1.5 * vert_exact
                )
                if score < score_min:
                    continue
                candidates.append(
                    AxisPair(
                        x_axis_row=int(r),
                        y_axis_col=int(c),
                        x_right=int(x1),
                        y_top=int(y0),
                        plot_bbox=(int(x0), int(y0), int(x1), int(y1)),
                        score=float(score),
                    )
                )

    # Pass 1: strict full-image axes (legacy behavior).
    _collect(
        row_min_frac=0.30,
        col_max_frac=0.70,
        row_run_min=0.42,
        col_run_min=0.34,
        inter_min=0.15,
        tr_min=0.008,
        score_min=1.8,
        embedded_mode=False,
    )

    # Pass 2: relaxed embedded-panel axes for montage figures.
    _collect(
        row_min_frac=None,
        col_max_frac=None,
        row_run_min=0.08,
        col_run_min=0.08,
        inter_min=0.12,
        tr_min=0.006,
        score_min=1.55,
        embedded_mode=True,
    )

    if not candidates:
        return []

    # Keep highest-score, non-overlapping panel candidates.
    selected: list[AxisPair] = []
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        if cand.score < 1.55:
            continue
        if any(_bbox_iou(cand.plot_bbox, s.plot_bbox) > 0.45 for s in selected):
            continue
        selected.append(cand)
        if len(selected) >= max(1, max_pairs):
            break
    return selected


def _detect_axis_pair(gray: np.ndarray) -> AxisPair | None:
    pairs = _detect_axis_pairs(gray, max_pairs=1)
    if not pairs:
        return None
    return pairs[0]


def _is_likely_plot(rgb: np.ndarray, page_text: str = "") -> bool:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return False

    h, w = rgb.shape[:2]
    if h < 160 or w < 200:
        return False

    gray = rgb.mean(axis=2)
    dark = gray < 170
    dark_ratio = float(dark.mean())

    if dark_ratio < 0.004 or dark_ratio > 0.80:
        return False

    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_map = (gx[: gy.shape[0], :] + gy[:, : gx.shape[1]]) > 45
    edge_density = float(edge_map.mean())

    x_hist = dark.sum(axis=0) / max(h, 1)
    y_hist = dark.sum(axis=1) / max(w, 1)
    left_band = float(np.max(x_hist[: max(6, w // 9)])) if w > 0 else 0.0
    bottom_band = float(np.max(y_hist[h - max(6, h // 9) :])) if h > 0 else 0.0
    right_band = float(np.max(x_hist[max(0, w - max(6, w // 9)) :])) if w > 0 else 0.0
    top_band = float(np.max(y_hist[: max(6, h // 9)])) if h > 0 else 0.0

    color_std = float(np.std(rgb.reshape(-1, 3), axis=0).mean() / 255.0)

    vert_strength = float(np.max(x_hist[: max(8, int(w * 0.30))])) if w > 0 else 0.0
    horiz_strength = float(np.max(y_hist[max(0, int(h * 0.55)) :])) if h > 0 else 0.0
    # Reject likely photo/micrograph textures lacking axis signatures.
    if edge_density > 0.09 and color_std > 0.16 and vert_strength < 0.22 and horiz_strength < 0.22:
        return False

    gray_u8 = rgb.mean(axis=2).astype(np.uint8)
    axis_pair = _detect_axis_pair(gray_u8)
    if axis_pair is None:
        return False
    bbox = axis_pair.plot_bbox
    x0, y0, x1, y1 = bbox

    x_axis_density, y_axis_density, x_axis_run, y_axis_run = _axis_continuity(gray_u8, bbox)
    # High-recall geometry gate for both L-shaped and boxed axes.
    if max(x_axis_density, y_axis_density) < 0.10:
        return False
    if x_axis_run < 0.24 and y_axis_run < 0.22:
        return False

    # Core x-y requirement: must be traceable as a curve-like signal.
    traced = _extract_dark_fallback(gray_u8, bbox)
    trace_score = 0
    if traced is not None:
        x_trace, _ = traced
        if len(x_trace) >= 25:
            trace_score = 3
        elif len(x_trace) >= 16:
            trace_score = 2

    text_score = _xy_text_score(page_text)

    score = 0
    if edge_density > 0.015:
        score += 1
    if left_band > 0.10:
        score += 1
    if bottom_band > 0.08:
        score += 1
    if right_band > 0.08:
        score += 1
    if top_band > 0.06:
        score += 1
    if color_std > 0.05:
        score += 1
    if 0.01 <= dark_ratio <= 0.50:
        score += 1
    if vert_strength > 0.22:
        score += 1
    if horiz_strength > 0.22:
        score += 1

    score += trace_score
    score += text_score
    if x_axis_run > 0.56:
        score += 1
    if y_axis_run > 0.50:
        score += 1
    if axis_pair.score > 2.2:
        score += 1

    # If text cues are absent, still require some geometric strength.
    if text_score == 0 and axis_pair.score < 2.0 and (x_axis_run < 0.35 and y_axis_run < 0.30):
        return False

    return score >= 6


def _extract_page_caption(page_text: str) -> str:
    flat_text = re.sub(r"\s+", " ", (page_text or "")).strip()
    if flat_text:
        m_fig = re.search(
            r"(fig(?:ure)?\.?\s*\d+[a-zA-Z]?\s*[\.:)]?\s*(?:\([a-zA-Z]\)\s*)?.{20,320})",
            flat_text,
            flags=re.IGNORECASE,
        )
        if m_fig:
            return m_fig.group(1).strip()[:260]
        m_num = re.search(r"(\d+\s*[\.\):]\s*(?:\([a-zA-Z]\)\s*)?.{20,320})", flat_text)
        if m_num:
            return m_num.group(1).strip()[:260]

    lines = [ln.strip() for ln in (page_text or "").splitlines() if ln.strip()]
    if not lines:
        return "x-y figure"

    def is_caption_start(line: str) -> bool:
        if re.match(r"^(fig(?:ure)?\.?\s*\d+[a-zA-Z]?)", line, flags=re.IGNORECASE):
            return True
        if re.match(r"^\d+\s*[\.\):]\s*(?:\([a-zA-Z]\)\s*)?", line):
            return True
        low = line.lower()
        if "(a)" in low and "(b)" in low and len(line) >= 30:
            return True
        return False

    for i, line_s in enumerate(lines):
        if not is_caption_start(line_s):
            continue
        block = [line_s]
        j = i + 1
        while j < len(lines) and len(" ".join(block)) < 320:
            nxt = lines[j]
            low = nxt.lower()
            if is_caption_start(nxt):
                break
            if low.startswith("doi") or low.startswith("received") or low.startswith("accepted"):
                break
            if "copyright" in low:
                break
            block.append(nxt)
            j += 1
        return re.sub(r"\s+", " ", " ".join(block)).strip()[:260]

    best = ""
    best_score = -10_000
    for line_s in lines:
        low = line_s.lower()
        if "pubs.aip.org" in low or low.startswith("apl materials") or low == "article":
            continue
        if low.startswith("doi") or low.startswith("received") or low.startswith("accepted"):
            continue
        if "copyright" in low:
            continue

        score = 0
        if 30 <= len(line_s) <= 240:
            score += 4
        elif 20 <= len(line_s) <= 320:
            score += 2
        if re.search(r"\([a-zA-Z]\)", line_s):
            score += 1
        if any(k in low for k in ("vs.", "strain", "stress", "temperature", "time", "phase", "curve")):
            score += 2
        if line_s[0].islower():
            score -= 3
        if re.match(r"^[A-Z]\s+AL\.", line_s):
            score -= 2

        if score > best_score:
            best_score = score
            best = line_s

    if best:
        return best[:260]
    return "x-y figure"


def _extract_caption_context(text: str, max_blocks: int = 3, max_chars: int = 2500) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    if not lines:
        return ""

    def is_caption_start(line: str) -> bool:
        if re.match(r"^(fig(?:ure)?\.?\s*\d+[a-zA-Z]?)", line, flags=re.IGNORECASE):
            return True
        if re.match(r"^\d+\s*[\.\):]\s*(?:\([a-zA-Z]\)\s*)?", line):
            return True
        low = line.lower()
        if "(a)" in low and "(b)" in low and len(line) >= 30:
            return True
        return False

    blocks: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln:
            i += 1
            continue
        if not is_caption_start(ln):
            i += 1
            continue

        block_lines = [ln]
        j = i + 1
        while j < len(lines) and len(block_lines) < 30:
            nxt = lines[j].strip()
            if not nxt:
                break
            low = nxt.lower()
            if is_caption_start(nxt):
                break
            if low.startswith("doi") or low.startswith("received") or low.startswith("accepted"):
                break
            if "copyright" in low:
                break
            block_lines.append(nxt)
            if len(" ".join(block_lines)) > 2400:
                break
            j += 1

        block = re.sub(r"\s+", " ", " ".join(block_lines)).strip()
        if block:
            blocks.append(block)
            if len(blocks) >= max_blocks:
                break
        i = j + 1

    if not blocks:
        cap = _extract_page_caption(text)
        return cap if cap != "x-y figure" else ""

    joined = " ".join(blocks)
    return joined[:max_chars]


def _clean_axis_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" :;,-")
    name = re.sub(r"^(fig(?:ure)?\.?\s*)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^(fig(?:ure)?\.?\s*\d+[a-zA-Z]?\s*[\)\].:-]?\s*)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^\([a-zA-Z]\)\s*", "", name)
    name = re.sub(r"^[\d\W]+", "", name)
    if name.lower() in {"x-y figure", "xy figure", "figure"}:
        return "x"
    return name[:60] if name else "axis"


def _clean_unit(unit: str) -> str:
    unit = re.sub(r"\s+", "", unit).strip("[]{}")
    return _canonicalize_unit_output(unit)[:32] if unit else "arb"


def infer_axis_labels_units(page_text: str) -> tuple[str, str, str, str]:
    axis_pairs: list[tuple[str, str]] = []
    pair_pat = re.compile(r"([A-Za-z][A-Za-z0-9\-/,%\.\s]{1,40})\(([^()\n]{1,18})\)")

    for line in page_text.splitlines():
        line_s = line.strip()
        if not line_s:
            continue
        if re.match(r"^(fig(?:ure)?\.?\s*\d+)", line_s, flags=re.IGNORECASE):
            continue
        for m in pair_pat.finditer(line_s):
            name = _clean_axis_name(m.group(1))
            unit = _clean_unit(m.group(2))
            if len(name) < 2:
                continue
            axis_pairs.append((name, unit))

    unique_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in axis_pairs:
        if pair in seen:
            continue
        seen.add(pair)
        unique_pairs.append(pair)

    if len(unique_pairs) >= 2:
        (x_name, x_unit), (y_name, y_unit) = unique_pairs[0], unique_pairs[1]
        return x_name, x_unit, y_name, y_unit

    keywords_x = ["temperature", "time", "strain", "voltage", "frequency", "wavelength", "pressure"]
    keywords_y = ["intensity", "stress", "current", "resistance", "conductivity", "modulus", "energy"]
    low = page_text.lower()

    x_name = "x"
    y_name = "y"
    for kw in keywords_x:
        if kw in low:
            x_name = kw
            break
    for kw in keywords_y:
        if kw in low:
            y_name = kw
            break

    return x_name, "arb", y_name, "arb"


def _parse_numeric_token(token: str) -> float | None:
    t = token.strip().replace(",", "")
    t = t.replace("−", "-")
    t = re.sub(r"^[^\d+\-\.eE]+|[^\d\.eE+\-]+$", "", t)
    if not t:
        return None
    try:
        return float(t)
    except Exception:
        return None


def _fit_linear_scale(samples: list[tuple[float, float]]) -> tuple[float, float] | None:
    if len(samples) < 3:
        return None
    x = np.asarray([s[0] for s in samples], dtype=float)
    y = np.asarray([s[1] for s in samples], dtype=float)
    if len(np.unique(x)) < 2:
        return None

    if len(np.unique(np.round(y, 6))) < 2:
        return None

    corr = float(np.corrcoef(x, y)[0, 1]) if x.size >= 3 else 0.0
    if not np.isfinite(corr) or abs(corr) < 0.65:
        return None

    a, b = np.polyfit(x, y, 1)
    if not np.isfinite(a) or not np.isfinite(b):
        return None
    if abs(a) < 1e-8:
        return None

    pred = a * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot < 1e-12 else 1.0 - (ss_res / ss_tot)
    if not np.isfinite(r2) or r2 < 0.72:
        return None

    if abs(a) > 1e6:
        return None
    return float(a), float(b)


def _px_to_page(rect: fitz.Rect, px: float, py: float, img_w: int, img_h: int) -> tuple[float, float]:
    pxn = float(px) / max(1.0, img_w - 1)
    pyn = float(py) / max(1.0, img_h - 1)
    return rect.x0 + pxn * rect.width, rect.y0 + pyn * rect.height


def _page_to_px(rect: fitz.Rect, x: float, y: float, img_w: int, img_h: int) -> tuple[float, float]:
    xn = (float(x) - rect.x0) / max(1e-9, rect.width)
    yn = (float(y) - rect.y0) / max(1e-9, rect.height)
    return xn * (img_w - 1), yn * (img_h - 1)


def _safe_float_bbox(vals: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(vals, (list, tuple)) or len(vals) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in vals]
    except Exception:
        return None
    if not all(np.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _axis_context_signal_score(text: str) -> int:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return -10
    low = t.lower()
    score = 0
    if re.search(r"[A-Za-z][A-Za-z0-9/\-\s]{1,40}[\(\[]\s*[A-Za-z0-9%/\-\^\.]{1,14}\s*[\)\]]", t):
        score += 3
    if re.search(r"\bvs\.?\b|dependence of|arrhenius|transient|spectra|spectrum", low):
        score += 2
    axis_terms = [
        "time",
        "temperature",
        "strain",
        "stress",
        "pressure",
        "magnetic field",
        "voltage",
        "current",
        "frequency",
        "wavelength",
        "energy",
        "intensity",
        "resistance",
        "conductivity",
        "modulus",
        "phase",
        "cross section",
        "degradation",
        "absorbance",
        "transmittance",
    ]
    score += sum(1 for k in axis_terms if k in low)
    if "pubs.aip.org" in low:
        score -= 4
    if " article " in f" {low} ":
        score -= 2
    if "copyright" in low:
        score -= 2
    if len(re.findall(r"[A-Za-z]{3,}", t)) < 3:
        score -= 1
    return int(score)


def _axis_pair_local_context(
    page: fitz.Page,
    image_rect: fitz.Rect | None,
    img_w: int,
    img_h: int,
    axis_pair: AxisPair,
    fallback_text: str,
    pad_mult: float = 0.9,
) -> str:
    if image_rect is None:
        return fallback_text
    x0, y0, x1, y1 = axis_pair.plot_bbox
    p0 = _px_to_page(image_rect, x0, y0, img_w, img_h)
    p1 = _px_to_page(image_rect, x1, y1, img_w, img_h)
    bx0 = min(p0[0], p1[0])
    by0 = min(p0[1], p1[1])
    bx1 = max(p0[0], p1[0])
    by1 = max(p0[1], p1[1])
    bw = max(1.0, bx1 - bx0)
    bh = max(1.0, by1 - by0)
    pad_x = max(14.0, bw * pad_mult)
    pad_y = max(14.0, bh * pad_mult)
    # Pull context across full page width to avoid truncated caption/title starts.
    # Favor more vertical capture so full figure-title/caption lines are retained.
    top_pad = max(pad_y * 1.9, 64.0)
    bottom_pad = max(pad_y * 1.0, 24.0)
    clip = fitz.Rect(
        page.rect.x0,
        max(page.rect.y0, by0 - top_pad),
        page.rect.x1,
        min(page.rect.y1, by1 + bottom_pad),
    )
    txt = page.get_text("text", clip=clip) or ""
    if not txt.strip():
        return fallback_text
    local_score = _axis_context_signal_score(txt)
    fallback_score = _axis_context_signal_score(fallback_text)
    if fallback_score >= local_score + 2:
        return fallback_text
    return txt


def _words_to_text(words: list[tuple], key_axis: int = 0) -> str:
    if not words:
        return ""
    ordered = sorted(words, key=lambda w: (round(float(w[1]), 1), float(w[key_axis])))
    joined = " ".join(str(w[4]) for w in ordered if str(w[4]).strip())
    return re.sub(r"\s+", " ", joined).strip()


def _merge_caption_context(primary_text: str, caption_text: str) -> str:
    primary = (primary_text or "").strip()
    caption = (caption_text or "").strip()
    if not caption:
        return primary
    if not primary:
        return caption
    if caption in primary:
        return primary
    return f"{primary}\n{caption}"


def _clean_axis_label_phrase(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    t = t.strip(" ,;:.-")
    return t


def _is_axis_semantic_label(name: str) -> bool:
    n = _clean_axis_name(name)
    low = n.lower()
    if not n:
        return False
    # Reject common chemical formulas/material names that frequently leak from figure text.
    if re.fullmatch(r"(?:[A-Z][a-z]?\d*){2,}", n) and not _has_axis_keyword(low):
        return False
    if _has_axis_keyword(low):
        return True
    key_hits = [
        "vth",
        "gate voltage",
        "drain current",
        "current density",
        "cross section",
        "degradation",
        "efficiency",
        "ratio",
        "index",
        "thickness",
        "distance",
        "position",
        "phase",
        "absorbance",
        "transmittance",
        "delta k",
        "n_b",
        "particle size",
        "diameter",
        "hydrodynamic size",
    ]
    if any(k in low for k in key_hits):
        return True
    if low in {"nb", "n_b", "delta k", "δk", "∆k", "δκ", "∆κ", "Δκ".lower()}:
        return True
    if re.fullmatch(r"[A-Za-z]{1,4}_[A-Za-z0-9]{1,4}", n):
        return True
    if re.fullmatch(r"(?:delta\s*[A-Za-z]|[Δ∆δ]\s*[A-Za-z\u03baκ])", n, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"[A-Za-z\u0394\u03b4\u03bc\u00b5]{1,4}(?:/[A-Za-z\u0394\u03b4\u03bc\u00b5]{1,4})?", n):
        return True
    if re.fullmatch(r"[A-Za-z]{1,5}\d{0,2}", n):
        return True
    if re.fullmatch(r"[A-Za-z]{1,3}[A-Za-z0-9]{0,4}\s*/\s*[A-Za-z]{1,3}[A-Za-z0-9]{0,4}", n):
        return True
    return False


def _is_probable_unit(unit: str) -> bool:
    if not unit:
        return False
    u = unit.strip()
    if not u:
        return False
    if _looks_like_link_or_doi(u):
        return False
    if _is_noise_unit_token(u):
        return False
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", u):
        return False
    common = {
        "k",
        "c",
        "degc",
        "s",
        "sec",
        "min",
        "h",
        "ms",
        "us",
        "ns",
        "hz",
        "khz",
        "mhz",
        "ghz",
        "ev",
        "mev",
        "j",
        "kj",
        "pa",
        "kpa",
        "mpa",
        "gpa",
        "v",
        "mv",
        "a",
        "ma",
        "ua",
        "μa",
        "µa",
        "ohm",
        "cm",
        "mm",
        "nm",
        "um",
        "μm",
        "µm",
        "deg",
        "°",
        "rad",
        "%",
        "a.u.",
        "a.u",
        "au",
        "s/m",
        "ev-1",
        "cm-2",
        "cm^-2",
        "cycle",
        "cycles",
        "count",
        "counts",
    }
    ul = u.lower()
    if ul in common:
        return True
    # Pure alphabetic words are usually not units unless known from standards/mapping.
    if re.fullmatch(r"[A-Za-z]+", u):
        if ul in {"cycle", "cycles", "count", "counts"}:
            return True
        ukey = _normalize_unit_key(u)
        rev = _load_unit_to_standard_labels()
        if ukey in rev:
            return True
        if len(ukey) > 1:
            for pfx in ("k", "m", "g", "u", "n", "p"):
                if ukey.startswith(pfx) and ukey[1:] in rev:
                    return True
        if len(u) == 1:
            return u in {"K", "C", "V", "A", "J", "N", "W", "T", "%", "s", "m", "g"}
        return False
    if len(u) == 1:
        return u in {"K", "C", "V", "A", "J", "N", "W", "T", "%", "s", "m", "g"}
    if len(u) <= 6 and re.fullmatch(r"[A-Za-z0-9%/\-\^\.\u00b5\u03a9]+", u) and re.search(r"[A-Za-z%\u00b5\u03a9]", u):
        return True
    if re.search(r"[/%\^]", u):
        return True
    if re.search(r"(cm|mm|nm|um|μm|µm|kg|g|mol|ev|hz|pa|v|a|ohm|deg|rad|min|sec|hr)", ul):
        return True
    return False


def _is_plausible_axis_label(name: str) -> bool:
    if not name:
        return False
    n = _clean_axis_name(name)
    if _looks_like_link_or_doi(n):
        return False
    if len(n) < 1 or len(n) > 36:
        return False
    words = n.split()
    if len(words) > 4:
        return False
    # Accept Latin and common Greek scientific symbols in axis labels (e.g., Δκ).
    if not any(re.search(r"[A-Za-z\u0370-\u03FF]", w) for w in words):
        return False
    banned = {
        "with",
        "and",
        "of",
        "in",
        "on",
        "to",
        "for",
        "the",
        "from",
        "then",
        "according",
        "eq",
        "equation",
        "table",
        "state",
        "states",
        "initial",
        "possible",
        "pubs",
        "article",
        "compares",
        "compare",
        "formal",
        "analysis",
        "curation",
        "lead",
        "author",
        "copyright",
        "axis",
        "label",
        "panel",
        "sample",
        "substrate",
        "org",
        "com",
        "net",
        "www",
        "http",
        "https",
        "doi",
    }
    if any(w.lower() in banned for w in words):
        return False
    low = n.lower()
    banned_phrases = [
        "according to",
        "fitting parameter",
        "x-y figure",
        "possible surface",
        "vicinal",
        "pubs.aip",
        "doi.org",
        "https",
        "http",
    ]
    if any(p in low for p in banned_phrases):
        return False
    return True


def _label_unit_from_phrase(text: str, default_label: str, default_unit: str) -> tuple[str, str, str]:
    t = _clean_axis_label_phrase(_strip_links_and_doi(text))
    if not t:
        return default_label, default_unit, "default"

    # Prefer explicit "label(unit)" / "label [unit]".
    m = re.search(r"(.{2,80}?)[\(\[]\s*([A-Za-z0-9%/\-\^\.\u00b5\u03a9]+)\s*[\)\]]", t)
    if m:
        name = _extract_label_before_delimiter(m.group(1))
        unit = _clean_unit(m.group(2))
        if name.lower() in {"band", "bands"} and unit.lower() == "s":
            return default_label, default_unit, "default"
        if _is_plausible_axis_label(name):
            if _is_probable_unit(unit):
                return name, unit, "explicit"
            return name, default_unit, "default"

    # Support "label/unit" notation when unit is shown after a forward slash.
    m = re.search(
        r"(.{1,96}?)\s*/\s*([A-Za-z\u00b5\u03bc\u03a9%][A-Za-z0-9\u00b5\u03bc\u03a9%/\-\^\.]{0,20})(?:\b|$)",
        t,
    )
    if m:
        name = _extract_label_before_delimiter(m.group(1))
        unit = _clean_unit(m.group(2))
        if _is_plausible_axis_label(name) and _is_probable_unit(unit):
            return name, unit, "slash"

    # Fallback: if trailing token looks like a unit, split it.
    toks = t.split()
    if len(toks) >= 2:
        tail = toks[-1]
        if _is_probable_unit(tail):
            name = _extract_label_before_delimiter(" ".join(toks[:-1]))
            if _is_plausible_axis_label(name):
                return name, _clean_unit(tail), "suffix"

    name = _trim_panel_token_prefix(_clean_axis_name(t))
    if not _is_plausible_axis_label(name):
        return default_label, default_unit, "default"
    if not _is_axis_semantic_label(name):
        return default_label, default_unit, "default"
    return name, default_unit, "default"


def _extract_vertical_label(words: list[tuple], x_tol: float = 8.0) -> str:
    if not words:
        return ""

    single_chars = []
    for w in words:
        txt = str(w[4]).strip()
        if len(txt) == 1 and re.match(r"[A-Za-z0-9%\-\u00b5\u03a9]", txt):
            single_chars.append(w)
    if len(single_chars) < 4:
        return ""

    xs = np.asarray([0.5 * (float(w[0]) + float(w[2])) for w in single_chars], dtype=float)
    x_med = float(np.median(xs))
    cluster = [w for w, x in zip(single_chars, xs) if abs(float(x) - x_med) <= x_tol]
    if len(cluster) < 4:
        return ""

    cluster_sorted = sorted(cluster, key=lambda w: float(w[1]))
    text = "".join(str(w[4]).strip() for w in cluster_sorted)
    return _clean_axis_label_phrase(text)


def _extract_parallel_axis_text(
    words: list[tuple],
    *,
    orientation: str,
    axis_coord: float,
    axis_min: float,
    axis_max: float,
    image_span_x: float,
    image_span_y: float,
) -> str:
    if not words:
        return ""

    def _collect(horizontal_relaxed: bool = False) -> list[tuple]:
        selected_local: list[tuple] = []
        if orientation == "horizontal":
            if horizontal_relaxed:
                band_top = axis_coord - 0.02 * image_span_y
                band_bottom = axis_coord + 0.38 * image_span_y
                x_pad = 0.08 * image_span_x
            else:
                band_top = axis_coord + 0.015 * image_span_y
                band_bottom = axis_coord + 0.24 * image_span_y
                x_pad = 0.02 * image_span_x
            for w in words:
                wx0, wy0, wx1, wy1 = float(w[0]), float(w[1]), float(w[2]), float(w[3])
                text = str(w[4]).strip()
                if not text or _parse_numeric_token(text) is not None:
                    continue
                cx = 0.5 * (wx0 + wx1)
                cy = 0.5 * (wy0 + wy1)
                width = wx1 - wx0
                height = wy1 - wy0
                if cx < axis_min - x_pad or cx > axis_max + x_pad:
                    continue
                if cy < band_top or cy > band_bottom:
                    continue
                if not horizontal_relaxed and width < 0.90 * height:
                    continue
                selected_local.append(w)
            return selected_local

        if horizontal_relaxed:
            band_left = axis_coord - 0.34 * image_span_x
            band_right = axis_coord + 0.03 * image_span_x
            y_pad_top = 0.08 * image_span_y
            y_pad_bottom = 0.08 * image_span_y
        else:
            band_left = axis_coord - 0.22 * image_span_x
            band_right = axis_coord - 0.01 * image_span_x
            y_pad_top = 0.03 * image_span_y
            y_pad_bottom = 0.02 * image_span_y
        for w in words:
            wx0, wy0, wx1, wy1 = float(w[0]), float(w[1]), float(w[2]), float(w[3])
            text = str(w[4]).strip()
            if not text or _parse_numeric_token(text) is not None:
                continue
            cx = 0.5 * (wx0 + wx1)
            cy = 0.5 * (wy0 + wy1)
            width = wx1 - wx0
            height = wy1 - wy0
            if cy < axis_min - y_pad_top or cy > axis_max + y_pad_bottom:
                continue
            if cx < band_left or cx > band_right:
                continue
            if not horizontal_relaxed and height < 0.85 * width and len(text) <= 2:
                continue
            selected_local.append(w)
        return selected_local

    selected = _collect(horizontal_relaxed=False)
    if orientation == "horizontal":
        text = _words_to_text(selected, key_axis=0)
        if text:
            return text
        return _words_to_text(_collect(horizontal_relaxed=True), key_axis=0)

    vertical = _extract_vertical_label(selected, x_tol=max(6.0, 0.025 * image_span_x))
    if len(vertical) >= 3:
        return vertical
    text = _words_to_text(selected, key_axis=1)
    if text:
        return text
    selected_relaxed = _collect(horizontal_relaxed=True)
    vertical_relaxed = _extract_vertical_label(selected_relaxed, x_tol=max(8.0, 0.04 * image_span_x))
    if len(vertical_relaxed) >= 3:
        return vertical_relaxed
    return _words_to_text(selected_relaxed, key_axis=1)


def _group_words_into_lines(words: list[tuple], y_tol: float = 12.0) -> list[list[tuple]]:
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (float(w[1]), float(w[0])))
    lines: list[list[tuple]] = []
    for w in ordered:
        cy = 0.5 * (float(w[1]) + float(w[3]))
        if not lines:
            lines.append([w])
            continue
        prev = lines[-1]
        prev_cy = float(np.mean([0.5 * (float(p[1]) + float(p[3])) for p in prev]))
        if abs(cy - prev_cy) <= y_tol:
            prev.append(w)
        else:
            lines.append([w])
    return lines


def _best_axis_phrase_candidate(words: list[tuple], default_label: str, span_x: float, vertical_ok: bool = False) -> tuple[str, str, str] | None:
    candidates: list[tuple[int, int, str, str, str]] = []
    for line in _group_words_into_lines(words, y_tol=max(10.0, 0.018 * span_x)):
        phrase = _words_to_text(line, key_axis=0)
        if not phrase:
            continue
        lab, unit, src = _label_unit_from_phrase(phrase, default_label, "arb")
        score = 0
        if src in {"explicit", "slash", "suffix"}:
            score += 4
        if lab != default_label:
            score += 2
        if unit != "arb":
            score += 2
        if lab != default_label and _is_axis_semantic_label(lab):
            score += 2
        if score > 0:
            candidates.append((score, len(line), lab, unit, src))

    if vertical_ok:
        vertical_phrase = _extract_vertical_label(words, x_tol=max(8.0, 0.03 * span_x))
        if vertical_phrase:
            lab, unit, src = _label_unit_from_phrase(vertical_phrase, default_label, "arb")
            score = 0
            if src in {"explicit", "slash", "suffix"}:
                score += 4
            if lab != default_label:
                score += 2
            if unit != "arb":
                score += 2
            if lab != default_label and _is_axis_semantic_label(lab):
                score += 2
            if score > 0:
                candidates.append((score, max(4, len(vertical_phrase)), lab, unit, src))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, lab, unit, src = candidates[0]
    return lab, unit, src


def _guess_unit_from_label(label: str) -> str:
    std_unit = _lookup_standard_unit(label)
    if std_unit:
        return std_unit

    key = (label or "").lower()
    mapping = {
        "temperature": "K",
        "time": "s",
        "duration": "s",
        "strain": "%",
        "stress": "MPa",
        "modulus": "GPa",
        "pressure": "Pa",
        "magnetic field": "T",
        "field": "T",
        "magnetization/ms": "-",
        "magnetization": "a.u.",
        "dm/dh": "a.u.",
        "voltage": "V",
        "vth": "V",
        "vg": "V",
        "vgs": "V",
        "vds": "V",
        "current": "A",
        "id": "A",
        "ig": "A",
        "drain current": "A",
        "frequency": "Hz",
        "wavelength": "nm",
        "wavenumber": "cm^-1",
        "photon energy": "eV",
        "energy": "eV",
        "resistance": "ohm",
        "conductivity": "S/m",
        "intensity": "a.u.",
        "absorbance": "a.u.",
        "transmittance": "%",
        "phase": "deg",
        "epsc": "uA",
        "degradation efficiency": "%",
        "efficiency": "%",
        "current density": "A/cm^2",
        "threshold voltage": "V",
        "gate voltage": "V",
        "ppf index": "-",
        "index": "-",
        "distance": "um",
        "position": "um",
        "particle size": "nm",
        "diameter": "nm",
        "hydrodynamic size": "nm",
        "cross section": "cm^-2",
        "photoionization cross section": "cm^-2",
        "n_b": "cycles",
        "nb": "cycles",
        "number of cycles": "cycles",
    }
    for k, unit in mapping.items():
        if k in key:
            return _canonicalize_unit_output(unit)
    return "arb"


def _has_axis_keyword(text: str) -> bool:
    low = (text or "").lower()
    keys = [
        "temperature",
        "time",
        "strain",
        "stress",
        "pressure",
        "magnetic field",
        "field",
        "voltage",
        "current",
        "frequency",
        "wavelength",
        "wavenumber",
        "energy",
        "magnetization",
        "dm/dh",
        "intensity",
        "transmittance",
        "absorbance",
        "resistance",
        "conductivity",
        "modulus",
        "size",
        "diameter",
        "particle size",
        "hydrodynamic size",
    ]
    return any(k in low for k in keys)


def _extract_length_unit_from_text(text: str) -> str | None:
    t = re.sub(r"\s+", " ", text or "")

    def norm_unit(raw: str) -> str:
        return _canonicalize_unit_output(raw)

    best: tuple[float, str] | None = None

    patterns = [
        (9.0, r"\b\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\s*(μm|µm|um|nm|mm|cm)\s*(?:\^?\s*2)?"),
        (8.0, r"scale\s*bar[^.,;:]{0,40}?(μm|µm|um|nm|mm|cm)"),
        (7.0, r"(?:region|area|window|scan)[^.,;:]{0,36}?(μm|µm|um|nm|mm|cm)\s*(?:\^?\s*2)?"),
    ]

    for base_score, pat in patterns:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            unit = norm_unit(m.group(1))
            score = base_score
            span = m.group(0).lower()
            if "scale" in span or "bar" in span:
                score += 1.5
            if "x" in span or "×" in span:
                score += 1.0
            cand = (score, unit)
            if best is None or cand[0] > best[0]:
                best = cand
    if best is None:
        return None
    return best[1]


def _infer_axis_from_context_text(text: str, panel_hint: str | None = None) -> tuple[str, str, str, str]:
    low_full = text.lower()
    low_norm = re.sub(r"\s+", " ", low_full)

    # Panel-aware caption parsing for multi-panel figures like "(a) ... (b) ...".
    panel_a = ""
    panel_b = ""
    ma = re.search(r"\(a\)\s*(.*?)(?=\([b-z]\)|$)", text, flags=re.IGNORECASE | re.DOTALL)
    mb = re.search(r"\(b\)\s*(.*?)(?=\([c-z]\)|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if ma:
        panel_a = _clean_axis_label_phrase(ma.group(1))
    if mb:
        panel_b = _clean_axis_label_phrase(mb.group(1))

    panel_text = low_norm
    if panel_hint == "a" and panel_a:
        panel_text = panel_a.lower()
    elif panel_hint == "b" and panel_b:
        panel_text = panel_b.lower()

    length_unit_panel = _extract_length_unit_from_text(panel_text)
    has_voltage_panel = ("voltage" in panel_text) or bool(
        re.search(r"(?<![A-Za-z0-9])[+-]?\d+(?:\.\d+)?\s*v(?![A-Za-z])", panel_text, flags=re.IGNORECASE)
    )
    if length_unit_panel and has_voltage_panel and (
        "scale bar" in panel_text or "bias" in panel_text or "pfm" in panel_text or "region" in panel_text
    ):
        return "length", length_unit_panel, "voltage", "V"

    # Domain-specific high-confidence mappings.
    if "tg/dta" in panel_text or "tga/dta" in panel_text or ("tga" in panel_text and "dta" in panel_text):
        return "temperature", "°C", "mass loss", "%"
    if "xrd" in panel_text or "2θ" in text or "2theta" in panel_text or "phi-scan" in panel_text:
        return "angle", "deg", "intensity", "a.u."
    if "ftir" in panel_text or ("transmittance" in panel_text and "wavenumber" in panel_text):
        return "wavenumber", "cm^-1", "transmittance", "%"
    if "uv-vis" in panel_text or "uv–vis" in panel_text or "absorption spectra" in panel_text:
        return "wavelength", "nm", "absorbance", "a.u."
    if "pl spectra" in panel_text:
        return "wavelength", "nm", "intensity", "a.u."
    if "photoionization cross section" in panel_text or ("cross section" in panel_text and "photon" in panel_text):
        return "photon energy", "eV", "photoionization cross section", "cm^-2"
    if ("vth" in panel_text or "threshold voltage" in panel_text) and "transient" in panel_text:
        return "time", "s", "threshold voltage", "V"
    if "ppf index" in panel_text:
        return "Δt", "s", "PPF index", "-"
    if "degradation" in panel_text and ("catalyst" in panel_text or "experiment" in panel_text):
        return "time", "min", "degradation efficiency", "%"
    if ("dynamic light scattering" in panel_text or re.search(r"\bdls\b", panel_text)) and (
        "size distribution" in panel_text or ("distribution" in panel_text and "nanoparticle" in panel_text)
    ):
        return "particle size", "nm", "intensity", "%"

    # Domain-specific fallback for magnetic-field figures.
    if re.search(r"d\s*m\s*/\s*d\s*h", panel_text):
        return "applied magnetic field", "T", "dM/dH", "a.u."
    if "magnetization" in panel_text:
        if re.search(r"m\s*/\s*m[s5]", panel_text) or "initial magnetization" in panel_text:
            return "applied magnetic field", "T", "magnetization/Ms", "-"
        return "applied magnetic field", "T", "magnetization", "a.u."

    if re.search(r"d\s*m\s*/\s*d\s*h", low_norm):
        return "applied magnetic field", "T", "dM/dH", "a.u."
    if "magnetization" in low_norm:
        if re.search(r"m\s*/\s*m[s5]", low_norm) or "initial magnetization" in low_norm:
            return "applied magnetic field", "T", "magnetization/Ms", "-"
        return "applied magnetic field", "T", "magnetization", "a.u."
    length_unit_full = _extract_length_unit_from_text(low_norm)
    has_voltage_full = ("voltage" in low_norm) or bool(
        re.search(r"(?<![A-Za-z0-9])[+-]?\d+(?:\.\d+)?\s*v(?![A-Za-z])", low_norm, flags=re.IGNORECASE)
    )
    if length_unit_full and has_voltage_full and ("scale bar" in low_norm or "bias" in low_norm or "pfm" in low_norm):
        return "length", length_unit_full, "voltage", "V"
    if "photoionization cross section" in low_norm or ("cross section" in low_norm and "photon energy" in low_norm):
        return "photon energy", "eV", "photoionization cross section", "cm^-2"
    if "ftir" in low_norm or ("transmittance" in low_norm and "wavenumber" in low_norm):
        return "wavenumber", "cm^-1", "transmittance", "%"
    if "degradation" in low_norm and ("catalyst" in low_norm or "imidacloprid" in low_norm):
        return "time", "min", "degradation efficiency", "%"
    if ("dynamic light scattering" in low_norm or re.search(r"\bdls\b", low_norm)) and (
        "size distribution" in low_norm or ("distribution" in low_norm and "nanoparticle" in low_norm)
    ):
        return "particle size", "nm", "intensity", "%"

    pairs: list[tuple[str, str]] = []
    for m in re.finditer(
        r"([A-Za-z][A-Za-z0-9\-/\s\u0394\u03bc\u00b5]{1,40})[\(\[]\s*([A-Za-z0-9%/\-\^\.\u00b5\u03a9]{1,16})\s*[\)\]]",
        text,
    ):
        name = _clean_axis_name(m.group(1))
        unit = _clean_unit(m.group(2))
        if _is_plausible_axis_label(name) and _is_probable_unit(unit):
            pairs.append((name, unit))
    if len(pairs) >= 2:
        return pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1]

    vs_m = re.search(r"([A-Za-z][A-Za-z0-9\-/\s\u0394\u03bc\u00b5]{1,35})\s+vs\.?\s+([A-Za-z][A-Za-z0-9\-/\s\u0394\u03bc\u00b5]{1,35})", text, flags=re.IGNORECASE)
    if vs_m:
        y_name = _clean_axis_name(vs_m.group(1))
        x_name = _clean_axis_name(vs_m.group(2))
        if _is_plausible_axis_label(x_name) and _is_plausible_axis_label(y_name):
            return x_name, _guess_unit_from_label(x_name), y_name, _guess_unit_from_label(y_name)

    dep_m = re.search(
        r"dependence of\s+([A-Za-z][A-Za-z0-9\-/\s\u0394\u03bc\u00b5]{1,35})\s+on\s+([A-Za-z][A-Za-z0-9\-/\s\u0394\u03bc\u00b5]{1,35})",
        low_norm,
        flags=re.IGNORECASE,
    )
    if dep_m:
        y_name = _clean_axis_name(dep_m.group(1))
        x_name = _clean_axis_name(dep_m.group(2))
        if _is_plausible_axis_label(x_name) and _is_plausible_axis_label(y_name):
            return x_name, _guess_unit_from_label(x_name), y_name, _guess_unit_from_label(y_name)

    rel_m = re.search(
        r"([A-Za-z\u0394\u2206\u03ba\u03b4][A-Za-z0-9\u0394\u2206\u03ba\u03b4/_\-]{0,14})\s*[–-]\s*([A-Za-z][A-Za-z0-9]{0,8})\s+(?:relationships?|responses?)",
        text,
        flags=re.IGNORECASE,
    )
    if rel_m:
        y_token_raw = rel_m.group(1)
        x_token_raw = rel_m.group(2)
        y_name_raw = _clean_axis_name(y_token_raw)
        x_name_raw = _clean_axis_name(x_token_raw)

        def _normalize_symbolic_axis_label(n: str) -> str:
            low_n = (n or "").lower().replace(" ", "")
            low_n = low_n.replace("∆", "δ").replace("κ", "k")
            if low_n in {"nb", "n_b"}:
                return "N_b"
            if low_n in {"δk", "dk", "deltak"} or ("delta" in low_n and "k" in low_n):
                return "delta k"
            return n

        y_name = _normalize_symbolic_axis_label(y_name_raw)
        x_name = _normalize_symbolic_axis_label(x_name_raw)
        y_token_low = (y_token_raw or "").lower()
        if ("∆" in y_token_raw or "Δ" in y_token_raw or "δ" in y_token_low) and ("κ" in y_token_raw or "k" in y_token_low):
            y_name = "delta k"
        if _is_plausible_axis_label(x_name):
            y_out = y_name if _is_plausible_axis_label(y_name) else "y"
            x_unit = "cycles" if "cyclic" in low_norm else _guess_unit_from_label(x_name)
            y_unit = _guess_unit_from_label(y_out) if y_out != "y" else "arb"
            return x_name, _canonicalize_unit_output(x_unit), y_out, _canonicalize_unit_output(y_unit)

    low = text.lower()
    def has_term(term: str) -> bool:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", low))

    x_order = [
        "time",
        "temperature",
        "frequency",
        "wavelength",
        "photon energy",
        "energy",
        "magnetic field",
        "field",
        "voltage",
        "gate voltage",
        "strain",
        "stress",
        "pressure",
        "distance",
        "position",
        "concentration",
        "thickness",
    ]
    y_order = [
        "dm/dh",
        "magnetization",
        "current density",
        "current",
        "resistance",
        "conductivity",
        "threshold voltage",
        "voltage",
        "absorbance",
        "transmittance",
        "phase",
        "ppf index",
        "epsc",
        "stress",
        "modulus",
        "energy",
        "intensity",
        "rate",
    ]
    x_label = "x"
    y_label = "y"
    for k in x_order:
        if has_term(k):
            x_label = k
            break
    for k in y_order:
        if has_term(k):
            y_label = k
            break
    return x_label, _guess_unit_from_label(x_label), y_label, _guess_unit_from_label(y_label)


def _scale_is_valid(scale: Any) -> bool:
    if not isinstance(scale, (list, tuple)) or len(scale) != 2:
        return False
    try:
        a = float(scale[0])
        b = float(scale[1])
    except Exception:
        return False
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    return 1e-8 < abs(a) <= 1e6


def _run_len_col(dark: np.ndarray, row: int, col: int, max_len: int) -> int:
    h = dark.shape[0]
    if row < 0 or row >= h or col < 0 or col >= dark.shape[1] or not dark[row, col]:
        return 0
    up = 0
    rr = row - 1
    while rr >= 0 and dark[rr, col] and up < max_len:
        up += 1
        rr -= 1
    down = 0
    rr = row + 1
    while rr < h and dark[rr, col] and down < max_len:
        down += 1
        rr += 1
    return up + down + 1


def _run_len_row(dark: np.ndarray, row: int, col: int, max_len: int) -> int:
    w = dark.shape[1]
    if row < 0 or row >= dark.shape[0] or col < 0 or col >= w or not dark[row, col]:
        return 0
    left = 0
    cc = col - 1
    while cc >= 0 and dark[row, cc] and left < max_len:
        left += 1
        cc -= 1
    right = 0
    cc = col + 1
    while cc < w and dark[row, cc] and right < max_len:
        right += 1
        cc += 1
    return left + right + 1


def _count_tick_peaks(lengths: list[int], min_sep: int = 3) -> int:
    if not lengths:
        return 0
    base = float(np.percentile(np.asarray(lengths, dtype=float), 60))
    threshold = max(3.0, base + 1.5)

    peaks = 0
    last_idx = -10_000
    for idx, ln in enumerate(lengths):
        if ln >= threshold and (idx - last_idx) >= max(1, min_sep):
            peaks += 1
            last_idx = idx
    return int(peaks)


def _estimate_axis_tick_counts(gray: np.ndarray, axis_pair: AxisPair) -> tuple[int, int]:
    h, w = gray.shape
    dark = gray < 170

    r = int(axis_pair.x_axis_row)
    c = int(axis_pair.y_axis_col)
    x0, y0, x1, y1 = axis_pair.plot_bbox

    max_v = max(8, int(0.04 * h))
    max_h = max(8, int(0.04 * w))
    min_sep_x = max(2, int(0.008 * max(1, x1 - x0)))
    min_sep_y = max(2, int(0.008 * max(1, y1 - y0)))

    col_lengths = [
        _run_len_col(dark, r, col, max_v) for col in range(max(c + 2, x0 + 1), min(w - 1, x1 - 1))
    ]
    row_lengths = [
        _run_len_row(dark, row, c, max_h) for row in range(max(1, y0 + 1), min(h - 1, y1 - 1))
    ]

    return _count_tick_peaks(col_lengths, min_sep=min_sep_x), _count_tick_peaks(row_lengths, min_sep=min_sep_y)


def _schematic_keyword_score(text: str) -> int:
    low = (text or "").lower()
    keys = [
        "schematic",
        "diagram",
        "experimental setup",
        "device structure",
        "device architecture",
        "illustration",
        "afm image",
        "afm images",
        "microstructure",
        "morphology",
        "sem image",
        "tem image",
        "optical image",
        "hrtem",
        "stm image",
        "hemt device",
        "photograph",
    ]
    score = sum(1 for k in keys if k in low)
    # "cross section" appears in valid x-y plot captions; treat as schematic only with explicit structure cues.
    if ("cross section" in low or "cross-section" in low) and (
        "schematic" in low or "device" in low or "structure" in low
    ):
        score += 1
    return score


def _axis_meta_is_strict_xy(axis_meta: dict[str, Any], context_text: str = "") -> bool:
    x_label = str(axis_meta.get("x_label") or "x").strip()
    y_label = str(axis_meta.get("y_label") or "y").strip()
    x_unit = str(axis_meta.get("x_unit") or "arb").strip()
    y_unit = str(axis_meta.get("y_unit") or "arb").strip()
    x_ticks_text = int(axis_meta.get("x_tick_count") or 0)
    y_ticks_text = int(axis_meta.get("y_tick_count") or 0)
    x_ticks_geom = int(axis_meta.get("x_tick_count_geom") or 0)
    y_ticks_geom = int(axis_meta.get("y_tick_count_geom") or 0)
    x_label_source = str(axis_meta.get("x_label_source") or "default")
    y_label_source = str(axis_meta.get("y_label_source") or "default")
    x_unit_source = str(axis_meta.get("x_unit_source") or "default")
    y_unit_source = str(axis_meta.get("y_unit_source") or "default")
    has_text_scale = _scale_is_valid(axis_meta.get("x_scale")) and _scale_is_valid(axis_meta.get("y_scale"))
    has_text_ticks = x_ticks_text >= 2 and y_ticks_text >= 2
    plot_dark_ratio = float(axis_meta.get("plot_dark_ratio") or 0.0)

    bbox = axis_meta.get("plot_bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x0, y0, x1, y1 = [int(v) for v in bbox]
            bw = max(1, x1 - x0)
            bh = max(1, y1 - y0)
            ratio = float(bw) / float(bh)
            # High-recall bounds: allow many plot aspect ratios, reject extreme slivers.
            if ratio < 0.12 or ratio > 8.5:
                return False
        except Exception:
            return False

    geom_max = max(x_ticks_geom, y_ticks_geom)
    has_local_labels = x_label_source == "local" or y_label_source == "local"
    has_local_units = x_unit_source.startswith("local") or y_unit_source.startswith("local")
    has_local_semantics = has_local_labels or has_local_units
    has_text_evidence = has_text_scale or has_text_ticks or has_local_semantics
    lookup_only_labels = x_label_source == "unit_lookup" and y_label_source == "unit_lookup"

    # Reject pseudo-plots inferred only from weak unit lookups with no readable text ticks/scales.
    if lookup_only_labels and x_ticks_text == 0 and y_ticks_text == 0 and not has_text_scale and plot_dark_ratio > 0.22:
        return False

    # Core high-recall acceptance: keep candidates with geometry or in-figure textual semantics.
    if not (geom_max >= 3 or has_text_evidence):
        return False

    schematic_hits = _schematic_keyword_score(context_text)
    # Mixed schematic+plot figures can leak pseudo-curves; require in-figure evidence in those cases.
    if schematic_hits >= 1 and not has_text_evidence:
        return False
    # Dense filled regions (common in micrographs/maps) need textual axis evidence.
    if plot_dark_ratio > 0.62 and not has_text_evidence:
        return False
    if plot_dark_ratio > 0.88:
        return False
    # Keep strong geometric plots even with schematic terms; reject only weak, dense candidates.
    if plot_dark_ratio > 0.78 and geom_max < 6 and not has_text_scale:
        return False
    if schematic_hits >= 3 and geom_max < 8 and not (has_text_scale or has_local_semantics):
        return False

    return True


def _extract_axis_metadata_from_figure(
    page: fitz.Page,
    image_rect: fitz.Rect | None,
    img_w: int,
    img_h: int,
    axis_pair: AxisPair,
    fallback_text: str,
    caption_text: str = "",
) -> dict[str, Any]:
    x_label, x_unit, y_label, y_unit = "x", "arb", "y", "arb"
    x_label_source, y_label_source = "default", "default"
    x_unit_source, y_unit_source = "default", "default"
    x_scale: tuple[float, float] | None = None
    y_scale: tuple[float, float] | None = None
    panel_hint: str | None = None

    plot_cx = 0.5 * (axis_pair.plot_bbox[0] + axis_pair.plot_bbox[2]) / max(1.0, float(img_w))
    if plot_cx <= 0.45:
        panel_hint = "a"
    elif plot_cx >= 0.55:
        panel_hint = "b"

    if image_rect is None:
        if fallback_text.strip():
            cx, cu, cy, cv = _infer_axis_from_context_text(fallback_text, panel_hint=panel_hint)
            if cx != "x":
                x_label = cx
                x_label_source = "context"
                if _is_probable_unit(cu):
                    x_unit = _clean_unit(cu)
                    x_unit_source = "context"
            if cy != "y":
                y_label = cy
                y_label_source = "context"
                if _is_probable_unit(cv):
                    y_unit = _clean_unit(cv)
                    y_unit_source = "context"

        if caption_text.strip() and (x_label == "x" or y_label == "y" or x_unit == "arb" or y_unit == "arb"):
            cx, cu, cy, cv = _infer_axis_from_context_text(caption_text, panel_hint=panel_hint)
            if x_label == "x" and cx != "x":
                x_label = cx
                x_label_source = "caption"
            if y_label == "y" and cy != "y":
                y_label = cy
                y_label_source = "caption"
            if x_unit == "arb" and _is_probable_unit(cu):
                x_unit = _clean_unit(cu)
                x_unit_source = "caption"
            if y_unit == "arb" and _is_probable_unit(cv):
                y_unit = _clean_unit(cv)
                y_unit_source = "caption"

        merged_text = " ".join(part for part in [fallback_text, caption_text] if part and str(part).strip())
        if merged_text and (x_label == "x" or y_label == "y" or x_unit == "arb" or y_unit == "arb"):
            cands = _collect_embedded_param_candidates(merged_text)
            used = set()
            if x_label != "x":
                used.add(_normalize_parameter_key(x_label))
            if y_label != "y":
                used.add(_normalize_parameter_key(y_label))
            pick_x = _pick_embedded_candidate(cands, axis_hint="x", used_labels=used)
            if x_label == "x" and pick_x is not None and pick_x[2] >= 5.0 and _is_plausible_axis_label(pick_x[0]):
                x_label = _clean_axis_name(pick_x[0])
                x_label_source = "embedded"
                used.add(_normalize_parameter_key(x_label))
            pick_y = _pick_embedded_candidate(cands, axis_hint="y", used_labels=used)
            if y_label == "y" and pick_y is not None and pick_y[2] >= 5.0 and _is_plausible_axis_label(pick_y[0]):
                y_label = _clean_axis_name(pick_y[0])
                y_label_source = "embedded"
                used.add(_normalize_parameter_key(y_label))
            pick_xu = _pick_embedded_unit_candidate(cands, axis_hint="x", current_label=x_label)
            if x_unit == "arb" and pick_xu is not None and pick_xu[2] >= 3.0 and _is_probable_unit(pick_xu[1]):
                x_unit = _clean_unit(pick_xu[1])
                x_unit_source = "embedded"
            pick_yu = _pick_embedded_unit_candidate(cands, axis_hint="y", current_label=y_label)
            if y_unit == "arb" and pick_yu is not None and pick_yu[2] >= 3.0 and _is_probable_unit(pick_yu[1]):
                y_unit = _clean_unit(pick_yu[1])
                y_unit_source = "embedded"

        if x_label == "x" and x_unit != "arb":
            x_guess = _lookup_standard_label_from_unit(x_unit, axis_hint="x")
            if x_guess:
                x_label = x_guess
                x_label_source = "unit_lookup"
        if y_label == "y" and y_unit != "arb":
            y_guess = _lookup_standard_label_from_unit(y_unit, axis_hint="y")
            if y_guess:
                y_label = y_guess
                y_label_source = "unit_lookup"

        if x_unit == "arb" and x_label != "x":
            x_unit = _guess_unit_from_label(x_label)
            x_unit_source = "guess"
        if y_unit == "arb" and y_label != "y":
            y_unit = _guess_unit_from_label(y_label)
            y_unit_source = "guess"

        return {
            "plot_bbox": list(axis_pair.plot_bbox),
            "plot_bbox_page": None,
            "x_label": x_label,
            "x_unit": x_unit,
            "y_label": y_label,
            "y_unit": y_unit,
            "x_scale": None,
            "y_scale": None,
            "x_tick_count": 0,
            "y_tick_count": 0,
            "x_label_source": x_label_source,
            "y_label_source": y_label_source,
            "x_unit_source": x_unit_source,
            "y_unit_source": y_unit_source,
        }

    x_axis_page_y = _px_to_page(image_rect, 0, axis_pair.x_axis_row, img_w, img_h)[1]
    y_axis_page_x = _px_to_page(image_rect, axis_pair.y_axis_col, 0, img_w, img_h)[0]
    y_top_page = _px_to_page(image_rect, 0, axis_pair.y_top, img_w, img_h)[1]
    x_right_page = _px_to_page(image_rect, axis_pair.x_right, 0, img_w, img_h)[0]
    bx0, by0, bx1, by1 = axis_pair.plot_bbox
    b0 = _px_to_page(image_rect, bx0, by0, img_w, img_h)
    b1 = _px_to_page(image_rect, bx1, by1, img_w, img_h)

    # Expand the OCR clip around the extracted figure so labels that sit farther from
    # the axis lines, especially in large or multi-panel crops, remain visible.
    pad_x = max(12.0, 0.08 * image_rect.width)
    pad_y = max(12.0, 0.12 * image_rect.height)
    clip = fitz.Rect(
        max(page.rect.x0, image_rect.x0 - pad_x),
        max(page.rect.y0, image_rect.y0 - pad_y),
        min(page.rect.x1, image_rect.x1 + pad_x),
        min(page.rect.y1, image_rect.y1 + pad_y),
    )
    words = page.get_text("words", clip=clip) or []
    embedded_words_text = _words_to_text(words, key_axis=0)

    x_label_words: list[tuple] = []
    y_label_words: list[tuple] = []
    x_tick_samples: list[tuple[float, float]] = []
    y_tick_samples: list[tuple[float, float]] = []

    for w in words:
        wx0, wy0, wx1, wy1, text = float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4]).strip()
        if not text:
            continue
        cx = 0.5 * (wx0 + wx1)
        cy = 0.5 * (wy0 + wy1)
        num = _parse_numeric_token(text)

        # x-axis tick labels: near and below horizontal axis line.
        if y_axis_page_x - 0.02 * image_rect.width <= cx <= x_right_page + 0.02 * image_rect.width:
            if x_axis_page_y - 0.06 * image_rect.height <= cy <= x_axis_page_y + 0.12 * image_rect.height:
                if num is not None:
                    px, _ = _page_to_px(image_rect, cx, cy, img_w, img_h)
                    x_tick_samples.append((px, float(num)))
                elif x_axis_page_y + 0.02 * image_rect.height <= cy <= x_axis_page_y + 0.24 * image_rect.height:
                    x_label_words.append(w)

        # y-axis tick labels: near and left of vertical axis line.
        if y_top_page - 0.03 * image_rect.height <= cy <= x_axis_page_y + 0.02 * image_rect.height:
            if y_axis_page_x - 0.20 * image_rect.width <= cx <= y_axis_page_x + 0.05 * image_rect.width:
                if num is not None:
                    _, py = _page_to_px(image_rect, cx, cy, img_w, img_h)
                    y_tick_samples.append((py, float(num)))
                elif cx < y_axis_page_x - 0.015 * image_rect.width:
                    y_label_words.append(w)

    x_fit = _fit_linear_scale(x_tick_samples)
    y_fit = _fit_linear_scale(y_tick_samples)
    if x_fit is not None:
        x_scale = x_fit
    if y_fit is not None:
        y_scale = y_fit

    x_label_guess = _extract_parallel_axis_text(
        words,
        orientation="horizontal",
        axis_coord=x_axis_page_y,
        axis_min=y_axis_page_x,
        axis_max=x_right_page,
        image_span_x=image_rect.width,
        image_span_y=image_rect.height,
    )
    y_label_guess = _extract_parallel_axis_text(
        words,
        orientation="vertical",
        axis_coord=y_axis_page_x,
        axis_min=y_top_page,
        axis_max=x_axis_page_y,
        image_span_x=image_rect.width,
        image_span_y=image_rect.height,
    )
    if not x_label_guess:
        x_label_guess = _words_to_text(x_label_words, key_axis=0)
    if not y_label_guess:
        y_label_guess_h = _words_to_text(y_label_words, key_axis=1)
        y_label_guess_v = _extract_vertical_label(y_label_words, x_tol=max(6.0, 0.03 * image_rect.width))
        y_label_guess = y_label_guess_v if len(y_label_guess_v) >= 3 else y_label_guess_h

    x_phrase = _best_axis_phrase_candidate(x_label_words, "x", image_rect.width, vertical_ok=False)
    if x_phrase is not None:
        x_lab, x_uni, x_unit_src = x_phrase
        if x_lab != "x":
            x_label = x_lab
            x_label_source = "local"
        if x_uni != "arb":
            x_unit = x_uni
            x_unit_source = f"local_{x_unit_src}"

    if y_label == "y" or y_unit == "arb":
        y_phrase = _best_axis_phrase_candidate(y_label_words, "y", image_rect.width, vertical_ok=True)
        if y_phrase is not None:
            y_lab, y_uni, y_unit_src = y_phrase
            if y_lab != "y":
                y_label = y_lab
                y_label_source = "local"
            if y_uni != "arb":
                y_unit = y_uni
                y_unit_source = f"local_{y_unit_src}"

    if x_label_guess and x_label == "x" and x_unit == "arb":
        x_lab, x_uni, x_unit_src = _label_unit_from_phrase(x_label_guess, "x", "arb")
        if x_lab != "x":
            x_label = x_lab
            x_label_source = "local"
        if x_uni != "arb":
            x_unit = x_uni
            x_unit_source = f"local_{x_unit_src}"

    if y_label_guess and y_label == "y" and y_unit == "arb":
        y_lab, y_uni, y_unit_src = _label_unit_from_phrase(y_label_guess, "y", "arb")
        if y_lab != "y":
            y_label = y_lab
            y_label_source = "local"
        if y_uni != "arb":
            y_unit = y_uni
            y_unit_source = f"local_{y_unit_src}"

    # Conservative local/page-context fallback to reduce x/y/arb when local text is unavailable.
    if x_label == "x" or y_label == "y":
        cx, cu, cy, cv = _infer_axis_from_context_text(fallback_text, panel_hint=panel_hint)
        if x_label == "x" and cx != "x":
            x_label = cx
            x_label_source = "context"
            if _is_probable_unit(cu):
                x_unit = cu
                x_unit_source = "context"
            else:
                x_unit = _guess_unit_from_label(cx)
                x_unit_source = "guess"
        if y_label == "y" and cy != "y":
            y_label = cy
            y_label_source = "context"
            if _is_probable_unit(cv):
                y_unit = cv
                y_unit_source = "context"
            else:
                y_unit = _guess_unit_from_label(cy)
                y_unit_source = "guess"

    # Caption fallback: when labels/units are absent in the panel itself, pull from figure caption text.
    # This is intentionally applied after local extraction, so explicit in-figure text remains authoritative.
    if caption_text.strip() and (x_label == "x" or y_label == "y" or x_unit == "arb" or y_unit == "arb"):
        cx, cu, cy, cv = _infer_axis_from_context_text(caption_text, panel_hint=panel_hint)
        if x_label == "x" and cx != "x":
            x_label = cx
            x_label_source = "caption"
        if y_label == "y" and cy != "y":
            y_label = cy
            y_label_source = "caption"
        if x_unit == "arb":
            if _is_probable_unit(cu):
                x_unit = cu
                x_unit_source = "caption"
            elif x_label != "x":
                x_unit = _guess_unit_from_label(x_label)
                x_unit_source = "guess"
        if y_unit == "arb":
            if _is_probable_unit(cv):
                y_unit = cv
                y_unit_source = "caption"
            elif y_label != "y":
                y_unit = _guess_unit_from_label(y_label)
                y_unit_source = "guess"

    # Caption unit-only fallback for embedded scales/annotations (e.g., "3 x 3 um^2", "bias voltage ... V").
    if caption_text.strip():
        if x_label == "x" and x_unit == "arb":
            u_len = _extract_length_unit_from_text(caption_text)
            if u_len:
                x_label = "length"
                x_unit = u_len
                x_label_source = "caption"
                x_unit_source = "caption"
        if y_label == "y" and y_unit == "arb":
            low_cap = caption_text.lower()
            if "voltage" in low_cap or re.search(
                r"(?<![A-Za-z0-9])[+-]?\d+(?:\.\d+)?\s*v(?![A-Za-z])", caption_text, flags=re.IGNORECASE
            ):
                y_label = "voltage"
                y_unit = "V"
                y_label_source = "caption"
                y_unit_source = "caption"

    # Generic embedded parameter/unit extraction from in-figure OCR text.
    weak_sources = {"default", "context", "caption", "guess", "unit_lookup"}
    has_embedded_words = bool((embedded_words_text or "").strip())
    if has_embedded_words or x_label == "x" or y_label == "y" or x_unit == "arb" or y_unit == "arb":
        in_figure_cands = _collect_embedded_param_candidates(embedded_words_text)
        all_text = " ".join(part for part in [embedded_words_text, fallback_text, caption_text] if part and str(part).strip())
        all_cands = _collect_embedded_param_candidates(all_text)
        cands: list[tuple[str, str, float]] = []
        cands.extend((lbl, unt, sc + 1.2) for lbl, unt, sc in in_figure_cands)
        cands.extend(all_cands)

        best_lu: dict[tuple[str, str], tuple[str, str, float]] = {}
        for lbl, unt, sc in cands:
            key = (_normalize_parameter_key(lbl), _normalize_unit_key(unt))
            if not key[0] and not key[1]:
                continue
            cur = best_lu.get(key)
            if cur is None or sc > cur[2]:
                best_lu[key] = (lbl, unt, float(sc))
        cands = list(best_lu.values())

        used = set()
        if x_label != "x":
            used.add(_normalize_parameter_key(x_label))
        if y_label != "y":
            used.add(_normalize_parameter_key(y_label))

        # Fill/override weak labels from embedded explicit pairs.
        pick_x = _pick_embedded_candidate(cands, axis_hint="x", used_labels=used)
        if pick_x is not None and _is_plausible_axis_label(pick_x[0]):
            can_set = x_label == "x" or x_label_source in weak_sources or (
                x_label_source == "local" and not _is_axis_semantic_label(x_label)
            )
            if can_set and pick_x[2] >= 5.0:
                x_label = _clean_axis_name(pick_x[0])
                x_label_source = "embedded"
                used.add(_normalize_parameter_key(x_label))

        pick_y = _pick_embedded_candidate(cands, axis_hint="y", used_labels=used)
        if pick_y is not None and _is_plausible_axis_label(pick_y[0]):
            can_set = y_label == "y" or y_label_source in weak_sources or (
                y_label_source == "local" and not _is_axis_semantic_label(y_label)
            )
            if can_set and pick_y[2] >= 5.0:
                y_label = _clean_axis_name(pick_y[0])
                y_label_source = "embedded"
                used.add(_normalize_parameter_key(y_label))

        # Fill/override weak units using embedded candidates, even when only unit cues are present.
        pick_xu = _pick_embedded_unit_candidate(cands, axis_hint="x", current_label=x_label)
        if pick_xu is not None:
            can_set = x_unit == "arb" or x_unit_source in weak_sources
            min_score = 3.0 if x_unit == "arb" else 8.2
            if can_set and pick_xu[2] >= min_score and _is_probable_unit(pick_xu[1]):
                x_unit = _clean_unit(pick_xu[1])
                x_unit_source = "embedded"

        pick_yu = _pick_embedded_unit_candidate(cands, axis_hint="y", current_label=y_label)
        if pick_yu is not None:
            can_set = y_unit == "arb" or y_unit_source in weak_sources
            min_score = 3.0 if y_unit == "arb" else 8.2
            if can_set and pick_yu[2] >= min_score and _is_probable_unit(pick_yu[1]):
                y_unit = _clean_unit(pick_yu[1])
                y_unit_source = "embedded"

    # Unit->label fallback: if label is still unknown but unit is known, infer label from standard mappings.
    if x_label == "x" and x_unit != "arb":
        x_guess = _lookup_standard_label_from_unit(x_unit, axis_hint="x")
        if x_guess:
            x_label = x_guess
            x_label_source = "unit_lookup"
    if y_label == "y" and y_unit != "arb":
        y_guess = _lookup_standard_label_from_unit(y_unit, axis_hint="y")
        if y_guess:
            y_label = y_guess
            y_label_source = "unit_lookup"

    # Final sanity guard: reject non-semantic labels leaked from local OCR fragments.
    if x_label not in {"x", "y"} and not _is_axis_semantic_label(x_label):
        x_guess = _lookup_standard_label_from_unit(x_unit, axis_hint="x") if x_unit != "arb" else None
        if x_guess:
            x_label = x_guess
            x_label_source = "unit_lookup"
        else:
            x_label = "x"
            x_label_source = "default"
    if y_label not in {"x", "y"} and not _is_axis_semantic_label(y_label):
        y_guess = _lookup_standard_label_from_unit(y_unit, axis_hint="y") if y_unit != "arb" else None
        if y_guess:
            y_label = y_guess
            y_label_source = "unit_lookup"
        else:
            y_label = "y"
            y_label_source = "default"

    if x_unit == "arb" and x_label != "x":
        x_unit = _guess_unit_from_label(x_label)
        x_unit_source = "guess"
    if y_unit == "arb" and y_label != "y":
        y_unit = _guess_unit_from_label(y_label)
        y_unit_source = "guess"

    return {
        "plot_bbox": list(axis_pair.plot_bbox),
        "plot_bbox_page": [float(min(b0[0], b1[0])), float(min(b0[1], b1[1])), float(max(b0[0], b1[0])), float(max(b0[1], b1[1]))],
        "x_label": x_label,
        "x_unit": x_unit,
        "y_label": y_label,
        "y_unit": y_unit,
        "x_scale": list(x_scale) if x_scale is not None else None,
        "y_scale": list(y_scale) if y_scale is not None else None,
        "x_tick_count": len(x_tick_samples),
        "y_tick_count": len(y_tick_samples),
        "x_label_source": x_label_source,
        "y_label_source": y_label_source,
        "x_unit_source": x_unit_source,
        "y_unit_source": y_unit_source,
    }


def _enrich_axis_meta_from_context(axis_meta: dict[str, Any], context_text: str, source_tag: str = "context") -> dict[str, Any]:
    text = (context_text or "").strip()
    if not text:
        return axis_meta

    x_label = str(axis_meta.get("x_label") or "x")
    x_unit = str(axis_meta.get("x_unit") or "arb")
    y_label = str(axis_meta.get("y_label") or "y")
    y_unit = str(axis_meta.get("y_unit") or "arb")
    x_label_source = str(axis_meta.get("x_label_source") or "default")
    y_label_source = str(axis_meta.get("y_label_source") or "default")
    x_unit_source = str(axis_meta.get("x_unit_source") or "default")
    y_unit_source = str(axis_meta.get("y_unit_source") or "default")

    cx, cu, cy, cv = _infer_axis_from_context_text(text)
    if x_label == "x" and cx != "x":
        x_label = cx
        x_label_source = source_tag
    if y_label == "y" and cy != "y":
        y_label = cy
        y_label_source = source_tag
    if x_unit == "arb" and _is_probable_unit(cu):
        x_unit = _clean_unit(cu)
        x_unit_source = source_tag
    if y_unit == "arb" and _is_probable_unit(cv):
        y_unit = _clean_unit(cv)
        y_unit_source = source_tag

    if x_label == "x" or y_label == "y" or x_unit == "arb" or y_unit == "arb":
        cands = _collect_embedded_param_candidates(text)
        used = set()
        if x_label != "x":
            used.add(_normalize_parameter_key(x_label))
        if y_label != "y":
            used.add(_normalize_parameter_key(y_label))
        if x_label == "x":
            px = _pick_embedded_candidate(cands, axis_hint="x", used_labels=used)
            if px is not None and px[2] >= 5.0 and _is_plausible_axis_label(px[0]):
                x_label = _clean_axis_name(px[0])
                x_label_source = "embedded_context"
                used.add(_normalize_parameter_key(x_label))
        if y_label == "y":
            py = _pick_embedded_candidate(cands, axis_hint="y", used_labels=used)
            if py is not None and py[2] >= 5.0 and _is_plausible_axis_label(py[0]):
                y_label = _clean_axis_name(py[0])
                y_label_source = "embedded_context"
                used.add(_normalize_parameter_key(y_label))
        if x_unit == "arb":
            pux = _pick_embedded_unit_candidate(cands, axis_hint="x", current_label=x_label)
            if pux is not None and pux[2] >= 3.0 and _is_probable_unit(pux[1]):
                x_unit = _clean_unit(pux[1])
                x_unit_source = "embedded_context"
        if y_unit == "arb":
            puy = _pick_embedded_unit_candidate(cands, axis_hint="y", current_label=y_label)
            if puy is not None and puy[2] >= 3.0 and _is_probable_unit(puy[1]):
                y_unit = _clean_unit(puy[1])
                y_unit_source = "embedded_context"

    if x_label == "x" and x_unit != "arb":
        x_guess = _lookup_standard_label_from_unit(x_unit, axis_hint="x")
        if x_guess:
            x_label = x_guess
            x_label_source = "unit_lookup"
    if y_label == "y" and y_unit != "arb":
        y_guess = _lookup_standard_label_from_unit(y_unit, axis_hint="y")
        if y_guess:
            y_label = y_guess
            y_label_source = "unit_lookup"

    if x_unit == "arb" and x_label != "x":
        x_unit = _guess_unit_from_label(x_label)
        x_unit_source = "guess"
    if y_unit == "arb" and y_label != "y":
        y_unit = _guess_unit_from_label(y_label)
        y_unit_source = "guess"

    axis_meta["x_label"] = x_label
    axis_meta["x_unit"] = x_unit
    axis_meta["y_label"] = y_label
    axis_meta["y_unit"] = y_unit
    axis_meta["x_label_source"] = x_label_source
    axis_meta["y_label_source"] = y_label_source
    axis_meta["x_unit_source"] = x_unit_source
    axis_meta["y_unit_source"] = y_unit_source
    return axis_meta


def _estimate_plot_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    h, w = gray.shape
    dark = gray < 170

    x_hist = dark.sum(axis=0)
    y_hist = dark.sum(axis=1)

    x_search_end = max(10, int(w * 0.45))
    y_search_start = max(0, int(h * 0.50))

    x_axis = int(np.argmax(x_hist[:x_search_end])) if x_search_end > 1 else 0
    y_axis = int(np.argmax(y_hist[y_search_start:]) + y_search_start) if y_search_start < h - 1 else h - 1

    ys, xs = np.where(dark)
    if xs.size == 0 or ys.size == 0:
        return int(0.08 * w), int(0.08 * h), int(0.95 * w), int(0.90 * h)

    x_right = int(np.percentile(xs, 99))
    y_top = int(np.percentile(ys, 1))

    x0 = max(0, min(w - 2, x_axis + 2))
    y1 = max(1, min(h - 1, y_axis - 1))
    x1 = max(x0 + 25, min(w - 1, x_right))
    y0 = min(y1 - 25, max(0, y_top))

    if x1 - x0 < 25 or y1 - y0 < 25:
        return int(0.08 * w), int(0.08 * h), int(0.95 * w), int(0.90 * h)

    return x0, y0, x1, y1


def _expanded_subplot_crop_bounds(
    plot_bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    left_frac: float = 0.24,
    right_frac: float = 0.10,
    top_frac: float = 0.10,
    bottom_frac: float = 0.28,
    min_pad_px: int = 10,
    full_width: bool = False,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(v) for v in plot_bbox]
    x0 = max(0, min(img_w - 1, x0))
    y0 = max(0, min(img_h - 1, y0))
    x1 = max(x0 + 1, min(img_w, x1))
    y1 = max(y0 + 1, min(img_h, y1))

    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    pad_l = max(min_pad_px, int(left_frac * bw))
    pad_r = max(min_pad_px, int(right_frac * bw))
    pad_t = max(min_pad_px, int(top_frac * bh))
    pad_b = max(min_pad_px, int(bottom_frac * bh))

    cx0 = 0 if full_width else max(0, x0 - pad_l)
    cy0 = max(0, y0 - pad_t)
    cx1 = img_w if full_width else min(img_w, x1 + pad_r)
    cy1 = min(img_h, y1 + pad_b)

    return int(cx0), int(cy0), int(cx1), int(cy1)


def _boxed_subplot_crop_bounds(
    gray: np.ndarray,
    axis_pair: AxisPair,
    *,
    left_frac: float = 0.24,
    right_frac: float = 0.10,
    top_frac: float = 0.10,
    bottom_frac: float = 0.28,
    min_pad_px: int = 10,
) -> tuple[int, int, int, int]:
    img_h, img_w = gray.shape
    x0, y0, x1, y1 = [int(v) for v in axis_pair.plot_bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    dark = gray < 165

    def _best_row(start: int, end: int, sx0: int, sx1: int, target: int) -> int | None:
        best_idx = None
        best_score = -1e9
        sx0 = max(0, sx0)
        sx1 = min(img_w, sx1)
        if sx1 - sx0 < 12:
            return None
        for r in range(max(0, start), min(img_h, end)):
            seg = dark[r, sx0:sx1]
            if seg.size < 12:
                continue
            dens = float(seg.mean())
            run = _longest_true_run(seg) / max(1.0, float(seg.size))
            score = 1.4 * dens + 1.8 * run - 0.0015 * abs(r - target)
            if run >= 0.55 and dens >= 0.28 and score > best_score:
                best_score = score
                best_idx = r
        return best_idx

    def _best_col(start: int, end: int, sy0: int, sy1: int, target: int) -> int | None:
        best_idx = None
        best_score = -1e9
        sy0 = max(0, sy0)
        sy1 = min(img_h, sy1)
        if sy1 - sy0 < 12:
            return None
        for c in range(max(0, start), min(img_w, end)):
            seg = dark[sy0:sy1, c]
            if seg.size < 12:
                continue
            dens = float(seg.mean())
            run = _longest_true_run(seg) / max(1.0, float(seg.size))
            score = 1.4 * dens + 1.8 * run - 0.0015 * abs(c - target)
            if run >= 0.55 and dens >= 0.28 and score > best_score:
                best_score = score
                best_idx = c
        return best_idx

    left_line = _best_col(
        int(axis_pair.y_axis_col - 0.10 * bw),
        int(axis_pair.y_axis_col + 0.06 * bw) + 1,
        int(y0 - 0.04 * bh),
        int(y1 + 0.06 * bh),
        axis_pair.y_axis_col,
    )
    right_line = _best_col(
        int(x1 - 0.05 * bw),
        int(x1 + 0.20 * bw) + 1,
        int(y0 - 0.04 * bh),
        int(y1 + 0.06 * bh),
        x1,
    )
    top_line = _best_row(
        int(y0 - 0.18 * bh),
        int(y0 + 0.10 * bh) + 1,
        int((left_line if left_line is not None else axis_pair.y_axis_col) - 0.02 * bw),
        int((right_line if right_line is not None else x1) + 0.02 * bw),
        y0,
    )
    bottom_line = _best_row(
        int(axis_pair.x_axis_row - 0.06 * bh),
        int(axis_pair.x_axis_row + 0.12 * bh) + 1,
        int((left_line if left_line is not None else axis_pair.y_axis_col) - 0.02 * bw),
        int((right_line if right_line is not None else x1) + 0.02 * bw),
        axis_pair.x_axis_row,
    )

    panel_x0 = left_line if left_line is not None else axis_pair.y_axis_col
    panel_x1 = right_line if right_line is not None else x1
    panel_y0 = top_line if top_line is not None else y0
    panel_y1 = bottom_line if bottom_line is not None else axis_pair.x_axis_row

    if panel_x1 <= panel_x0 or panel_y1 <= panel_y0:
        return _expanded_subplot_crop_bounds(
            axis_pair.plot_bbox,
            img_w=img_w,
            img_h=img_h,
            left_frac=left_frac,
            right_frac=right_frac,
            top_frac=top_frac,
            bottom_frac=bottom_frac,
            min_pad_px=min_pad_px,
            full_width=False,
        )

    return _expanded_subplot_crop_bounds(
        (int(panel_x0), int(panel_y0), int(panel_x1), int(panel_y1)),
        img_w=img_w,
        img_h=img_h,
        left_frac=left_frac,
        right_frac=right_frac,
        top_frac=top_frac,
        bottom_frac=bottom_frac,
        min_pad_px=min_pad_px,
        full_width=False,
    )


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 260) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    idx = np.linspace(0, x.size - 1, max_points).astype(int)
    return x[idx], y[idx]


def _trace_curve_from_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[list[float], list[float]] | None:
    x0, y0, x1, y1 = bbox
    h, w = mask.shape
    if h < 6 or w < 6:
        return None

    xs: list[int] = []
    ys: list[float] = []
    col_thickness: list[int] = []
    for cx in range(w):
        y_idx = np.where(mask[:, cx])[0]
        if y_idx.size == 0:
            continue
        ys_med = float(np.median(y_idx))
        xs.append(cx + x0)
        ys.append(ys_med + y0)
        col_thickness.append(int(y_idx.size))

    if len(xs) < 18:
        return None

    coverage = len(xs) / max(1.0, w)
    if coverage < 0.22:
        return None

    thick = np.asarray(col_thickness, dtype=float)
    thick_med = float(np.median(thick))
    thick_p90 = float(np.percentile(thick, 90))
    thick_wide_ratio = float(np.mean(thick > max(8.0, 0.14 * h)))

    # Reject broad filled regions that are common in non-plot diagrams/photos.
    if thick_med > max(5.0, 0.08 * h):
        return None
    if thick_p90 > max(10.0, 0.18 * h):
        return None
    if thick_wide_ratio > 0.25:
        return None

    x_pix = np.asarray(xs, dtype=float)
    y_pix = np.asarray(ys, dtype=float)

    y_span_pix = float(np.max(y_pix) - np.min(y_pix))
    if y_span_pix < max(3.0, 0.03 * h):
        return None

    dy = np.abs(np.diff(y_pix))
    if dy.size > 0 and float(np.percentile(dy, 95)) > max(14.0, 0.28 * h):
        return None

    x_data = (x_pix - x0) / max(1.0, (x1 - x0 - 1))
    y_data = (y1 - 1 - y_pix) / max(1.0, (y1 - y0 - 1))

    y_data = np.clip(y_data, -0.5, 1.5)

    if y_data.size >= 7:
        kernel = np.ones(5, dtype=float) / 5.0
        y_data = np.convolve(y_data, kernel, mode="same")

    x_data, y_data = _downsample_xy(x_data, y_data)
    return _float_list(x_data.tolist()), _float_list(y_data.tolist())


def _trace_curve_from_mask_relaxed(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[list[float], list[float]] | None:
    x0, y0, x1, y1 = bbox
    h, w = mask.shape
    if h < 6 or w < 6:
        return None

    xs: list[int] = []
    ys: list[float] = []
    for cx in range(w):
        y_idx = np.where(mask[:, cx])[0]
        if y_idx.size == 0:
            continue
        xs.append(cx + x0)
        ys.append(float(np.median(y_idx)) + y0)

    if len(xs) < 14:
        return None

    x_pix = np.asarray(xs, dtype=float)
    y_pix = np.asarray(ys, dtype=float)
    x_data = (x_pix - x0) / max(1.0, (x1 - x0 - 1))
    y_data = (y1 - 1 - y_pix) / max(1.0, (y1 - y0 - 1))
    y_data = np.clip(y_data, -0.5, 1.5)
    x_data, y_data = _downsample_xy(x_data, y_data)
    return _float_list(x_data.tolist()), _float_list(y_data.tolist())


def _extract_color_series(rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> list[tuple[list[float], list[float], tuple[int, int, int]]]:
    x0, y0, x1, y1 = bbox
    plot = rgb[y0:y1, x0:x1]
    if plot.size == 0:
        return []

    flat = plot.reshape(-1, 3).astype(np.int16)
    brightness = flat.mean(axis=1)
    not_background = brightness < 245
    not_black = ~((flat[:, 0] < 55) & (flat[:, 1] < 55) & (flat[:, 2] < 55))
    candidate = not_background & not_black

    if int(candidate.sum()) < 180:
        return []

    colors = (flat[candidate] // 32) * 32
    uniq, counts = np.unique(colors, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]

    min_count = max(150, int(plot.shape[0] * plot.shape[1] * 0.003))
    selected_colors: list[np.ndarray] = []
    for idx in order:
        if counts[idx] < min_count:
            continue
        c = uniq[idx]
        if int(c[0]) > 235 and int(c[1]) > 235 and int(c[2]) > 235:
            continue
        selected_colors.append(c)
        if len(selected_colors) >= 5:
            break

    out: list[tuple[list[float], list[float], tuple[int, int, int]]] = []
    for c in selected_colors:
        dist = np.linalg.norm(plot.astype(np.int16) - c.reshape(1, 1, 3), axis=2)
        mask = dist <= 30
        if float(mask.mean()) > 0.35:
            continue
        traced = _trace_curve_from_mask(mask, bbox)
        if traced is None:
            continue
        x, y = traced
        if len(x) < 18:
            continue
        out.append((x, y, (int(c[0]), int(c[1]), int(c[2]))))

    return out


def _extract_dark_fallback(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[list[float], list[float]] | None:
    x0, y0, x1, y1 = bbox
    plot = gray[y0:y1, x0:x1]
    if plot.size == 0:
        return None

    threshold = float(min(150, np.percentile(plot, 20)))
    mask = plot <= threshold

    # Remove edges where axes tend to dominate.
    if mask.shape[1] > 4:
        mask[:, :3] = False
    if mask.shape[0] > 4:
        mask[-3:, :] = False

    traced = _trace_curve_from_mask(mask, bbox)
    if traced is not None:
        return traced
    return _trace_curve_from_mask_relaxed(mask, bbox)


def _apply_axis_scale(
    x_values: list[float], y_values: list[float], bbox: tuple[int, int, int, int], axis_meta: dict[str, Any] | None
) -> tuple[list[float], list[float]]:
    if not axis_meta:
        return x_values, y_values

    x0, y0, x1, y1 = bbox
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    x_out = x.copy()
    y_out = y.copy()

    x_scale = axis_meta.get("x_scale")
    y_scale = axis_meta.get("y_scale")

    if isinstance(x_scale, (list, tuple)) and len(x_scale) == 2:
        a, b = float(x_scale[0]), float(x_scale[1])
        x_pix = x0 + x * max(1.0, (x1 - x0 - 1))
        x_out = a * x_pix + b

    if isinstance(y_scale, (list, tuple)) and len(y_scale) == 2:
        a, b = float(y_scale[0]), float(y_scale[1])
        y_pix = (y1 - 1) - y * max(1.0, (y1 - y0 - 1))
        y_out = a * y_pix + b

    return _float_list(x_out.tolist()), _float_list(y_out.tolist())


def digitize_figure_image(
    image_path: Path,
    page_text: str,
    pdf_file: str,
    page_index: int,
    image_index: int,
    axis_meta: dict[str, Any] | None = None,
) -> FigureData | None:
    try:
        rgb = np.array(Image.open(image_path).convert("RGB"))
    except Exception:
        return None

    gray = np.array(Image.fromarray(rgb).convert("L"))

    bbox: tuple[int, int, int, int]
    meta_bbox = (axis_meta or {}).get("plot_bbox")
    if (
        isinstance(meta_bbox, (list, tuple))
        and len(meta_bbox) == 4
        and all(isinstance(v, (int, float)) for v in meta_bbox)
    ):
        bx0, by0, bx1, by1 = [int(v) for v in meta_bbox]
        if bx1 - bx0 >= 30 and by1 - by0 >= 30:
            bbox = (bx0, by0, bx1, by1)
        else:
            bbox = _estimate_plot_bbox(gray)
    else:
        bbox = _estimate_plot_bbox(gray)

    color_series = _extract_color_series(rgb, bbox)

    series: list[SeriesData] = []
    if color_series:
        for idx, (x, y, color) in enumerate(color_series[:4]):
            x_scaled, y_scaled = _apply_axis_scale(x, y, bbox, axis_meta)
            series.append(
                SeriesData(
                    series_id=f"series_{idx:02d}",
                    series_label=f"series_{idx:02d}_rgb{color[0]}_{color[1]}_{color[2]}",
                    x=x_scaled,
                    y=y_scaled,
                )
            )
    else:
        fallback = _extract_dark_fallback(gray, bbox)
        if fallback is None:
            return None
        x, y = fallback
        x_scaled, y_scaled = _apply_axis_scale(x, y, bbox, axis_meta)
        series.append(SeriesData(series_id="series_00", series_label="series_00_dark", x=x_scaled, y=y_scaled))

    if not series:
        return None

    descriptor = _extract_page_caption(page_text)
    if axis_meta:
        x_label = str(axis_meta.get("x_label") or "x")
        x_unit = str(axis_meta.get("x_unit") or "arb")
        y_label = str(axis_meta.get("y_label") or "y")
        y_unit = str(axis_meta.get("y_unit") or "arb")
    else:
        x_label, x_unit, y_label, y_unit = infer_axis_labels_units(page_text)

    if x_unit == "arb" and x_label != "x":
        x_unit = _guess_unit_from_label(x_label)
    if y_unit == "arb" and y_label != "y":
        y_unit = _guess_unit_from_label(y_label)
    if x_label == "x" and x_unit != "arb":
        x_guess = _lookup_standard_label_from_unit(x_unit, axis_hint="x")
        if x_guess:
            x_label = x_guess
    if y_label == "y" and y_unit != "arb":
        y_guess = _lookup_standard_label_from_unit(y_unit, axis_hint="y")
        if y_guess:
            y_label = y_guess

    # Embedded page-text fallback when metadata still has unknown labels/units.
    if x_label == "x" or y_label == "y" or x_unit == "arb" or y_unit == "arb":
        cands = _collect_embedded_param_candidates(page_text or "")
        used = set()
        if x_label != "x":
            used.add(_normalize_parameter_key(x_label))
        if y_label != "y":
            used.add(_normalize_parameter_key(y_label))

        if x_label == "x":
            pick_x = _pick_embedded_candidate(cands, axis_hint="x", used_labels=used)
            if pick_x is not None and pick_x[2] >= 5.0 and _is_plausible_axis_label(pick_x[0]):
                x_label = _clean_axis_name(pick_x[0])
                used.add(_normalize_parameter_key(x_label))
        if y_label == "y":
            pick_y = _pick_embedded_candidate(cands, axis_hint="y", used_labels=used)
            if pick_y is not None and pick_y[2] >= 5.0 and _is_plausible_axis_label(pick_y[0]):
                y_label = _clean_axis_name(pick_y[0])
                used.add(_normalize_parameter_key(y_label))

        if x_unit == "arb":
            pick_xu = _pick_embedded_unit_candidate(cands, axis_hint="x", current_label=x_label)
            if pick_xu is not None and pick_xu[2] >= 3.0 and _is_probable_unit(pick_xu[1]):
                x_unit = _clean_unit(pick_xu[1])
        if y_unit == "arb":
            pick_yu = _pick_embedded_unit_candidate(cands, axis_hint="y", current_label=y_label)
            if pick_yu is not None and pick_yu[2] >= 3.0 and _is_probable_unit(pick_yu[1]):
                y_unit = _clean_unit(pick_yu[1])

        if x_label == "x" and x_unit != "arb":
            x_guess = _lookup_standard_label_from_unit(x_unit, axis_hint="x")
            if x_guess:
                x_label = x_guess
        if y_label == "y" and y_unit != "arb":
            y_guess = _lookup_standard_label_from_unit(y_unit, axis_hint="y")
            if y_guess:
                y_label = y_guess

    figure_id = _slug(f"{Path(pdf_file).stem}_p{page_index:03d}_i{image_index:03d}")

    return FigureData(
        figure_id=figure_id,
        pdf_file=pdf_file,
        page_index=page_index,
        image_index=image_index,
        image_file=str(image_path),
        descriptor=descriptor,
        x_label=x_label,
        x_unit=x_unit,
        y_label=y_label,
        y_unit=y_unit,
        series=series,
    )


def _series_scalars(series: SeriesData) -> dict[str, float]:
    x = np.asarray(series.x, dtype=float)
    y = np.asarray(series.y, dtype=float)
    if x.size < 2:
        return {
            "n_points": float(x.size),
            "x_min": float(np.min(x)) if x.size else 0.0,
            "x_max": float(np.max(x)) if x.size else 0.0,
            "y_min": float(np.min(y)) if y.size else 0.0,
            "y_max": float(np.max(y)) if y.size else 0.0,
            "y_mean": float(np.mean(y)) if y.size else 0.0,
            "y_std": float(np.std(y)) if y.size else 0.0,
            "slope": 0.0,
        }

    slope = float(np.polyfit(x, y, deg=1)[0]) if len(np.unique(x)) >= 2 else 0.0
    return {
        "n_points": float(x.size),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "slope": slope,
    }


def _figure_scalars(fig: FigureData) -> dict[str, float]:
    all_x = np.concatenate([np.asarray(s.x, dtype=float) for s in fig.series])
    all_y = np.concatenate([np.asarray(s.y, dtype=float) for s in fig.series])
    return {
        "n_series": float(len(fig.series)),
        "n_points_total": float(all_x.size),
        "x_min": float(np.min(all_x)),
        "x_max": float(np.max(all_x)),
        "y_min": float(np.min(all_y)),
        "y_max": float(np.max(all_y)),
        "y_mean": float(np.mean(all_y)),
        "y_std": float(np.std(all_y)),
    }


def export_ground_truth_csv(figures: list[FigureData], csv_dir: Path) -> tuple[Path, Path]:
    csv_dir.mkdir(parents=True, exist_ok=True)
    points_csv = csv_dir / "ground_truth_xy_points.csv"
    figures_csv = csv_dir / "ground_truth_figures.csv"

    with points_csv.open("w", encoding="utf-8", newline="") as f_points:
        writer = csv.DictWriter(
            f_points,
            fieldnames=[
                "figure_id",
                "pdf_file",
                "page_index",
                "image_index",
                "image_file",
                "series_id",
                "series_label",
                "x",
                "y",
                "x_label",
                "x_unit",
                "y_label",
                "y_unit",
                "descriptor",
                "series_scalars_json",
            ],
        )
        writer.writeheader()

        for fig in figures:
            for s in fig.series:
                s_scalars = _series_scalars(s)
                s_scalars_json = json.dumps(s_scalars, sort_keys=True)
                for x_val, y_val in zip(s.x, s.y):
                    writer.writerow(
                        {
                            "figure_id": fig.figure_id,
                            "pdf_file": fig.pdf_file,
                            "page_index": fig.page_index,
                            "image_index": fig.image_index,
                            "image_file": fig.image_file,
                            "series_id": s.series_id,
                            "series_label": s.series_label,
                            "x": float(x_val),
                            "y": float(y_val),
                            "x_label": fig.x_label,
                            "x_unit": fig.x_unit,
                            "y_label": fig.y_label,
                            "y_unit": fig.y_unit,
                            "descriptor": fig.descriptor,
                            "series_scalars_json": s_scalars_json,
                        }
                    )

    with figures_csv.open("w", encoding="utf-8", newline="") as f_figures:
        writer = csv.DictWriter(
            f_figures,
            fieldnames=[
                "figure_id",
                "pdf_file",
                "page_index",
                "image_index",
                "image_file",
                "descriptor",
                "x_label",
                "x_unit",
                "y_label",
                "y_unit",
                "n_series",
                "n_points_total",
                "figure_scalars_json",
            ],
        )
        writer.writeheader()

        for fig in figures:
            f_scalars = _figure_scalars(fig)
            writer.writerow(
                {
                    "figure_id": fig.figure_id,
                    "pdf_file": fig.pdf_file,
                    "page_index": fig.page_index,
                    "image_index": fig.image_index,
                    "image_file": fig.image_file,
                    "descriptor": fig.descriptor,
                    "x_label": fig.x_label,
                    "x_unit": fig.x_unit,
                    "y_label": fig.y_label,
                    "y_unit": fig.y_unit,
                    "n_series": int(f_scalars["n_series"]),
                    "n_points_total": int(f_scalars["n_points_total"]),
                    "figure_scalars_json": json.dumps(f_scalars, sort_keys=True),
                }
            )

    return points_csv, figures_csv


def _wrong_unit(unit: str) -> str:
    mapping = {
        "K": "C",
        "C": "K",
        "s": "ms",
        "ms": "s",
        "Hz": "kHz",
        "kHz": "Hz",
        "eV": "meV",
        "meV": "eV",
        "Pa": "MPa",
        "MPa": "Pa",
        "V": "mV",
        "mV": "V",
        "A": "mA",
        "mA": "A",
        "arb": "normalized",
    }
    return mapping.get(unit, f"{unit}_wrong" if unit else "arb_wrong")


def _apply_plotting_level(x: np.ndarray, y: np.ndarray, level: str) -> tuple[np.ndarray, np.ndarray]:
    n = x.size
    if n < 4:
        return x, y

    if level == "underplotting":
        idx = np.arange(0, n, max(2, n // 45))
        return x[idx], y[idx]

    if level == "overplotting":
        order = np.argsort(x)
        xs = x[order]
        ys = y[order]
        n_new = int(max(n + 10, n * 1.8))
        x_new = np.linspace(float(xs.min()), float(xs.max()), n_new)
        y_new = np.interp(x_new, xs, ys)
        return x_new, y_new

    idx = np.arange(0, n, max(1, n // 110))
    return x[idx], y[idx]


def _complex_distort(x: np.ndarray, y: np.ndarray, mode: str, rng: random.Random) -> np.ndarray:
    if mode == "none" or x.size < 5:
        return y

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_span = max(1e-9, x_max - x_min)
    x_norm = (x - x_min) / x_span
    y_span = max(1e-6, float(np.max(y) - np.min(y)))

    if mode == "wave":
        phase = rng.uniform(0.0, np.pi)
        return y + 0.08 * y_span * np.sin(6 * np.pi * x_norm + phase)

    if mode == "kink":
        kink_at = rng.uniform(0.45, 0.70)
        return y + np.where(x_norm > kink_at, (x_norm - kink_at) * 0.35 * y_span, 0.0)

    if mode == "spike":
        center = rng.uniform(0.30, 0.80)
        width = rng.uniform(0.03, 0.09)
        gauss = np.exp(-((x_norm - center) ** 2) / (2.0 * width * width))
        return y + 0.24 * y_span * gauss

    return y


def build_variations() -> list[VariationConfig]:
    # Exactly 16 variations per plot.
    return [
        VariationConfig("v00", "neutral", plotting_level="neutral", visual_density="medium", resolution_blurriness="medium"),
        VariationConfig("v01", "overplotting", plotting_level="overplotting"),
        VariationConfig("v02", "underplotting", plotting_level="underplotting"),
        VariationConfig("v03", "x_shift", x_shift=0.10),
        VariationConfig("v04", "y_shift", y_shift=-0.10),
        VariationConfig("v05", "curve_misassignment", curve_misassignment=True),
        VariationConfig("v06", "unit_error", unit_error=True),
        VariationConfig("v07", "scale_error", scale_error=1.25),
        VariationConfig("v08", "axis_orientation_swap", axis_orientation_error="swap_xy"),
        VariationConfig("v09", "visual_density_high", visual_density="high"),
        VariationConfig("v10", "visual_density_low", visual_density="low"),
        VariationConfig("v11", "resolution_high", resolution_blurriness="high"),
        VariationConfig("v12", "resolution_low", resolution_blurriness="low"),
        VariationConfig("v13", "complex_distortion_wave", complex_feature_distortion="wave"),
        VariationConfig(
            "v14",
            "combined_stress_a",
            plotting_level="overplotting",
            x_shift=-0.06,
            scale_error=0.85,
            visual_density="high",
            resolution_blurriness="low",
            complex_feature_distortion="spike",
        ),
        VariationConfig(
            "v15",
            "combined_stress_b",
            plotting_level="underplotting",
            y_shift=0.08,
            curve_misassignment=True,
            unit_error=True,
            axis_orientation_error="flip_y",
            visual_density="low",
            resolution_blurriness="high",
            complex_feature_distortion="kink",
        ),
    ]


def _copy_series(series: list[SeriesData]) -> list[SeriesData]:
    return [SeriesData(s.series_id, s.series_label, list(s.x), list(s.y)) for s in series]


def generate_synthetic_base_figure(fig: FigureData, seed: int) -> FigureData:
    figure_seed = seed + int(hashlib.md5(fig.figure_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(figure_seed)

    out_series: list[SeriesData] = []
    for s in fig.series:
        x = np.asarray(s.x, dtype=float)
        y = np.asarray(s.y, dtype=float)
        if x.size < 8:
            continue

        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        x_unique, unique_idx = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[unique_idx]
        if x_unique.size < 4:
            continue

        x_new = np.linspace(float(np.min(x_unique)), float(np.max(x_unique)), 160)
        y_interp = np.interp(x_new, x_unique, y_unique)

        y_span = max(1e-6, float(np.max(y_interp) - np.min(y_interp)))
        noise = rng.normal(loc=0.0, scale=0.025 * y_span, size=x_new.shape[0])
        drift = 0.015 * y_span * np.sin(np.linspace(0, 2.5 * np.pi, x_new.shape[0]))
        y_new = y_interp + noise + drift

        out_series.append(
            SeriesData(
                series_id=s.series_id,
                series_label=f"{s.series_label}_synth",
                x=_float_list(x_new.tolist()),
                y=_float_list(y_new.tolist()),
            )
        )

    if not out_series:
        out_series = _copy_series(fig.series)

    return FigureData(
        figure_id=f"{fig.figure_id}_synth",
        pdf_file=fig.pdf_file,
        page_index=fig.page_index,
        image_index=fig.image_index,
        image_file=fig.image_file,
        descriptor=f"{fig.descriptor} [synthetic]",
        x_label=fig.x_label,
        x_unit=fig.x_unit,
        y_label=fig.y_label,
        y_unit=fig.y_unit,
        series=out_series,
    )


def apply_variation(fig: FigureData, cfg: VariationConfig, seed: int) -> FigureData:
    rng = random.Random(seed)

    varied = FigureData(
        figure_id=f"{fig.figure_id}_{cfg.variation_id}",
        pdf_file=fig.pdf_file,
        page_index=fig.page_index,
        image_index=fig.image_index,
        image_file=fig.image_file,
        descriptor=fig.descriptor,
        x_label=fig.x_label,
        x_unit=fig.x_unit,
        y_label=fig.y_label,
        y_unit=fig.y_unit,
        series=_copy_series(fig.series),
    )

    if cfg.curve_misassignment:
        if len(varied.series) >= 2:
            y_rot = [np.asarray(s.y, dtype=float) for s in varied.series]
            y_rot = [y_rot[-1]] + y_rot[:-1]
            for s, y_new in zip(varied.series, y_rot):
                s.y = _float_list(y_new.tolist())
                s.series_label = f"{s.series_label}_misassigned"
        else:
            s0 = varied.series[0]
            s0.y = list(reversed(s0.y))
            s0.series_label = f"{s0.series_label}_misassigned"

    for s in varied.series:
        x = np.asarray(s.x, dtype=float)
        y = np.asarray(s.y, dtype=float)

        if cfg.scale_error != 1.0:
            y = y * cfg.scale_error

        if cfg.x_shift != 0.0:
            x = x + cfg.x_shift
        if cfg.y_shift != 0.0:
            y = y + cfg.y_shift

        y = _complex_distort(x, y, cfg.complex_feature_distortion, rng)
        x, y = _apply_plotting_level(x, y, cfg.plotting_level)

        s.x = _float_list(x.tolist())
        s.y = _float_list(y.tolist())

    if cfg.axis_orientation_error in {"flip_x", "flip_y", "swap_xy"}:
        all_x = np.concatenate([np.asarray(s.x, dtype=float) for s in varied.series])
        all_y = np.concatenate([np.asarray(s.y, dtype=float) for s in varied.series])
        x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
        y_min, y_max = float(np.min(all_y)), float(np.max(all_y))

        for s in varied.series:
            x = np.asarray(s.x, dtype=float)
            y = np.asarray(s.y, dtype=float)

            if cfg.axis_orientation_error == "flip_x":
                x = x_max + x_min - x
            elif cfg.axis_orientation_error == "flip_y":
                y = y_max + y_min - y
            elif cfg.axis_orientation_error == "swap_xy":
                x, y = y, x

            s.x = _float_list(x.tolist())
            s.y = _float_list(y.tolist())

        if cfg.axis_orientation_error == "swap_xy":
            varied.x_label, varied.y_label = varied.y_label, varied.x_label
            varied.x_unit, varied.y_unit = varied.y_unit, varied.x_unit

    if cfg.unit_error:
        varied.x_unit = _wrong_unit(varied.x_unit)
        varied.y_unit = _wrong_unit(varied.y_unit)

    return varied


def _resolution_params(level: str) -> tuple[tuple[float, float], int, float]:
    if level == "high":
        return (8.0, 5.0), 300, 0.0
    if level == "low":
        return (6.0, 3.8), 90, 1.6
    return (7.2, 4.6), 180, 0.0


def _density_params(level: str) -> tuple[float, float, float]:
    if level == "high":
        return 2.0, 5.0, 0.95
    if level == "low":
        return 0.95, 2.6, 0.72
    return 1.4, 3.8, 0.85


def _format_axis_label(name: str, unit: str) -> str:
    if unit:
        return f"{name} ({unit})"
    return name


def _stats_panel(fig: FigureData) -> str:
    all_x = np.concatenate([np.asarray(s.x, dtype=float) for s in fig.series])
    all_y = np.concatenate([np.asarray(s.y, dtype=float) for s in fig.series])

    slope = 0.0
    if fig.series:
        x0 = np.asarray(fig.series[0].x, dtype=float)
        y0 = np.asarray(fig.series[0].y, dtype=float)
        if x0.size >= 2 and len(np.unique(x0)) >= 2:
            slope = float(np.polyfit(x0, y0, 1)[0])

    return (
        f"n_series={len(fig.series)}\n"
        f"n_points={all_x.size}\n"
        f"x:[{np.min(all_x):.3g},{np.max(all_x):.3g}]\n"
        f"y_mean={np.mean(all_y):.3g}\n"
        f"y_std={np.std(all_y):.3g}\n"
        f"slope0={slope:.3g}"
    )


def render_plot(fig: FigureData, out_path: Path, cfg: VariationConfig | None = None) -> bool:
    try:
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    cfg_local = cfg or VariationConfig("base", "base")

    figsize, dpi, blur_radius = _resolution_params(cfg_local.resolution_blurriness)
    lw, ms, alpha = _density_params(cfg_local.visual_density)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig_mpl, ax = plt.subplots(figsize=figsize, dpi=dpi)

    colors = ["#0B3D91", "#B22222", "#228B22", "#8B008B", "#FF8C00"]
    plotted_x: list[np.ndarray] = []
    plotted_y: list[np.ndarray] = []
    for idx, s in enumerate(fig.series):
        x = np.asarray(s.x, dtype=float)
        y = np.asarray(s.y, dtype=float)
        if x.size == 0 or y.size == 0:
            continue

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        ax.plot(
            x,
            y,
            linewidth=lw,
            alpha=alpha,
            marker="o",
            markersize=ms,
            markerfacecolor="white",
            markeredgewidth=max(0.5, lw * 0.5),
            markeredgecolor=colors[idx % len(colors)],
            color=colors[idx % len(colors)],
            label=s.series_label[:36],
            zorder=2,
        )
        plotted_x.append(x)
        plotted_y.append(y)

    if plotted_x and plotted_y:
        all_x = np.concatenate(plotted_x)
        all_y = np.concatenate(plotted_y)
        y_max_idx = int(np.argmax(all_y))
        y_min_idx = int(np.argmin(all_y))
        x_max_val = float(all_x[y_max_idx])
        y_max_val = float(all_y[y_max_idx])
        x_min_val = float(all_x[y_min_idx])
        y_min_val = float(all_y[y_min_idx])

        ax.scatter([x_max_val], [y_max_val], s=28, color="red", zorder=4)
        ax.scatter([x_min_val], [y_min_val], s=28, color="blue", zorder=4)
        ax.annotate(
            "max",
            xy=(x_max_val, y_max_val),
            xytext=(6, 8),
            textcoords="offset points",
            color="red",
            fontsize=8,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "red"},
            zorder=5,
        )
        ax.annotate(
            "min",
            xy=(x_min_val, y_min_val),
            xytext=(6, -12),
            textcoords="offset points",
            color="blue",
            fontsize=8,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "blue"},
            zorder=5,
        )

    ax.set_title(fig.descriptor[:90])
    ax.set_xlabel(_format_axis_label(fig.x_label, fig.x_unit))
    ax.set_ylabel(_format_axis_label(fig.y_label, fig.y_unit))
    ax.grid(alpha=0.25)
    if len(fig.series) > 1:
        ax.legend(fontsize=8, loc="best")

    panel = _stats_panel(fig)
    if cfg is not None:
        panel += f"\nvar={cfg.variation_id}:{cfg.name}"

    ax.text(
        0.02,
        0.98,
        panel,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.80, "edgecolor": "gray"},
    )

    fig_mpl.tight_layout()
    fig_mpl.savefig(out_path)
    plt.close(fig_mpl)

    if blur_radius > 0:
        try:
            img = Image.open(out_path)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            img.save(out_path)
        except Exception:
            return False

    return True


def _synthetic_points_fieldnames() -> list[str]:
    return [
        "figure_id",
        "parent_figure_id",
        "raw_figure_id",
        "variation_id",
        "variation_name",
        "pdf_file",
        "page_index",
        "image_index",
        "series_id",
        "series_label",
        "x",
        "y",
        "raw_image_file",
        "raw_series_id",
        "raw_series_label",
        "raw_points_csv",
        "raw_figures_csv",
        "raw_series_scalars_json",
        "raw_figure_scalars_json",
        "x_label",
        "x_unit",
        "y_label",
        "y_unit",
        "descriptor",
        "over_underplotting",
        "x_shift",
        "y_shift",
        "curve_misassignment",
        "unit_error",
        "scale_error",
        "axis_orientation_error",
        "visual_density",
        "resolution_blurriness",
        "complex_feature_distortion",
    ]


def _raw_series_lookup(fig: FigureData) -> dict[str, SeriesData]:
    return {str(series.series_id): series for series in fig.series}


def _classify_figure_image(
    gray: np.ndarray, axis_pairs: list[AxisPair], likely_image: bool, context_text: str
) -> tuple[str, float, int]:
    dark_ratio = float((gray < 170).mean()) if gray.size else 1.0
    schematic_hits = _schematic_keyword_score(context_text)
    axis_count = len(axis_pairs)

    # Strong schematic prior: captions with schematic/diagram cues are usually non-xy.
    if schematic_hits >= 3:
        return "schematics", dark_ratio, schematic_hits
    if schematic_hits >= 2 and (dark_ratio > 0.26 or axis_count <= 1):
        return "schematics", dark_ratio, schematic_hits
    if dark_ratio > 0.90 and schematic_hits >= 1:
        return "schematics", dark_ratio, schematic_hits

    if likely_image and axis_count > 0 and dark_ratio < 0.72:
        return "xy_plots", dark_ratio, schematic_hits
    if axis_count >= 2 and dark_ratio < 0.78 and schematic_hits == 0:
        return "xy_plots", dark_ratio, schematic_hits

    if dark_ratio > 0.92 and axis_count == 0:
        return "schematics", dark_ratio, schematic_hits
    if axis_count > 0 and dark_ratio < 0.58 and schematic_hits == 0:
        return "xy_plots", dark_ratio, schematic_hits
    return "arbitrary", dark_ratio, schematic_hits


def export_figure_classification_csv(rows: list[dict[str, Any]], csv_dir: Path) -> Path:
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv = csv_dir / "figure_classification.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pdf_file",
                "page_index",
                "source_image_index",
                "classification",
                "classified_image_file",
                "axis_pair_count",
                "likely_xy_image",
                "dark_ratio",
                "schematic_keyword_hits",
                "accepted_subplot_count",
                "context_excerpt",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return out_csv


def extract_plot_images_from_pdf(
    pdf_path: Path,
    figures_dir: Path,
    classified_dirs: dict[str, Path] | None = None,
    classification_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return assets

    pdf_slug = _slug(pdf_path.stem)
    pdf_out_dir = figures_dir / pdf_slug
    pdf_out_dir.mkdir(parents=True, exist_ok=True)

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_text = page.get_text("text") or ""
        page_caption_context = _extract_caption_context(page_text)
        images = page.get_images(full=True)
        accepted_page_bboxes: list[tuple[float, float, float, float]] = []

        for image_index, img in enumerate(images):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
            except Exception:
                continue

            image_bytes = base.get("image")
            if image_bytes is None:
                continue

            try:
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception:
                continue

            context_text = page_text
            rects: list[fitz.Rect] = []
            try:
                rects = page.get_image_rects(xref)
                if rects:
                    rect = rects[0]
                    pad_x = max(16.0, 0.08 * rect.width)
                    pad_y = max(14.0, 0.08 * rect.height)
                    clip = fitz.Rect(
                        max(page.rect.x0, rect.x0 - pad_x),
                        max(page.rect.y0, rect.y0 - pad_y),
                        min(page.rect.x1, rect.x1 + pad_x),
                        min(page.rect.y1, rect.y1 + pad_y),
                    )
                    clipped_text = page.get_text("text", clip=clip) or ""
                    if clipped_text.strip():
                        context_text = clipped_text
            except Exception:
                pass

            image_caption_context = _extract_caption_context(context_text)
            caption_context = image_caption_context if image_caption_context else page_caption_context
            if page_caption_context and page_caption_context not in caption_context:
                caption_context = f"{caption_context} {page_caption_context}".strip()

            arr = np.array(pil_img)
            gray = np.array(pil_img.convert("L"))
            axis_pairs = _detect_axis_pairs(gray, max_pairs=16)
            likely_image = _is_likely_plot(arr, page_text=context_text)
            classification, image_dark_ratio, schematic_hits = _classify_figure_image(
                gray=gray, axis_pairs=axis_pairs, likely_image=likely_image, context_text=context_text
            )

            accepted_subplots = 0
            # Extraction path is strictly for figures initially classified as x-y plots.
            if axis_pairs and classification == "xy_plots":
                image_rect = rects[0] if rects else None
                for sub_index, axis_pair in enumerate(axis_pairs):
                    if _is_composite_axis_pair(axis_pair, axis_pairs, img_w=int(arr.shape[1]), img_h=int(arr.shape[0])):
                        continue
                    axis_meta = _extract_axis_metadata_from_figure(
                        page=page,
                        image_rect=image_rect,
                        img_w=int(arr.shape[1]),
                        img_h=int(arr.shape[0]),
                        axis_pair=axis_pair,
                        fallback_text=context_text,
                        caption_text=caption_context,
                    )
                    x0, y0, x1, y1 = axis_pair.plot_bbox
                    if x1 > x0 and y1 > y0:
                        plot_crop = gray[y0:y1, x0:x1]
                        axis_meta["plot_dark_ratio"] = float((plot_crop < 170).mean())
                    else:
                        axis_meta["plot_dark_ratio"] = 1.0
                    gx_ticks, gy_ticks = _estimate_axis_tick_counts(gray, axis_pair)
                    axis_meta["x_tick_count_geom"] = int(gx_ticks)
                    axis_meta["y_tick_count_geom"] = int(gy_ticks)
                    geom_tick_max = max(int(gx_ticks), int(gy_ticks))
                    pair_context_text = _axis_pair_local_context(
                        page=page,
                        image_rect=image_rect,
                        img_w=int(arr.shape[1]),
                        img_h=int(arr.shape[0]),
                        axis_pair=axis_pair,
                        fallback_text=context_text,
                        pad_mult=0.95,
                    )
                    axis_meta = _enrich_axis_meta_from_context(axis_meta, pair_context_text, source_tag="pair_context")

                    # User requirement: keep only perpendicular x-y axes with visible ticks.
                    has_bi_axis_ticks = int(gx_ticks) >= 2 and int(gy_ticks) >= 2
                    if not has_bi_axis_ticks:
                        continue

                    if not _axis_meta_is_strict_xy(axis_meta, context_text=pair_context_text):
                        fallback_text_evidence = (
                            int(axis_meta.get("x_tick_count") or 0) > 0
                            or int(axis_meta.get("y_tick_count") or 0) > 0
                            or str(axis_meta.get("x_label_source") or "").startswith("local")
                            or str(axis_meta.get("y_label_source") or "").startswith("local")
                            or str(axis_meta.get("x_label_source") or "") in {"embedded", "embedded_context", "pair_context"}
                            or str(axis_meta.get("y_label_source") or "") in {"embedded", "embedded_context", "pair_context"}
                        )
                        # High-recall fallback: keep only sparse, axis-like geometry when metadata is sparse.
                        if not (
                            fallback_text_evidence
                            and
                            _schematic_keyword_score(pair_context_text) == 0
                            and likely_image
                            and axis_pair.score >= 2.2
                            and axis_meta["plot_dark_ratio"] < 0.45
                            and geom_tick_max >= 4
                        ):
                            continue

                    asset_image_index = int(image_index * 100 + sub_index)
                    out_name = f"{pdf_slug}_p{page_index:03d}_i{image_index:03d}_s{sub_index:02d}.png"
                    out_path = pdf_out_dir / out_name
                    cx0, cy0, cx1, cy1 = _boxed_subplot_crop_bounds(
                        gray,
                        axis_pair,
                        left_frac=0.26,
                        right_frac=0.11,
                        top_frac=0.10,
                        bottom_frac=0.30,
                        min_pad_px=10,
                    )
                    if cx1 - cx0 < 40 or cy1 - cy0 < 35:
                        continue

                    crop_rgb = arr[cy0:cy1, cx0:cx1]
                    if crop_rgb.size == 0:
                        continue

                    axis_meta["plot_bbox"] = [int(x0 - cx0), int(y0 - cy0), int(x1 - cx0), int(y1 - cy0)]
                    x_scale = axis_meta.get("x_scale")
                    if isinstance(x_scale, (list, tuple)) and len(x_scale) == 2:
                        a, b = float(x_scale[0]), float(x_scale[1])
                        axis_meta["x_scale"] = [a, b + a * float(cx0)]
                    y_scale = axis_meta.get("y_scale")
                    if isinstance(y_scale, (list, tuple)) and len(y_scale) == 2:
                        a, b = float(y_scale[0]), float(y_scale[1])
                        axis_meta["y_scale"] = [a, b + a * float(cy0)]

                    try:
                        Image.fromarray(crop_rgb.astype(np.uint8), mode="RGB").save(out_path)
                    except Exception:
                        continue

                    assets.append(
                        {
                            "pdf_file": str(pdf_path),
                            "page_index": page_index,
                            "image_index": asset_image_index,
                            "image_path": str(out_path),
                            "page_text": page_text,
                            "context_text": _merge_caption_context(pair_context_text, caption_context),
                            "axis_meta": axis_meta,
                            "source_image_index": image_index,
                            "subplot_index": sub_index,
                        }
                    )
                    pb = _safe_float_bbox(axis_meta.get("plot_bbox_page"))
                    if pb is not None:
                        accepted_page_bboxes.append(pb)
                    accepted_subplots += 1

            final_classification = classification
            if final_classification == "xy_plots" and accepted_subplots == 0:
                final_classification = "arbitrary"
            if final_classification == "schematics":
                final_classification = "arbitrary"

            classified_file = ""
            if classified_dirs is not None:
                class_root = classified_dirs.get(final_classification)
                if class_root is not None:
                    class_pdf_dir = class_root / pdf_slug
                    class_pdf_dir.mkdir(parents=True, exist_ok=True)
                    class_name = f"{pdf_slug}_p{page_index:03d}_i{image_index:03d}.png"
                    class_path = class_pdf_dir / class_name
                    try:
                        pil_img.save(class_path)
                        classified_file = str(class_path)
                    except Exception:
                        classified_file = ""

            if classification_rows is not None:
                classification_rows.append(
                    {
                        "pdf_file": str(pdf_path),
                        "page_index": int(page_index),
                        "source_image_index": int(image_index),
                        "classification": final_classification,
                        "classified_image_file": classified_file,
                        "axis_pair_count": int(len(axis_pairs)),
                        "likely_xy_image": int(1 if likely_image else 0),
                        "dark_ratio": float(image_dark_ratio),
                        "schematic_keyword_hits": int(schematic_hits),
                        "accepted_subplot_count": int(accepted_subplots),
                        "context_excerpt": re.sub(r"\s+", " ", context_text.strip())[:220],
                    }
                )

        # Page-level fallback: capture vector/embedded plots not exposed as standalone image objects.
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            page_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n > 3:
                page_rgb = page_rgb[:, :, :3]
            page_gray = page_rgb.mean(axis=2).astype(np.uint8)
        except Exception:
            page_rgb = None
            page_gray = None

        if page_rgb is not None and page_gray is not None:
            page_axis_pairs = _detect_axis_pairs(page_gray, max_pairs=24)
            for page_sub_index, axis_pair in enumerate(page_axis_pairs):
                if _is_composite_axis_pair(
                    axis_pair,
                    page_axis_pairs,
                    img_w=int(page_rgb.shape[1]),
                    img_h=int(page_rgb.shape[0]),
                ):
                    continue
                gx_ticks, gy_ticks = _estimate_axis_tick_counts(page_gray, axis_pair)
                if int(gx_ticks) < 2 or int(gy_ticks) < 2:
                    continue

                pair_context_text = _axis_pair_local_context(
                    page=page,
                    image_rect=page.rect,
                    img_w=int(page_rgb.shape[1]),
                    img_h=int(page_rgb.shape[0]),
                    axis_pair=axis_pair,
                    fallback_text=page_text,
                    pad_mult=1.2,
                )
                axis_meta = _extract_axis_metadata_from_figure(
                    page=page,
                    image_rect=page.rect,
                    img_w=int(page_rgb.shape[1]),
                    img_h=int(page_rgb.shape[0]),
                    axis_pair=axis_pair,
                    fallback_text=pair_context_text,
                    caption_text=page_caption_context,
                )
                x0, y0, x1, y1 = axis_pair.plot_bbox
                if x1 <= x0 or y1 <= y0:
                    continue
                plot_crop = page_gray[y0:y1, x0:x1]
                axis_meta["plot_dark_ratio"] = float((plot_crop < 170).mean()) if plot_crop.size else 1.0
                axis_meta["x_tick_count_geom"] = int(gx_ticks)
                axis_meta["y_tick_count_geom"] = int(gy_ticks)

                if not _axis_meta_is_strict_xy(axis_meta, context_text=pair_context_text):
                    continue

                pb = _safe_float_bbox(axis_meta.get("plot_bbox_page"))
                if pb is not None and any(_bbox_iou(pb, prev) > 0.58 for prev in accepted_page_bboxes):
                    continue

                cx0, cy0, cx1, cy1 = _boxed_subplot_crop_bounds(
                    page_gray,
                    axis_pair,
                    left_frac=0.28,
                    right_frac=0.12,
                    top_frac=0.12,
                    bottom_frac=0.32,
                    min_pad_px=12,
                )
                if cx1 - cx0 < 40 or cy1 - cy0 < 35:
                    continue

                crop_rgb = page_rgb[cy0:cy1, cx0:cx1]
                if crop_rgb.size == 0:
                    continue

                axis_meta["plot_bbox"] = [int(x0 - cx0), int(y0 - cy0), int(x1 - cx0), int(y1 - cy0)]
                x_scale = axis_meta.get("x_scale")
                if isinstance(x_scale, (list, tuple)) and len(x_scale) == 2:
                    a, b = float(x_scale[0]), float(x_scale[1])
                    axis_meta["x_scale"] = [a, b + a * float(cx0)]
                y_scale = axis_meta.get("y_scale")
                if isinstance(y_scale, (list, tuple)) and len(y_scale) == 2:
                    a, b = float(y_scale[0]), float(y_scale[1])
                    axis_meta["y_scale"] = [a, b + a * float(cy0)]

                source_image_index = int(900 + page_sub_index)
                out_name = f"{pdf_slug}_p{page_index:03d}_i{source_image_index:03d}_s00.png"
                out_path = pdf_out_dir / out_name
                try:
                    Image.fromarray(crop_rgb.astype(np.uint8), mode="RGB").save(out_path)
                except Exception:
                    continue

                assets.append(
                    {
                        "pdf_file": str(pdf_path),
                        "page_index": page_index,
                        "image_index": int(source_image_index * 100),
                        "image_path": str(out_path),
                        "page_text": page_text,
                        "context_text": _merge_caption_context(pair_context_text, page_caption_context),
                        "axis_meta": axis_meta,
                        "source_image_index": source_image_index,
                        "subplot_index": 0,
                    }
                )
                if pb is not None:
                    accepted_page_bboxes.append(pb)

    doc.close()
    return _filter_redundant_assets(assets)


def _variation_stats_row(fig: FigureData, parent_figure_id: str, cfg: VariationConfig, plot_path: Path) -> dict[str, Any]:
    scalars = _figure_scalars(fig)
    out = {
        "figure_id": fig.figure_id,
        "parent_figure_id": parent_figure_id,
        "variation_id": cfg.variation_id,
        "variation_name": cfg.name,
        "plot_file": str(plot_path),
        "n_series": int(scalars["n_series"]),
        "n_points_total": int(scalars["n_points_total"]),
        "x_min": scalars["x_min"],
        "x_max": scalars["x_max"],
        "y_min": scalars["y_min"],
        "y_max": scalars["y_max"],
        "y_mean": scalars["y_mean"],
        "y_std": scalars["y_std"],
    }
    out.update(cfg.as_mode_dict())
    return out


def run_pipeline(
    pdf_dir: Path,
    out_dir: Path,
    max_pdfs: int | None = None,
    max_figures_per_pdf: int | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    dirs = ensure_dirs(out_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    pdfs = discover_pdfs(pdf_dir, max_pdfs=max_pdfs)

    extracted_assets: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    figures: list[FigureData] = []

    for pdf_path in pdfs:
        assets = extract_plot_images_from_pdf(
            pdf_path,
            dirs["figures"],
            classified_dirs={
                "xy_plots": dirs["classified_xy"],
                "schematics": dirs["classified_arbitrary"],
                "arbitrary": dirs["classified_arbitrary"],
            },
            classification_rows=classification_rows,
        )
        if max_figures_per_pdf is not None:
            assets = assets[: max(0, max_figures_per_pdf)]

        extracted_assets.extend(assets)

        for asset in assets:
            fig = digitize_figure_image(
                image_path=Path(asset["image_path"]),
                page_text=str(asset.get("context_text") or asset.get("page_text") or ""),
                pdf_file=str(asset["pdf_file"]),
                page_index=int(asset["page_index"]),
                image_index=int(asset["image_index"]),
                axis_meta=asset.get("axis_meta"),
            )
            if fig is None:
                continue
            if not fig.series:
                continue
            if max(len(s.x) for s in fig.series) < 16:
                continue
            figures.append(fig)

    classification_csv = export_figure_classification_csv(classification_rows, dirs["csv"])
    classification_counts = {
        "xy_plots": sum(1 for r in classification_rows if str(r.get("classification")) == "xy_plots"),
        "schematics": sum(1 for r in classification_rows if str(r.get("classification")) == "schematics"),
        "arbitrary": sum(1 for r in classification_rows if str(r.get("classification")) == "arbitrary"),
    }

    gt_points_csv, gt_figures_csv = export_ground_truth_csv(figures, dirs["csv"])

    synthetic_points_csv = dirs["csv"] / "synthetic_xy_points.csv"
    synthetic_variation_csv = dirs["csv"] / "synthetic_variations.csv"

    variations = build_variations()

    raw_plots_count = 0
    synthetic_points_count = 0
    synthetic_plots_count = 0
    synthetic_base_count = 0

    with synthetic_points_csv.open("w", encoding="utf-8", newline="") as f_points, synthetic_variation_csv.open(
        "w", encoding="utf-8", newline=""
    ) as f_vars:
        points_writer = csv.DictWriter(f_points, fieldnames=_synthetic_points_fieldnames())
        points_writer.writeheader()

        var_fieldnames = [
            "figure_id",
            "parent_figure_id",
            "variation_id",
            "variation_name",
            "plot_file",
            "n_series",
            "n_points_total",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "y_mean",
            "y_std",
            "over_underplotting",
            "x_shift",
            "y_shift",
            "curve_misassignment",
            "unit_error",
            "scale_error",
            "axis_orientation_error",
            "visual_density",
            "resolution_blurriness",
            "complex_feature_distortion",
        ]
        vars_writer = csv.DictWriter(f_vars, fieldnames=var_fieldnames)
        vars_writer.writeheader()

        for fig in figures:
            raw_plot = dirs["plots_raw"] / f"{_slug(fig.figure_id)}.png"
            if render_plot(fig, raw_plot):
                raw_plots_count += 1

            base_fig = generate_synthetic_base_figure(fig, seed=seed)
            raw_series_by_id = _raw_series_lookup(fig)
            raw_figure_scalars_json = json.dumps(_figure_scalars(fig), sort_keys=True)
            base_plot = dirs["plots_base"] / f"{_slug(base_fig.figure_id)}.png"
            if render_plot(base_fig, base_plot):
                synthetic_plots_count += 1
                synthetic_base_count += 1

            for cfg in variations:
                variation_seed = seed + int(hashlib.md5(f"{base_fig.figure_id}_{cfg.variation_id}".encode("utf-8")).hexdigest()[:8], 16)
                varied = apply_variation(base_fig, cfg, variation_seed)

                fig_out_dir = dirs["plots_aug"] / _slug(base_fig.figure_id)
                fig_out_dir.mkdir(parents=True, exist_ok=True)
                out_plot = fig_out_dir / f"{cfg.variation_id}_{_slug(cfg.name)}.png"

                if render_plot(varied, out_plot, cfg=cfg):
                    synthetic_plots_count += 1

                vars_writer.writerow(_variation_stats_row(varied, base_fig.figure_id, cfg, out_plot))

                mode = cfg.as_mode_dict()
                for s in varied.series:
                    raw_series = raw_series_by_id.get(str(s.series_id))
                    raw_series_id = raw_series.series_id if raw_series is not None else ""
                    raw_series_label = raw_series.series_label if raw_series is not None else ""
                    raw_series_scalars_json = (
                        json.dumps(_series_scalars(raw_series), sort_keys=True) if raw_series is not None else ""
                    )
                    for x_val, y_val in zip(s.x, s.y):
                        row = {
                            "figure_id": varied.figure_id,
                            "parent_figure_id": base_fig.figure_id,
                            "raw_figure_id": fig.figure_id,
                            "variation_id": cfg.variation_id,
                            "variation_name": cfg.name,
                            "pdf_file": varied.pdf_file,
                            "page_index": varied.page_index,
                            "image_index": varied.image_index,
                            "series_id": s.series_id,
                            "series_label": s.series_label,
                            "x": float(x_val),
                            "y": float(y_val),
                            "raw_image_file": fig.image_file,
                            "raw_series_id": raw_series_id,
                            "raw_series_label": raw_series_label,
                            "raw_points_csv": str(gt_points_csv),
                            "raw_figures_csv": str(gt_figures_csv),
                            "raw_series_scalars_json": raw_series_scalars_json,
                            "raw_figure_scalars_json": raw_figure_scalars_json,
                            "x_label": varied.x_label,
                            "x_unit": varied.x_unit,
                            "y_label": varied.y_label,
                            "y_unit": varied.y_unit,
                            "descriptor": varied.descriptor,
                            **mode,
                        }
                        points_writer.writerow(row)
                        synthetic_points_count += 1

    report = {
        "run_id": run_id,
        "input_pdf_dir": str(pdf_dir),
        "output_dir": str(out_dir),
        "pdf_count": len(pdfs),
        "extracted_plot_images": len(extracted_assets),
        "digitized_figures": len(figures),
        "classified_figures_total": len(classification_rows),
        "classified_xy_plots": int(classification_counts["xy_plots"]),
        "classified_schematics": int(classification_counts["schematics"]),
        "classified_arbitrary": int(classification_counts["arbitrary"]),
        "figure_classification_csv": str(classification_csv),
        "ground_truth_points_csv": str(gt_points_csv),
        "ground_truth_figures_csv": str(gt_figures_csv),
        "synthetic_points_csv": str(synthetic_points_csv),
        "synthetic_variations_csv": str(synthetic_variation_csv),
        "raw_plots": raw_plots_count,
        "base_synthetic_plots": synthetic_base_count,
        "synthetic_aug_variations_per_plot": len(variations),
        "synthetic_plots_written_total": synthetic_plots_count,
        "synthetic_points_written": synthetic_points_count,
        "augment_modes": [
            "over_underplotting",
            "x_shift",
            "y_shift",
            "curve_misassignment",
            "unit_error",
            "scale_error",
            "axis_orientation_error",
            "visual_density",
            "resolution_blurriness",
            "complex_feature_distortion",
        ],
    }

    report_path = dirs["reports"] / f"run_report_{run_id}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report["report_json"] = str(report_path)
    return report
