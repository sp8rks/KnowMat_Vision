from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


NIST_MDR_ZIP_URL = "https://materialsdata.nist.gov/bitstream/handle/11256/950/dictionaries_immi_20171205.zip?isAllowed=y&sequence=1"


def _normalize_key(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.replace("−", "-")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _relaxed_json_load(text: str) -> dict:
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(cleaned)


def _unit_to_symbol(raw: str) -> str:
    u = (raw or "").strip().lower()
    mapping = {
        "pascal": "Pa",
        "meter^2/second": "m^2s^-1",
        "1/(pascal*second)": "Pa^-1s^-1",
        "vickers hardness number": "HV",
        "hv": "HV",
        "watt/meter*kelvin": "Wm^-1K^-1",
        "watt/meter^2": "Wm^-2",
        "newton/ampere^2": "NA^-2",
        "kelvin": "K",
        "joule/mole": "Jmol^-1",
        "joule/(mol*kelvin)": "Jmol^-1K^-1",
        "siemens/meter": "Sm^-1",
        "ohm meter": "ohm m",
    }
    return mapping.get(u, raw.strip())


def _extract_rows_from_dictionary(json_path: Path) -> list[dict[str, str]]:
    try:
        obj = _relaxed_json_load(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    instances = obj.get("instances")
    if not isinstance(instances, list):
        return []

    rows: list[dict[str, str]] = []
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        param = str(inst.get("name") or "").strip()
        if not param:
            continue

        unit = ""
        units = inst.get("units")
        if isinstance(units, list) and units:
            first = units[0]
            if isinstance(first, dict):
                unit_raw = str(first.get("symbol") or first.get("name") or "").strip()
                if not unit_raw:
                    # Handle malformed dictionaries with duplicate keys.
                    vals = [str(v).strip() for v in first.values() if str(v).strip()]
                    unit_raw = vals[0] if vals else ""
                unit = _unit_to_symbol(unit_raw)

        aliases: list[str] = []
        syn = inst.get("synonyms")
        if isinstance(syn, list):
            for s in syn:
                if not isinstance(s, dict):
                    continue
                name = str(s.get("name") or "").strip()
                if name:
                    aliases.append(name)

        rows.append(
            {
                "parameter": param,
                "unit": unit,
                "aliases": "|".join(dict.fromkeys(aliases)),
            }
        )
    return rows


def _read_existing(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [
            {
                "parameter": str(r.get("parameter") or "").strip(),
                "unit": str(r.get("unit") or "").strip(),
                "aliases": str(r.get("aliases") or "").strip(),
            }
            for r in reader
            if str(r.get("parameter") or "").strip()
        ]


def _merge(existing: list[dict[str, str]], imported: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}

    def upsert(row: dict[str, str], imported_row: bool) -> None:
        key = _normalize_key(row["parameter"])
        if not key:
            return
        cur = merged.get(key)
        if cur is None:
            merged[key] = {
                "parameter": row["parameter"],
                "unit": row["unit"],
                "aliases": row["aliases"],
            }
            return

        # Prefer existing custom unit definitions unless missing/arb.
        if imported_row:
            if (not cur["unit"] or cur["unit"] == "arb") and row["unit"]:
                cur["unit"] = row["unit"]
        else:
            if row["unit"]:
                cur["unit"] = row["unit"]

        alias_parts = [a for a in [cur.get("aliases", ""), row.get("aliases", "")] if a]
        aliases: list[str] = []
        for part in alias_parts:
            aliases.extend([a.strip() for a in part.split("|") if a.strip()])
        cur["aliases"] = "|".join(dict.fromkeys(aliases))

    for row in imported:
        upsert(row, imported_row=True)
    for row in existing:
        upsert(row, imported_row=False)

    out = list(merged.values())
    out.sort(key=lambda r: r["parameter"].lower())
    return out


def _write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["parameter", "unit", "aliases"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import parameter-unit list from NIST MDR dictionaries.")
    parser.add_argument(
        "--zip-path",
        default="",
        help="Optional local ZIP path. If omitted, downloads from NIST MDR.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(Path(__file__).resolve().parents[1] / "src" / "material_parameter_units.csv"),
        help="Destination CSV to update/merge.",
    )
    args = parser.parse_args()

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="nist_mdr_units_"))
    try:
        zip_path = Path(args.zip_path).resolve() if args.zip_path else tmp_dir / "nist_mdr_dictionaries.zip"
        if not args.zip_path:
            urllib.request.urlretrieve(NIST_MDR_ZIP_URL, str(zip_path))

        unpack_dir = tmp_dir / "unzipped"
        unpack_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(unpack_dir)

        imported_rows: list[dict[str, str]] = []
        for json_file in unpack_dir.rglob("*.json"):
            imported_rows.extend(_extract_rows_from_dictionary(json_file))

        existing_rows = _read_existing(out_csv)
        merged = _merge(existing_rows, imported_rows)
        _write_rows(out_csv, merged)

        print(f"Imported rows from NIST dictionaries: {len(imported_rows)}")
        print(f"Rows written after merge: {len(merged)}")
        print(f"Output CSV: {out_csv}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

