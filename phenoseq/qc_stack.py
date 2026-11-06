#!/usr/bin/env python3
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path
from typing import TypeAlias
import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.windows import Window, from_bounds

NODATA = -32768
SCALE = {"S2": 10000.0, "S1": 100.0}
YEARS = range(2018, 2026)
BLOCK = 8
SUBSET = (497500.0, 4074500.0, 507500.0, 4084500.0)
F32: TypeAlias = npt.NDArray[np.float32]


def block_mean(a: F32, k: int) -> tuple[F32, F32]:
    h, w = (a.shape[0] // k * k, a.shape[1] // k * k)
    b = a[:h, :w].reshape(h // k, k, w // k, k)
    valid = np.isfinite(b)
    n = valid.sum(axis=(1, 3)).astype(np.float32)
    s = np.where(valid, b, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, s / n, np.nan).astype(np.float32)
    return (m, (n / (k * k)).astype(np.float32))


def to_int16(a: F32, scale: float) -> npt.NDArray[np.int16]:
    out = np.where(np.isfinite(a), np.round(a * scale), NODATA)
    return out.astype(np.int16)


def main(folder: Path) -> None:
    rows: list[dict[str, object]] = []
    report: list[str] = []
    store: dict[str, np.ndarray] = {}
    ref_profile: tuple[object, ...] | None = None
    files = []
    for y in YEARS:
        files += [("S2", y, folder / f"aoi_s2_ndvi_evi_{y}.tif")]
        files += [("S1", y, folder / f"aoi_s1_vv_vh_{y}.tif")]
    for sensor, year, path in files:
        if not path.exists():
            report.append(f"MISSING  {path.name}")
            continue
        with rasterio.open(path) as ds:
            geom = (
                ds.crs.to_epsg(),
                ds.width,
                ds.height,
                tuple((round(v, 1) for v in ds.bounds)),
            )
            if ref_profile is None:
                ref_profile = geom
                report.append(
                    f"Grid: EPSG:{geom[0]}  {geom[1]} x {geom[2]} px  bounds {geom[3]}"
                )
                report.append(
                    f"Pixel size: {ds.res}  dtype: {ds.dtypes[0]}  bands: {ds.count}"
                )
            elif geom != ref_profile:
                report.append(f"GRID DIFFERS  {path.name}: {geom}")
            names = list(ds.descriptions)
            if not all(names):
                names = [f"b{i + 1:02d}" for i in range(ds.count)]
            sub_win: Window | None = None
            if year == 2025:
                sub_win = (
                    from_bounds(*SUBSET, transform=ds.transform)
                    .round_offsets()
                    .round_lengths()
                )
            means, fracs, subs = ([], [], [])
            report.append(f"\n{path.name}")
            report.append(
                f"  {'band':8s} {'valid %':>8s} {'median':>9s} {'p02':>9s} {'p98':>9s}"
            )
            for i, name in enumerate(names, start=1):
                raw = ds.read(i)
                a = np.where(raw == NODATA, np.nan, raw / SCALE[sensor]).astype(
                    np.float32
                )
                v = a[np.isfinite(a)]
                share = 100.0 * v.size / a.size
                med, p02, p98 = (
                    (float(np.median(v)), *map(float, np.percentile(v, [2, 98])))
                    if v.size
                    else (np.nan, np.nan, np.nan)
                )
                rows.append(
                    {
                        "file": path.name,
                        "sensor": sensor,
                        "year": year,
                        "band": name,
                        "valid_pct": round(share, 2),
                        "median": round(med, 4),
                        "p02": round(p02, 4),
                        "p98": round(p98, 4),
                    }
                )
                flag = (
                    "  <-- empty"
                    if share == 0
                    else "  <-- sparse"
                    if share < 50
                    else ""
                )
                report.append(
                    f"  {name:8s} {share:8.1f} {med:9.3f} {p02:9.3f} {p98:9.3f}{flag}"
                )
                m, f = block_mean(a, BLOCK)
                means.append(to_int16(m, SCALE[sensor]))
                fracs.append(np.round(f * 100).astype(np.uint8))
                if sub_win is not None:
                    subs.append(ds.read(i, window=sub_win))
            key = f"{sensor.lower()}_{year}"
            store[key] = np.stack(means)
            if sensor == "S2":
                store[key + "_validpct"] = np.stack(fracs)
            store[key + "_bands"] = np.array(names)
            if subs:
                store[key + "_subset30m"] = np.stack(subs)
            if "transform_240m" not in store:
                t = ds.transform
                store["transform_240m"] = np.array(
                    [t.a * BLOCK, t.b, t.c, t.d, t.e * BLOCK, t.f]
                )
                store["crs_epsg"] = np.array(ds.crs.to_epsg())
            if sub_win is not None and "subset_bounds" not in store:
                store["subset_bounds"] = np.array(ds.window_bounds(sub_win))
    with open(folder / "aoi_qc_bands.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["file"])
        wr.writeheader()
        wr.writerows(rows)
    report.append("\nMonths with < 50 % valid pixels (per file and variable):")
    for sensor, year, path in files:
        sub = [r for r in rows if r["file"] == path.name]
        for var in sorted({re.sub("_\\d\\d$", "", str(r["band"])) for r in sub}):
            bad = [
                str(r["band"])[-2:]
                for r in sub
                if str(r["band"]).startswith(var) and float(str(r["valid_pct"])) < 50
            ]
            if bad:
                report.append(f"  {path.name:28s} {var:5s} months {', '.join(bad)}")
    text = "\n".join(report)
    (folder / "aoi_qc_report.txt").write_text(text + "\n")
    np.savez_compressed(folder / "aoi_summary_240m.npz", **store)
    print(text)
    out = folder / "aoi_summary_240m.npz"
    print(f"\nWrote {out} ({out.stat().st_size / 1000000.0:.1f} MB)")


if __name__ == "__main__":
    default = Path(__file__).resolve().parent.parent / "GEE_data"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
