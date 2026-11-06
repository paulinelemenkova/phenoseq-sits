#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypeAlias
import numpy as np
import numpy.typing as npt


def _use_bundled_proj() -> None:
    import importlib.util
    import os

    spec = importlib.util.find_spec("rasterio")
    if spec and spec.origin:
        bundled = Path(spec.origin).parent / "proj_data"
        if (bundled / "proj.db").exists():
            os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = str(bundled)


_use_bundled_proj()
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject

F32: TypeAlias = npt.NDArray[np.float32]
F64: TypeAlias = npt.NDArray[np.float64]
I16: TypeAlias = npt.NDArray[np.int16]
U8: TypeAlias = npt.NDArray[np.uint8]
YEARS = list(range(2018, 2026))
NODATA = -32768
try:
    EPSG: CRS | None = CRS.from_epsg(32637)
except Exception:
    EPSG = None
DOY = np.array(
    [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=np.float64
)
CLASSES = {
    1: "winter wheat",
    2: "wheat + second-crop maize",
    3: "cotton",
    4: "main-crop maize",
    5: "fallow",
    6: "perennial",
}
CFG: dict[str, Any] = {
    "harmonics": 3,
    "alpha": 0.5,
    "season_peak_min": 0.5,
    "season_amp_min": 0.3,
    "fit_rmse_max": 0.08,
    "wheat_pos": (60, 135),
    "summer_pos_late": (213, 290),
    "cotton_sos": (121, 200),
    "maize_main_pos": (166, 212),
    "fallow_max": 0.3,
    "perennial_min": 0.35,
    "block_px": 100,
    "test_share": 0.3,
    "per_class_year": 4000,
    "rf_trees": [200, 500],
    "rf_max_features": ["sqrt", 0.33],
    "cv_max": 30000,
    "seed": 42,
}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_stack(path: Path, scale: float) -> tuple[F32, dict[str, Any]]:
    with rasterio.open(path) as ds:
        raw = ds.read()
        prof = ds.profile.copy()
    if EPSG is not None:
        prof["crs"] = EPSG
    out = np.where(raw == NODATA, np.nan, raw / scale).astype(np.float32)
    return (out, prof)


def write_tif(
    path: Path,
    arr: npt.NDArray[Any],
    prof: dict[str, Any],
    names: list[str] | None = None,
    nodata: float | None = None,
) -> None:
    a = arr if arr.ndim == 3 else arr[None]
    p = prof.copy()
    p.update(
        count=a.shape[0],
        dtype=a.dtype.name,
        compress="lzw",
        nodata=nodata,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        predictor=2 if a.dtype.kind in "iu" else 1,
    )
    with rasterio.open(path, "w", **p) as ds:
        ds.write(a)
        for i, n in enumerate(names or [], start=1):
            ds.set_band_description(i, n)


def align(
    path: Path, prof: dict[str, Any], bands: int | None = None
) -> npt.NDArray[Any] | None:
    if not path.exists():
        return None
    with rasterio.open(path) as ds:
        n = bands or ds.count
        dst = np.zeros((n, prof["height"], prof["width"]), dtype=ds.dtypes[0])
        for b in range(n):
            reproject(
                rasterio.band(ds, b + 1),
                dst[b],
                dst_transform=prof["transform"],
                dst_crs=prof["crs"],
                resampling=Resampling.nearest,
            )
    return dst


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def chunks(n: int, size: int) -> Iterator[slice]:
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


def design(k: int, t: F64) -> F64:
    cols = [np.ones_like(t)]
    for j in range(1, k + 1):
        w = 2 * np.pi * j * t / 365.0
        cols += [np.cos(w), np.sin(w)]
    return np.stack(cols, axis=1)


def harmonic_fill(v: F64, k: int) -> tuple[F64, F64]:
    x = design(k, DOY)
    w = np.isfinite(v).astype(np.float64)
    y = np.nan_to_num(v)
    xtwx = np.einsum("tp,nt,tq->npq", x, w, x) + 1e-06 * np.eye(x.shape[1])
    xtwy = np.einsum("tp,nt,nt->np", x, w, y)
    coef = np.linalg.solve(xtwx, xtwy[..., None])[..., 0]
    fit = coef @ x.T
    rmse = np.sqrt(np.sum(w * (fit - y) ** 2, axis=1) / np.maximum(w.sum(axis=1), 1))
    return (np.where(w > 0, v, fit), rmse)


def dlog(t: F64, p: F64) -> F64:
    vmin, vmax, m1, n1, m2, n2 = (p[:, i : i + 1] for i in range(6))
    s1 = 1 / (1 + np.exp(-m1 * (t - n1)))
    s2 = 1 / (1 + np.exp(-m2 * (t - n2)))
    return vmin + (vmax - vmin) * (s1 - s2)


def jacobian(t: F64, p: F64) -> F64:
    vmin, vmax, m1, n1, m2, n2 = (p[:, i : i + 1] for i in range(6))
    s1 = 1 / (1 + np.exp(-m1 * (t - n1)))
    s2 = 1 / (1 + np.exp(-m2 * (t - n2)))
    a = vmax - vmin
    d1, d2 = (s1 * (1 - s1), s2 * (1 - s2))
    return np.stack(
        [
            1 - (s1 - s2),
            s1 - s2,
            a * d1 * (t - n1),
            -a * d1 * m1,
            -a * d2 * (t - n2),
            a * d2 * m2,
        ],
        axis=2,
    )


def fit_dlog(t: F64, v: F64, w: F64, iters: int = 40) -> tuple[F64, F64]:
    n = v.shape[0]
    vv = np.where(w > 0, v, np.nan)
    k = np.nanargmax(np.nan_to_num(vv, nan=-9), axis=1)
    tk = t[k]
    p = np.stack(
        [
            np.nanmin(vv, axis=1),
            np.nanmax(vv, axis=1),
            np.full(n, 0.08),
            tk - 30,
            np.full(n, 0.08),
            tk + 30,
        ],
        axis=1,
    )
    lo = np.array([0.0, 0.15, 0.02, t[0] - 30, 0.02, t[0]])
    hi = np.array([0.6, 1.0, 0.2, t[-1], 0.2, t[-1] + 30])
    p = np.clip(p, lo, hi)
    lam = np.full(n, 0.01)

    def cost(pp: F64) -> F64:
        return np.sum(w * (dlog(t, pp) - v) ** 2, axis=1)

    c = cost(p)
    eye = np.eye(6)
    for _ in range(iters):
        r = (dlog(t, p) - v) * w
        jac = jacobian(t, p) * w[..., None]
        h = np.einsum("ntp,ntq->npq", jac, jac)
        g = np.einsum("ntp,nt->np", jac, r)
        diag = np.einsum("npp->np", h)[:, :, None] * eye
        step = np.linalg.solve(
            h + lam[:, None, None] * (diag + 1e-09 * eye), -g[..., None]
        )
        pn = np.clip(p + step[..., 0], lo, hi)
        pn[:, 5] = np.maximum(pn[:, 5], pn[:, 3] + 10)
        cn = cost(pn)
        ok = cn < c
        p[ok], c[ok] = (pn[ok], cn[ok])
        lam = np.where(ok, lam / 3, lam * 3).clip(1e-07, 10000000.0)
    rmse = np.sqrt(c / np.maximum(w.sum(axis=1), 1))
    return (p, rmse)


def season_metrics(p: F64, t0: F64, t1: float, alpha: float) -> dict[str, F64]:
    days = np.arange(float(np.min(t0)), t1 + 1.0)
    f = dlog(days, p)
    inside = days[None, :] >= t0[:, None]
    f = np.where(inside, f, np.nan)
    first = np.argmax(inside, axis=1)
    base = np.minimum(f[np.arange(len(f)), first], f[:, -1])
    peak = np.nanmax(f, axis=1)
    thr = base + alpha * (peak - base)
    above = f >= thr[:, None]
    sos = days[np.argmax(above, axis=1)]
    eos = days[len(days) - 1 - np.argmax(above[:, ::-1], axis=1)]
    pos = days[np.nanargmax(f, axis=1)]
    return {"sos": sos, "pos": pos, "eos": eos, "peak": peak, "amp": peak - base}


PHENO_BANDS = [
    "w_sos",
    "w_pos",
    "w_eos",
    "w_peak",
    "w_amp",
    "w_rmse",
    "s_sos",
    "s_pos",
    "s_eos",
    "s_peak",
    "s_amp",
    "s_rmse",
    "ndvi_min",
    "ndvi_max",
]


def phenology_year(ndvi: F32, k: int, alpha: float) -> tuple[I16, F64]:
    _, hgt, wid = ndvi.shape
    flat = ndvi.reshape(12, -1).T.astype(np.float64)
    out = np.full((len(PHENO_BANDS), flat.shape[0]), NODATA, dtype=np.int16)
    hrmse = np.full(flat.shape[0], np.nan)
    valid = np.isfinite(flat).sum(axis=1) >= 8
    idx = np.nonzero(valid)[0]
    tw, ts = (DOY[:6], DOY[4:])
    for sl in chunks(len(idx), 60000):
        ii = idx[sl]
        v, hr = harmonic_fill(flat[ii], k)
        hrmse[ii] = hr
        pw, rw = fit_dlog(tw, v[:, :6], np.ones((len(ii), 6)))
        mw = season_metrics(pw, np.full(len(ii), tw[0]), tw[-1], alpha)
        w_ok = (mw["peak"] >= CFG["season_peak_min"]) & (
            mw["amp"] >= CFG["season_amp_min"]
        )
        wts = np.ones((len(ii), 8))
        wts[w_ok, 0] = 0.0
        ps, rs = fit_dlog(ts, v[:, 4:], wts)
        ms = season_metrics(ps, np.where(w_ok, ts[1], ts[0]), ts[-1], alpha)
        cols = [
            mw["sos"],
            mw["pos"],
            mw["eos"],
            mw["peak"] * 10000.0,
            mw["amp"] * 10000.0,
            rw * 10000.0,
            ms["sos"],
            ms["pos"],
            ms["eos"],
            ms["peak"] * 10000.0,
            ms["amp"] * 10000.0,
            rs * 10000.0,
            np.min(v, axis=1) * 10000.0,
            np.max(v, axis=1) * 10000.0,
        ]
        for b, col in enumerate(cols):
            out[b, ii] = np.round(np.nan_to_num(col, nan=NODATA)).clip(-32767, 32767)
    return (out.reshape(len(PHENO_BANDS), hgt, wid), hrmse)


def step_phenology(src: Path, dst: Path) -> None:
    rows = []
    for y in YEARS:
        t = time.time()
        s2, prof = read_stack(src / f"aoi_s2_ndvi_evi_{y}.tif", 10000.0)
        ndvi = s2[:12]
        rng = np.random.default_rng(CFG["seed"] + y)
        flat = ndvi.reshape(12, -1).T.astype(np.float64)
        samp = flat[
            rng.choice(flat.shape[0], size=min(50000, flat.shape[0]), replace=False)
        ]
        samp = samp[np.isfinite(samp).all(axis=1)]
        for k in (2, 3):
            rows.append(
                {
                    "year": y,
                    "K": k,
                    "harmonic_rmse_median": round(
                        float(np.median(harmonic_fill(samp, k)[1])), 4
                    ),
                }
            )
        ph, _ = phenology_year(ndvi, CFG["harmonics"], CFG["alpha"])
        p = {**prof, "dtype": "int16"}
        write_tif(dst / f"phenometrics_{y}.tif", ph, p, PHENO_BANDS, NODATA)
        log(f"phenology {y}: {time.time() - t:.0f} s")
    write_csv(dst / "tables_harmonic_rmse.csv", rows)


def rule_labels(ph: I16, crop: npt.NDArray[np.bool_]) -> U8:
    g = {n: ph[i].astype(np.float64) for i, n in enumerate(PHENO_BANDS)}
    nod = ph[0] == NODATA
    for n in PHENO_BANDS:
        if n not in ("w_sos", "w_pos", "w_eos", "s_sos", "s_pos", "s_eos"):
            g[n] = g[n] / 10000.0
    w_ok = (g["w_peak"] >= CFG["season_peak_min"]) & (
        g["w_amp"] >= CFG["season_amp_min"]
    )
    s_ok = (g["s_peak"] >= CFG["season_peak_min"]) & (
        g["s_amp"] >= CFG["season_amp_min"]
    )
    good = (g["w_rmse"] <= CFG["fit_rmse_max"]) & (g["s_rmse"] <= CFG["fit_rmse_max"])
    inr = lambda x, r: (x >= r[0]) & (x <= r[1])
    wheat_t = inr(g["w_pos"], CFG["wheat_pos"])
    late = inr(g["s_pos"], CFG["summer_pos_late"])
    rules = {
        1: w_ok & wheat_t & ~s_ok,
        2: w_ok & wheat_t & s_ok & late,
        3: ~w_ok & s_ok & late & inr(g["s_sos"], CFG["cotton_sos"]),
        4: ~w_ok & s_ok & inr(g["s_pos"], CFG["maize_main_pos"]),
        5: ~w_ok & ~s_ok & (g["ndvi_max"] < CFG["fallow_max"]),
        6: ~w_ok & ~s_ok & (g["ndvi_min"] >= CFG["perennial_min"]),
    }
    n_true = sum((r.astype(np.int8) for r in rules.values()))
    lab = np.zeros(ph.shape[1:], dtype=np.uint8)
    for c, r in rules.items():
        lab[r & (n_true == 1)] = c
    lab[~good | nod | ~crop] = 0
    return lab


def cropland_mask(src: Path, prof: dict[str, Any], dst: Path) -> npt.NDArray[np.bool_]:
    wc = align(src / "aoi_worldcover_2021.tif", prof, bands=1)
    if wc is not None:
        mask = wc[0] == 40
        log(f"cropland mask from WorldCover: {mask.mean() * 100:.1f} % of the window")
    else:
        greens = np.zeros((prof["height"], prof["width"]), dtype=np.int16)
        for y in YEARS:
            with rasterio.open(dst / f"phenometrics_{y}.tif") as ds:
                greens += ds.read(PHENO_BANDS.index("ndvi_max") + 1) >= 4500
        mask = greens >= 2
        log(f"cropland mask from NDVI (no WorldCover file): {mask.mean() * 100:.1f} %")
    write_tif(
        dst / "cropland_mask.tif",
        mask.astype(np.uint8),
        {**prof, "dtype": "uint8"},
        ["cropland"],
        255,
    )
    return mask


def step_labels(src: Path, dst: Path) -> None:
    with rasterio.open(dst / f"phenometrics_{YEARS[0]}.tif") as ds:
        prof = {**ds.profile, "crs": EPSG or ds.crs}
    crop = cropland_mask(src, prof, dst)
    rows = []
    for y in YEARS:
        with rasterio.open(dst / f"phenometrics_{y}.tif") as ds:
            ph = ds.read()
        lab = rule_labels(ph, crop)
        write_tif(
            dst / f"labels_{y}.tif", lab, {**prof, "dtype": "uint8"}, ["label"], 255
        )
        for c, name in CLASSES.items():
            rows.append({"year": y, "class": name, "pixels": int((lab == c).sum())})
        rows.append(
            {
                "year": y,
                "class": "unlabelled cropland",
                "pixels": int(((lab == 0) & crop).sum()),
            }
        )
        log(
            f"labels {y}: "
            + ", ".join((f"{c}={int((lab == c).sum())}" for c in CLASSES))
        )
    write_csv(dst / "tables_label_counts.csv", rows)
    wcer = align(src / "aoi_worldcereal_2021.tif", prof)
    if wcer is not None:
        with rasterio.open(dst / "labels_2021.tif") as ds:
            lab = ds.read(1)
        wheat_wc, maize_m, maize_s = (wcer[0] > 50, wcer[1] > 50, wcer[2] > 50)
        agree = []
        for c, ref in (
            (1, wheat_wc),
            (2, wheat_wc & maize_s),
            (3, ~wheat_wc & ~maize_m),
            (4, maize_m),
        ):
            sel = lab == c
            agree.append(
                {
                    "class": CLASSES[c],
                    "labels": int(sel.sum()),
                    "agree_pct": round(
                        100 * float(ref[sel].mean()) if sel.any() else 0, 1
                    ),
                }
            )
        write_csv(dst / "tables_worldcereal_agreement_2021.csv", agree)


def features(src: Path, y: int, with_s1: bool) -> tuple[F32, dict[str, Any]]:
    s2, prof = read_stack(src / f"aoi_s2_ndvi_evi_{y}.tif", 10000.0)
    feats = [s2[:12], s2[12:]]
    if with_s1:
        s1, _ = read_stack(src / f"aoi_s1_vv_vh_{y}.tif", 100.0)
        vv, vh = (10 ** (s1[:12] / 10), 10 ** (s1[12:] / 10))
        feats.append(4 * vh / (vv + vh))
    f = np.concatenate(feats).astype(np.float32)
    for g in range(0, f.shape[0], 12):
        blk = f[g : g + 12]
        m = np.nanmean(blk, axis=0)
        f[g : g + 12] = np.where(np.isfinite(blk), blk, m)
    return (np.nan_to_num(f, nan=0.0), prof)


def sample_pixels(dst: Path, prof: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(CFG["seed"])
    rows_, cols_ = np.mgrid[0 : prof["height"], 0 : prof["width"]]
    block = rows_ // CFG["block_px"] * 1000 + cols_ // CFG["block_px"]
    ublocks = np.unique(block)
    test_blocks = set(
        rng.choice(
            ublocks, size=int(len(ublocks) * CFG["test_share"]), replace=False
        ).tolist()
    )
    is_test = np.isin(block, list(test_blocks))
    out = []
    for y in YEARS:
        with rasterio.open(dst / f"labels_{y}.tif") as ds:
            lab = ds.read(1)
        for c in CLASSES:
            for split, sel in (("train", ~is_test), ("test", is_test)):
                ii = np.flatnonzero((lab == c) & sel)
                cap = (
                    CFG["per_class_year"]
                    if split == "train"
                    else CFG["per_class_year"] // 2
                )
                if len(ii) > cap:
                    ii = rng.choice(ii, size=cap, replace=False)
                out.append(
                    {
                        "year": y,
                        "cls": c,
                        "split": split,
                        "idx": ii,
                        "block": block.ravel()[ii],
                    }
                )
    return out


def gather(
    src: Path, samples: list[dict[str, Any]], with_s1: bool
) -> dict[str, F32 | npt.NDArray[Any]]:
    xs: dict[str, list[F32]] = {"train": [], "test": []}
    ys: dict[str, list[npt.NDArray[Any]]] = {"train": [], "test": []}
    gs: dict[str, list[npt.NDArray[Any]]] = {"train": [], "test": []}
    for y in YEARS:
        f, _ = features(src, y, with_s1)
        flat = f.reshape(f.shape[0], -1)
        for s in samples:
            if s["year"] != y or len(s["idx"]) == 0:
                continue
            xs[s["split"]].append(flat[:, s["idx"]].T)
            ys[s["split"]].append(np.full(len(s["idx"]), s["cls"]))
            gs[s["split"]].append(s["block"])
    return {
        f"{k}_{sp}": np.concatenate(v[sp])
        for k, v in (("x", xs), ("y", ys), ("g", gs))
        for sp in ("train", "test")
    }


def metrics(
    y_true: npt.NDArray[Any], y_pred: npt.NDArray[Any], model: str
) -> tuple[dict[str, Any], list[dict[str, Any]], npt.NDArray[Any]]:
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
    )

    labels = [c for c in CLASSES if (y_true == c).any()]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    summary = {
        "model": model,
        "OA": round(accuracy_score(y_true, y_pred), 4),
        "kappa": round(cohen_kappa_score(y_true, y_pred), 4),
        "macro_F1": round(f1_score(y_true, y_pred, average="macro", labels=labels), 4),
        "n_test": len(y_true),
    }
    per = []
    for i, c in enumerate(labels):
        pa = cm[i, i] / max(cm[i].sum(), 1)
        ua = cm[i, i] / max(cm[:, i].sum(), 1)
        per.append(
            {
                "model": model,
                "class": CLASSES[c],
                "UA": round(ua, 4),
                "PA": round(pa, 4),
                "F1": round(2 * ua * pa / max(ua + pa, 1e-09), 4),
                "n_test": int(cm[i].sum()),
            }
        )
    return (summary, per, cm)


def train_rf(data: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    best: tuple[float, dict[str, Any]] = (-1.0, {})
    x, y, g = (data["x_train"], data["y_train"], data["g_train"])
    sub = np.random.default_rng(CFG["seed"]).permutation(len(y))[: CFG["cv_max"]]
    x, y, g = (x[sub], y[sub], g[sub])
    for trees in CFG["rf_trees"]:
        for mf in CFG["rf_max_features"]:
            scores = []
            for tr, va in GroupKFold(n_splits=3).split(x, y, g):
                rf = RandomForestClassifier(
                    n_estimators=trees,
                    max_features=mf,
                    n_jobs=-1,
                    random_state=CFG["seed"],
                    min_samples_leaf=2,
                )
                rf.fit(x[tr], y[tr])
                scores.append(float((rf.predict(x[va]) == y[va]).mean()))
            log(f"  RF trees={trees} max_features={mf}: CV OA {np.mean(scores):.4f}")
            if np.mean(scores) > best[0]:
                best = (
                    float(np.mean(scores)),
                    {"n_estimators": trees, "max_features": mf},
                )
    rf = RandomForestClassifier(
        **best[1], n_jobs=-1, random_state=CFG["seed"], min_samples_leaf=2
    )
    rf.fit(data["x_train"], data["y_train"])
    return (rf, {**best[1], "cv_OA": round(best[0], 4)})


class TempCNNWrapper:
    def __init__(self, n_ch: int, n_cls: int, epochs: int = 30) -> None:
        import torch
        from torch import nn

        torch.manual_seed(CFG["seed"])
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        self.torch, self.n_ch, self.epochs = (torch, n_ch, epochs)
        layers: list[Any] = []
        c_in = n_ch
        for _ in range(3):
            layers += [
                nn.Conv1d(c_in, 64, 5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
            c_in = 64
        self.net = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(64 * 12, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_cls),
        )
        self.mu = np.zeros(n_ch, dtype=np.float32)
        self.sd = np.ones(n_ch, dtype=np.float32)

    def _prep(self, x: F32) -> Any:
        a = x.reshape(len(x), self.n_ch, 12)
        a = (a - self.mu[None, :, None]) / self.sd[None, :, None]
        return self.torch.from_numpy(a.astype(np.float32))

    def fit(self, x: F32, y: npt.NDArray[Any]) -> None:
        torch = self.torch
        a = x.reshape(len(x), self.n_ch, 12)
        self.mu, self.sd = (a.mean(axis=(0, 2)), a.std(axis=(0, 2)) + 1e-06)
        xt, yt = (self._prep(x), torch.from_numpy((y - 1).astype(np.int64)))
        opt = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=0.0001)
        lossf = torch.nn.CrossEntropyLoss()
        self.net.train()
        for ep in range(self.epochs):
            perm = torch.randperm(len(xt))
            tot = 0.0
            for b in range(0, len(xt), 512):
                ib = perm[b : b + 512]
                opt.zero_grad()
                loss = lossf(self.net(xt[ib]), yt[ib])
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(ib)
            log(f"  TempCNN epoch {ep + 1}/{self.epochs}: loss {tot / len(xt):.4f}")

    def predict(self, x: F32) -> npt.NDArray[Any]:
        self.net.eval()
        out = []
        with self.torch.no_grad():
            for b in range(0, len(x), 100000):
                out.append(
                    self.net(self._prep(x[b : b + 100000])).argmax(1).numpy() + 1
                )
        return np.concatenate(out)


def step_classify(src: Path, dst: Path, tempcnn: bool) -> None:
    with rasterio.open(dst / f"labels_{YEARS[0]}.tif") as ds:
        prof = {**ds.profile, "crs": EPSG or ds.crs}
    with rasterio.open(dst / "cropland_mask.tif") as ds:
        crop = ds.read(1) == 1
    samples = sample_pixels(dst, prof)
    summary, per, selected, models = ([], [], {}, {})
    for with_s1 in (True, False):
        tag = "S1+S2" if with_s1 else "S2 only"
        data = gather(src, samples, with_s1)
        log(
            f"{tag}: {len(data['y_train'])} training, {len(data['y_test'])} test pixels"
        )
        rf, sel = train_rf(data)
        s, p, cm = metrics(data["y_test"], rf.predict(data["x_test"]), f"RF {tag}")
        summary.append(s)
        per += p
        np.savetxt(
            dst / f"tables_confusion_RF_{tag.replace(' ', '_').replace('+', '')}.csv",
            cm,
            fmt="%d",
            delimiter=",",
        )
        selected[f"RF {tag}"] = sel
        models[f"RF {tag}"] = (rf, with_s1)
        log(f"RF {tag}: OA {s['OA']}, macro-F1 {s['macro_F1']}")
        if tempcnn:
            n_ch = 3 if with_s1 else 2
            net = TempCNNWrapper(n_ch, max(CLASSES))
            net.fit(data["x_train"], data["y_train"])
            s, p, cm = metrics(
                data["y_test"], net.predict(data["x_test"]), f"TempCNN {tag}"
            )
            summary.append(s)
            per += p
            np.savetxt(
                dst
                / f"tables_confusion_TempCNN_{tag.replace(' ', '_').replace('+', '')}.csv",
                cm,
                fmt="%d",
                delimiter=",",
            )
            selected[f"TempCNN {tag}"] = {
                "epochs": net.epochs,
                "filters": 64,
                "kernel": 5,
                "dropout": 0.2,
                "lr": 0.001,
                "batch": 512,
            }
            models[f"TempCNN {tag}"] = (net, with_s1)
            log(f"TempCNN {tag}: OA {s['OA']}, macro-F1 {s['macro_F1']}")
    write_csv(dst / "tables_accuracy_summary.csv", summary)
    write_csv(dst / "tables_accuracy_per_class.csv", per)
    best = max(summary, key=lambda r: r["macro_F1"])["model"]
    (dst / "params_selected.json").write_text(
        json.dumps(
            {
                "config": {k: v for k, v in CFG.items()},
                "selected": selected,
                "map_model": best,
            },
            indent=2,
            default=str,
        )
    )
    log(f"mapping with {best}")
    model, with_s1 = models[best]
    for y in YEARS:
        f, _ = features(src, y, with_s1)
        flat = f.reshape(f.shape[0], -1).T
        pred = np.zeros(flat.shape[0], dtype=np.uint8)
        ii = np.flatnonzero(crop.ravel())
        for sl in chunks(len(ii), 400000):
            pred[ii[sl]] = model.predict(flat[ii[sl]])
        write_tif(
            dst / f"cropmap_{y}.tif",
            pred.reshape(crop.shape),
            {**prof, "dtype": "uint8"},
            ["class"],
            0,
        )
        log(f"cropmap {y} written")


def step_tempcnn(src: Path, dst: Path) -> None:
    with rasterio.open(dst / f"labels_{YEARS[0]}.tif") as ds:
        prof = {**ds.profile, "crs": EPSG or ds.crs}
    samples = sample_pixels(dst, prof)
    summary, per, selected = ([], [], {})
    for with_s1 in (True, False):
        tag = "S1+S2" if with_s1 else "S2 only"
        data = gather(src, samples, with_s1)
        log(f"TempCNN {tag}: {len(data['y_train'])} training pixels")
        net = TempCNNWrapper(3 if with_s1 else 2, max(CLASSES))
        net.fit(data["x_train"], data["y_train"])
        sm, p, cm = metrics(
            data["y_test"], net.predict(data["x_test"]), f"TempCNN {tag}"
        )
        summary.append(sm)
        per += p
        np.savetxt(
            dst
            / f"tables_confusion_TempCNN_{tag.replace(' ', '_').replace('+', '')}.csv",
            cm,
            fmt="%d",
            delimiter=",",
        )
        selected[f"TempCNN {tag}"] = {
            "epochs": net.epochs,
            "filters": 64,
            "kernel": 5,
            "dropout": 0.2,
            "lr": 0.001,
            "batch": 512,
        }
        log(f"TempCNN {tag}: OA {sm['OA']}, macro-F1 {sm['macro_F1']}")
    write_csv(dst / "tables_accuracy_tempcnn_summary.csv", summary)
    write_csv(dst / "tables_accuracy_tempcnn_per_class.csv", per)
    (dst / "params_tempcnn.json").write_text(json.dumps(selected, indent=2))


def step_summarise(src: Path, dst: Path) -> None:
    maps = {}
    for y in YEARS:
        with rasterio.open(dst / f"cropmap_{y}.tif") as ds:
            maps[y] = ds.read(1)
            prof = {**ds.profile, "crs": EPSG or ds.crs}
    area, trans, pheno, prof_rows = ([], [], [], [])
    px_ha = abs(prof["transform"].a * prof["transform"].e) / 10000.0
    for y in YEARS:
        m = maps[y]
        for c, name in CLASSES.items():
            area.append(
                {
                    "year": y,
                    "class": name,
                    "pixels": int((m == c).sum()),
                    "area_ha": round(float((m == c).sum()) * px_ha, 1),
                }
            )
        if y + 1 in maps:
            n = maps[y + 1]
            for a, name_a in CLASSES.items():
                for b, name_b in CLASSES.items():
                    trans.append(
                        {
                            "from_year": y,
                            "to_year": y + 1,
                            "from": name_a,
                            "to": name_b,
                            "pixels": int(((m == a) & (n == b)).sum()),
                        }
                    )
        with rasterio.open(dst / f"phenometrics_{y}.tif") as ds:
            ph = ds.read()
        ok = {}
        for sea in ("w", "s"):
            pk = ph[PHENO_BANDS.index(f"{sea}_peak")] / 10000.0
            am = ph[PHENO_BANDS.index(f"{sea}_amp")] / 10000.0
            ok[sea] = (pk >= CFG["season_peak_min"]) & (am >= CFG["season_amp_min"])
        for c, name in CLASSES.items():
            sel = m == c
            if not sel.any():
                continue
            row: dict[str, Any] = {"year": y, "class": name}
            for b, bn in enumerate(PHENO_BANDS):
                keep = sel & ok[bn[0]] if bn[:2] in ("w_", "s_") else sel
                v = ph[b][keep].astype(np.float64)
                v = v[v != NODATA]
                row[f"{bn}_n"] = len(v)
                if bn not in ("w_sos", "w_pos", "w_eos", "s_sos", "s_pos", "s_eos"):
                    v = v / 10000.0
                if len(v):
                    q = np.percentile(v, [25, 50, 75])
                    row.update(
                        {
                            f"{bn}_q25": round(q[0], 4),
                            f"{bn}_median": round(q[1], 4),
                            f"{bn}_q75": round(q[2], 4),
                        }
                    )
            pheno.append(row)
        s2, _ = read_stack(src / f"aoi_s2_ndvi_evi_{y}.tif", 10000.0)
        s1, _ = read_stack(src / f"aoi_s1_vv_vh_{y}.tif", 100.0)
        for c, name in CLASSES.items():
            sel = m == c
            if not sel.any():
                continue
            for var, arr in (
                ("NDVI", s2[:12]),
                ("EVI", s2[12:]),
                ("VV", s1[:12]),
                ("VH", s1[12:]),
            ):
                vals = arr[:, sel]
                for stat, f in (("mean", np.nanmean), ("sd", np.nanstd)):
                    prof_rows.append(
                        {
                            "year": y,
                            "class": name,
                            "variable": var,
                            "stat": stat,
                            **{
                                str(i + 1): round(float(x), 4)
                                for i, x in enumerate(f(vals, axis=1))
                            },
                        }
                    )
        h3, w3 = (m.shape[0] // 3, m.shape[1] // 3)
        blk = (
            m[: h3 * 3, : w3 * 3]
            .reshape(h3, 3, w3, 3)
            .transpose(0, 2, 1, 3)
            .reshape(h3, w3, 9)
        )
        counts = np.stack([(blk == c).sum(axis=2) for c in range(7)], axis=2)
        ql = counts.argmax(axis=2).astype(np.uint8)
        t = prof["transform"]
        qprof = {
            **prof,
            "height": h3,
            "width": w3,
            "dtype": "uint8",
            "transform": rasterio.Affine(t.a * 3, t.b, t.c, t.d, t.e * 3, t.f),
        }
        write_tif(dst / f"quicklook_cropmap_90m_{y}.tif", ql, qprof, ["class"], 0)
    write_csv(dst / "tables_area_by_class_year.csv", area)
    write_csv(dst / "tables_transitions.csv", trans)
    write_csv(dst / "tables_phenometrics_by_class_year.csv", pheno)
    write_csv(dst / "tables_profiles_by_class_year.csv", prof_rows)
    log("summaries written")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    here = Path(__file__).resolve().parent.parent / "GEE_data"
    ap.add_argument("--data", type=Path, default=here)
    ap.add_argument(
        "--steps", nargs="+", default=["phenology", "labels", "classify", "summarise"]
    )
    ap.add_argument("--tempcnn", action="store_true")
    ap.add_argument(
        "--block-px",
        type=int,
        default=CFG["block_px"],
        help="spatial hold-out block size in pixels (default 100 = 3 km)",
    )
    ap.add_argument("--per-class-year", type=int, default=CFG["per_class_year"])
    a = ap.parse_args()
    CFG["block_px"], CFG["per_class_year"] = (a.block_px, a.per_class_year)
    dst = a.data / "pipeline_out"
    dst.mkdir(exist_ok=True)
    t0 = time.time()
    if "phenology" in a.steps:
        step_phenology(a.data, dst)
    if "labels" in a.steps:
        step_labels(a.data, dst)
    if "classify" in a.steps:
        step_classify(a.data, dst, a.tempcnn)
    if "tempcnn" in a.steps:
        step_tempcnn(a.data, dst)
    if "summarise" in a.steps:
        step_summarise(a.data, dst)
    log(f"done in {(time.time() - t0) / 60:.1f} min; outputs in {dst}")


if __name__ == "__main__":
    main()
