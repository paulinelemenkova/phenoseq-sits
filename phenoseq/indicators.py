#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, theilslopes

D = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
P, OUT = (D / "pipeline_out", D / "results")
OUT.mkdir(exist_ok=True)
YEARS = list(range(2018, 2026))
CROPS = [
    "winter wheat",
    "wheat + second-crop maize",
    "cotton",
    "main-crop maize",
    "fallow",
    "perennial",
]
area = pd.read_csv(P / "tables_area_by_class_year.csv")
wide = area.pivot(index="year", columns="class", values="area_ha")[CROPS]
crop_cols = [c for c in CROPS if c != "fallow"]
tot = wide.sum(axis=1)
share = wide.div(tot, axis=0) * 100
share["double_crop_pct_of_cropped"] = (
    wide["wheat + second-crop maize"] / wide[crop_cols].sum(axis=1) * 100
)
harvests = wide[crop_cols].sum(axis=1) + wide["wheat + second-crop maize"]
share["cropping_intensity"] = harvests / tot
share["total_ha"] = tot
share["cotton_ha"] = wide["cotton"]
share["second_maize_ha"] = wide["wheat + second-crop maize"]
share["wheat_ha_all"] = wide["winter wheat"] + wide["wheat + second-crop maize"]
share["fallow_ha"] = wide["fallow"]
share.round(3).to_csv(OUT / "area_share.csv")
ph = pd.read_csv(P / "tables_phenometrics_by_class_year.csv")
series = {
    ("wheat (all)", "w_sos"): ["winter wheat", "wheat + second-crop maize"],
    ("wheat (all)", "w_pos"): ["winter wheat", "wheat + second-crop maize"],
    ("wheat (all)", "w_eos"): ["winter wheat", "wheat + second-crop maize"],
    ("cotton", "s_sos"): ["cotton"],
    ("cotton", "s_pos"): ["cotton"],
    ("cotton", "s_eos"): ["cotton"],
    ("second-crop maize", "s_sos"): ["wheat + second-crop maize"],
    ("second-crop maize", "s_pos"): ["wheat + second-crop maize"],
    ("second-crop maize", "s_eos"): ["wheat + second-crop maize"],
    ("cotton", "s_peak"): ["cotton"],
}
rows, tests = ([], [])
for (crop, var), classes in series.items():
    vals = []
    for y in YEARS:
        sub = ph[(ph.year == y) & ph["class"].isin(classes)]
        w = sub[f"{var}_n"].to_numpy(float)
        q = {
            k: float(np.average(sub[f"{var}_{k}"], weights=w))
            for k in ("q25", "median", "q75")
        }
        rows.append(
            {
                "crop": crop,
                "metric": var[2:],
                "year": y,
                **{k: round(v, 3) for k, v in q.items()},
                "n": int(w.sum()),
            }
        )
        vals.append(q["median"])
    v = np.array(vals)
    sl, ic, lo, hi = theilslopes(v, YEARS)
    tau, p = kendalltau(YEARS, v)
    tests.append(
        {
            "crop": crop,
            "metric": var[2:],
            "sen_slope_per_yr": round(sl, 3),
            "slope_lo95": round(lo, 3),
            "slope_hi95": round(hi, 3),
            "kendall_tau": round(tau, 3),
            "p": round(p, 3),
            "mean": round(float(v.mean()), 3),
            "min": round(float(v.min()), 3),
            "max": round(float(v.max()), 3),
        }
    )
pd.DataFrame(rows).to_csv(OUT / "phenotrends.csv", index=False)
pd.DataFrame(tests).to_csv(OUT / "trend_tests.csv", index=False)
era = pd.read_csv(D / "aoi_era5_daily_2018_2025.csv", parse_dates=["date"])
era["y"], era["doy"] = (era.date.dt.year, era.date.dt.dayofyear)
TB = 15.6
hrows = []
for y in YEARS:
    e = era[era.y == y]
    ja = e[(e.date.dt.month >= 7) & (e.date.dt.month <= 8)]
    gs = e[(e.doy >= 121) & (e.doy <= 273)]
    tm = (gs.temperature_2m_max + gs.temperature_2m_min) / 2
    row = {
        "year": y,
        "hsd35_julaug": int((ja.temperature_2m_max > 35).sum()),
        "hsd38_julaug": int((ja.temperature_2m_max > 38).sum()),
        "hsd40_julaug": int((ja.temperature_2m_max > 40).sum()),
        "tmax_julaug": round(float(ja.temperature_2m_max.mean()), 2),
        "gdd_may_sep": round(float(np.clip(tm - TB, 0, None).sum()), 0),
        "p_annual": round(float(e.precipitation_sum.sum()), 1),
        "p_novapr": round(
            float(
                era[
                    (era.y == y - 1) & (era.date.dt.month >= 11)
                    | (era.y == y) & (era.date.dt.month <= 4)
                ].precipitation_sum.sum()
            ),
            1,
        )
        if y > 2018
        else np.nan,
        "et0_annual": round(float(e.et0_fao_evapotranspiration.sum()), 0),
    }
    hrows.append(row)
heat = pd.DataFrame(hrows)
pt = pd.DataFrame(rows)
for crop, met, col in (
    ("cotton", "pos", "cotton_pos"),
    ("cotton", "sos", "cotton_sos"),
    ("cotton", "eos", "cotton_eos"),
    ("cotton", "peak", "cotton_peak"),
    ("second-crop maize", "pos", "maize2_pos"),
    ("wheat (all)", "pos", "wheat_pos"),
):
    heat[col] = (
        pt[(pt.crop == crop) & (pt.metric == met)]
        .set_index("year")["median"]
        .reindex(YEARS)
        .to_numpy()
    )
heat["cotton_share_pct"] = share["cotton"].reindex(YEARS).to_numpy()
heat["double_pct"] = share["double_crop_pct_of_cropped"].reindex(YEARS).to_numpy()
heat["fallow_pct"] = share["fallow"].reindex(YEARS).to_numpy()
heat.to_csv(OUT / "heat.csv", index=False)
crow = []
for x in (
    "hsd35_julaug",
    "hsd38_julaug",
    "hsd40_julaug",
    "tmax_julaug",
    "gdd_may_sep",
    "p_novapr",
):
    for yv in (
        "cotton_pos",
        "cotton_sos",
        "cotton_eos",
        "cotton_peak",
        "maize2_pos",
        "cotton_share_pct",
        "double_pct",
        "fallow_pct",
        "wheat_pos",
    ):
        ok = heat[[x, yv]].dropna()
        r, p = spearmanr(ok[x], ok[yv])
        crow.append(
            {
                "x": x,
                "y": yv,
                "spearman_r": round(float(r), 3),
                "p": round(float(p), 3),
                "n": len(ok),
            }
        )
pd.DataFrame(crow).to_csv(OUT / "heat_corr.csv", index=False)
tr = pd.read_csv(P / "tables_transitions.csv")
m = (
    tr.groupby(["from", "to"])
    .pixels.sum()
    .unstack()
    .reindex(index=CROPS, columns=CROPS)
)
(m.div(m.sum(axis=1), axis=0) * 100).round(2).to_csv(OUT / "transitions_mean.csv")
KC = {"wheat": (0.4, 1.15, 0.3), "cotton": (0.35, 1.2, 0.6), "maize": (0.3, 1.2, 0.35)}
EFF = 0.55


def kc_curve(
    sos: float, pos: float, eos: float, crop: str
) -> tuple[np.ndarray, np.ndarray]:
    ini, mid, end = KC[crop]
    d0, d_end = (max(1.0, sos - 25), min(365.0, eos + 10))
    mid_end = pos + (eos - pos) / 2
    days = np.arange(d0, d_end + 1)
    kc = np.interp(days, [d0, sos, pos, mid_end, d_end], [ini, ini, mid, mid, end])
    return (days, kc)


def pe_monthly(p: np.ndarray) -> np.ndarray:
    return np.where(p <= 250, p * (125 - 0.2 * p) / 125, 125 + 0.1 * p)


wrows = []
for y in YEARS:
    e = era[era.y == y].set_index("doy")
    pm = e.groupby(e.date.dt.month).precipitation_sum.sum()
    ndays = e.groupby(e.date.dt.month).size()
    pe_per_day = dict(zip(pm.index, pe_monthly(pm.to_numpy()) / ndays.to_numpy()))
    pe_day = e.date.dt.month.map(pe_per_day)

    def season(crop: str, cls: str, s: str) -> tuple[float, float, float]:
        sub = ph[(ph.year == y) & (ph["class"] == cls)].iloc[0]
        days, kc = kc_curve(
            sub[f"{s}_sos_median"], sub[f"{s}_pos_median"], sub[f"{s}_eos_median"], crop
        )
        etc = float((kc * e.loc[days, "et0_fao_evapotranspiration"].to_numpy()).sum())
        pe = float(pe_day.loc[days].sum())
        return (etc, pe, max(etc - pe, 0.0))

    plan = {
        "winter wheat": [("wheat", "winter wheat", "w")],
        "wheat + second-crop maize": [
            ("wheat", "wheat + second-crop maize", "w"),
            ("maize", "wheat + second-crop maize", "s"),
        ],
        "cotton": [("cotton", "cotton", "s")],
        "main-crop maize": [("maize", "main-crop maize", "s")],
    }
    for cls, seasons in plan.items():
        etc = pe = nir = 0.0
        for crop, c2, s in seasons:
            a, b, c = season(crop, c2, s)
            etc, pe, nir = (etc + a, pe + b, nir + c)
        ha = float(wide.loc[y, cls])
        wrows.append(
            {
                "year": y,
                "class": cls,
                "etc_mm": round(etc, 0),
                "pe_mm": round(pe, 0),
                "nir_mm": round(nir, 0),
                "area_ha": round(ha, 0),
                "nir_hm3": round(nir * ha * 10 / 1000000.0, 1),
                "gross_hm3": round(nir * ha * 10 / 1000000.0 / EFF, 1),
            }
        )
pd.DataFrame(wrows).to_csv(OUT / "water.csv", index=False)
print("done")
