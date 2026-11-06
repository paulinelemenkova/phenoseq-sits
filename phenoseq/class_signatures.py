#!/usr/bin/env python3
from __future__ import annotations
import csv
import sys
import numpy as np

NODATA = -32768
CLASSES = [
    "winter wheat",
    "wheat + second-crop maize",
    "cotton",
    "main-crop maize",
    "fallow",
]


def main(src: str, dst: str) -> None:
    d = np.load(src, allow_pickle=True)
    s2 = d["s2_2025_subset30m"].astype(np.float64)
    s1 = d["s1_2025_subset30m"].astype(np.float64)
    s2[s2 == NODATA] = np.nan
    s1[s1 == NODATA] = np.nan
    s2 /= 10000.0
    s1 /= 100.0
    var = {
        "NDVI": s2[:12].reshape(12, -1).T,
        "EVI": s2[12:].reshape(12, -1).T,
        "VV": s1[:12].reshape(12, -1).T,
        "VH": s1[12:].reshape(12, -1).T,
    }
    n = var["NDVI"]
    w = np.nanmax(n[:, 1:4], axis=1)
    s = np.nanmax(n[:, 7:10], axis=1)
    j = n[:, 6]
    m = np.nanargmax(np.nan_to_num(n[:, 4:], nan=-1.0), axis=1) + 5
    lab = np.full(n.shape[0], -1)
    lab[np.nanmax(n, axis=1) < 0.25] = 4
    lab[(w > 0.55) & (s < 0.3)] = 0
    lab[(w > 0.55) & (s > 0.55) & (j < 0.4)] = 1
    lab[(w < 0.3) & (s > 0.55) & (m >= 8) & (m <= 10)] = 2
    lab[(w < 0.3) & (np.nanmax(n[:, 5:7], axis=1) > 0.6) & (m <= 7)] = 3
    with open(dst, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["class", "n", "share_pct", "variable", "stat", *range(1, 13)])
        for k, name in enumerate(CLASSES):
            sel = lab == k
            for vname, arr in var.items():
                for stat, f in (("mean", np.nanmean), ("sd", np.nanstd)):
                    vals = f(arr[sel], axis=0) if sel.any() else np.full(12, np.nan)
                    wr.writerow(
                        [
                            name,
                            int(sel.sum()),
                            round(100 * sel.mean(), 1),
                            vname,
                            stat,
                            *np.round(vals, 4),
                        ]
                    )
    print(
        {c: int((lab == k).sum()) for k, c in enumerate(CLASSES)},
        "unlabelled",
        int((lab == -1).sum()),
        "of",
        lab.size,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
