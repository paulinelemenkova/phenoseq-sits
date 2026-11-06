# PhenoSeq-SITS: Phenometric Inversion of Latent Multi-Cropping Sequences

---

## Overview

PhenoSeq-SITS is an end-to-end, label-free crop-sequence inference framework. It runs on dense optical (MSI L2A) and C-band SAR (IW GRD, VV/VH) time series. It needs no in-situ annotation.

The framework factorises each pixel-level spectro-temporal trajectory into a set of parametric growing-season kernels:

- **Season kernels.** Each kernel is a double sigmoid fitted by bounded Levenberg–Marquardt (LM) optimisation. The fit is fully vectorised over N × T tensors.
- **Phenometrics.** SOS, POS and EOS are obtained by amplitude-threshold (α) inversion of each fitted kernel.
- **Pseudo-labels.** The phenometrics drive a deterministic rule-induction operator. It emits high-confidence pseudo-labels (PL) under a mutual-exclusivity constraint and rejects ambiguous fits.
- **Classification.** The PL distil the phenological priors into a Random Forest (RF) ensemble. Generalisation is quantified by spatially-blocked GroupKFold cross-validation, which suppresses spatial-autocorrelation leakage. An optional 1D temporal CNN (TempCNN) branch provides a deep-learning (DL) counterpart.
- **Downstream analytics:**
  - non-parametric trend estimation: Theil–Sen β with Mann–Kendall (MK) τ;
  - rank-correlation coupling with reanalysis thermal forcing: growing degree days (GDD), heat-stress-day (HSD) counts, Spearman ρ;
  - first-order Markov transition operators between annual label fields;
  - crop-coefficient evapotranspiration (ET) accounting: FAO-56 K_c–ET_c, with USDA-SCS effective precipitation P_eff;
  - cropping-intensity (CI) metrics.

## Architecture

```
gee/                       server-side SITS mosaicking (GEE, JavaScript)
  mosaic_sits.js           CS+ CDF masking -> monthly median tensors (NDVI, EVI | σ⁰VV, σ⁰VH) -> GeoTIFF
  ancillary_grid.js        co-registered auxiliary layers on the export lattice
phenoseq/                  client-side inference (Python)
  qc_stack.py              raster integrity diagnostics, band-wise validity statistics, block-mean decimation
  pipeline.py              harmonic gap filling -> DL-LM inversion -> PL induction -> RF / TempCNN -> summaries
  class_signatures.py      class-conditional temporal signatures on a full-resolution subset
  indicators.py            trend statistics, thermal coupling, transition operators, FAO-56 water balance
```

## Methodological core

| Stage | Operator | Key hyper-parameters |
|---|---|---|
| Cloud/shadow suppression | Cloud Score+ cumulative distribution (cs_cdf) | τ_CS = 0.60 |
| SAR radiometry | dB → linear power, median, → dB; border-noise rejection | σ⁰ < −30 dB masked |
| Temporal regularisation | Fourier-basis least squares | K = 3 harmonics, T = 365 d |
| Season kernel | double logistic v(t) = v_min + Δv[σ(m₁(t−n₁)) − σ(m₂(t−n₂))] | box constraints, LM damping λ ∈ [1e−7, 1e7] |
| Phenometric inversion | amplitude-threshold crossing | α = 0.5 |
| Season acceptance | peak / amplitude gating | NDVI_peak ≥ 0.5, Δv ≥ 0.3, RMSE ≤ 0.08 |
| PL induction | exclusive rule set over (season count, POS, SOS) | 6 classes, ambiguity rejection |
| Supervision | RF (Gini), grid search | n_trees ∈ {200, 500}, m_try ∈ {√p, p/3} |
| Validation | spatially-blocked GroupKFold | 100 px blocks, 30 % hold-out |
| SAR descriptor | radar vegetation index RVI = 4σ⁰VH / (σ⁰VV + σ⁰VH) | linear domain |
| DL branch (optional) | TempCNN, 1D conv along t | 64 filters, k = 5, Adam 1e−3, 30 epochs |

## Execution

```bash
# 1  server side: run gee/mosaic_sits.js in the GEE Code Editor, export the yearly stacks to GEE_data/
# 2  integrity diagnostics
python3 phenoseq/qc_stack.py GEE_data
# 3  full inference graph
python3 phenoseq/pipeline.py --data GEE_data
python3 phenoseq/pipeline.py --steps classify summarise      # partial re-execution
python3 phenoseq/pipeline.py --steps tempcnn                 # DL branch (requires torch)
# 4  downstream analytics
python3 phenoseq/indicators.py data
```

**Input schema.** The inputs are `aoi_s2_ndvi_evi_YYYY.tif` (24 bands, scaled ×10⁴) and `aoi_s1_vv_vh_YYYY.tif` (24 bands, dB ×10²). Both are on a common UTM lattice at 30 m, with nodata = −32768. Optional inputs are `aoi_worldcover_2021.tif` and `aoi_worldcereal_2021.tif`.

**Output schema.** Outputs are written to `GEE_data/pipeline_out/`:

- per-year phenometric rasters (int16);
- PL rasters and class rasters (uint8);
- confusion tensors and accuracy tables (OA, κ, per-class F1/UA/PA);
- class-by-year area, phenometric and signature tables;
- an inter-annual transition matrix;
- `params_selected.json`.

## Dependencies

Python ≥ 3.10, NumPy, SciPy, pandas, rasterio (bundled PROJ), scikit-learn. PyTorch is needed only for the TempCNN branch. A GEE account is needed for server-side mosaicking.

```bash
pip install -r requirements.txt
```

## Citation

Manuscript under review. Citation details will be added upon publication.
