# ML‑Accelerated Emission Spectrum Tutorial (NEA/TD‑DFT → ML)

A hands‑on, reproducible tutorial showing how to (i) build a **reference emission spectrum** from a rate table via the **nuclear ensemble approach (NEA)**, (ii) generate an **ML‑accelerated spectrum**, and (iii) **quantify accuracy** with RIC and other metrics — using the scripts in `script/` and a tiny example dataset.

> **Who is this for?**
> Researchers modeling fluorescence/phosphorescence spectra who want a lightweight, scriptable workflow to compare TD‑DFT (or any QC) vs ML spectra on the same footing.

---

## TL;DR

```bash
# 0) Install deps
python -m pip install numpy matplotlib

# 1) Put your reference rates at data/emission-rate.dat  (cols: E[eV], lambda[nm], diff_rate)

# 2) Build a reference spectrum (NEA convolution)
python script/emission-rate-TD.py data/emission-rate.dat --delta 0.06 --eps 0.002

# 3) Drop your ML spectrum at train/500/spectrum/emission/emission_spectrum_full.dat
#    (and other training sizes, e.g., 50, 100, 40000, ... as you wish)

# 4) Evaluate RIC vs full reference (TD and ML paths)
bash script/TD-RIC.sh
bash script/ML-RIC.sh

# 5) Plot ML vs REF (energy & wavelength domains)
python script/plot_emission_compare.py \
  --ml-ev train/500/spectrum/emission/emission_spectrum_eV.dat \
  --ref-rate data/emission-rate.dat --delta 0.06 --eps 0.002
```

---

## Repository layout

```
ML-Emission-Tutorial/
├─ README.md
├─ script/
│  ├─ emission-rate-TD.py           # Build reference emission spectrum from emission-rate.dat
│  ├─ emission_spectrum.py          # Generic NEA convolution (used by the RIC scripts)
│  ├─ plot_emission_compare.py      # Compare ML vs REF (energy & wavelength views)
│  ├─ disparity_emission.py         # Compute metrics (RIC-like, RMSE, peak shift, overlap)
│  ├─ make_train_labels.py          # Build full-length y.*.train files from indices
│  ├─ extract_train_labels.py       # Extract E/f labels by training indices
│  ├─ mix_labels.py                 # Merge truth for train indices with ML predictions
│  ├─ training_set_generator.sh     # Split itrain.dat → isubtrain.dat + ivalidate.dat (80/20)
│  ├─ ML-RIC.sh                     # RIC for ML spectra (vs reference)
│  └─ TD-RIC.sh                     # RIC for TD subsets (vs full reference)
├─ ML_train_emission.sh             # One-click: train E/f → ML rate → convolve → ML spectrum
├─ data/
│  └─ emission-rate.dat             # Example; replace with your table (see format below)
├─ train/
│  ├─ 50/spectrum/emission/         # Put outputs for N=50 here (see Quickstart)
│  ├─ 100/spectrum/emission/
│  └─ 500/spectrum/emission/
├─ train_nums                       # e.g. a list: 50\n100\n500\n40000
└─ LICENSE                          # MIT recommended (or your choice)
```

---

## 1) Install

* **Python** ≥ 3.8
* **Packages**: `numpy`, `matplotlib`

```bash
python -m pip install -r requirements.txt  # optional
# or
python -m pip install numpy matplotlib
```

> This tutorial is **ML‑framework agnostic** (MLatom, SchNet, etc.). You only need to drop your predicted spectra in the expected paths.

### 1.1 Install MLatom (local)

Create and use a dedicated Conda environment:

```bash
# Option A: minimal
conda create -n mlatom python=3.10 -y
conda activate mlatom
pip install mlatom
# If desired, add commonly used extras:
pip install numpy scipy torch torchani tqdm matplotlib statsmodels h5py pyh5md

# Option B: from an official env file (if provided)
conda create -n mlatom --file mlatom.yml
conda activate mlatom
```

> On XACS cloud you can directly use the preconfigured environment; here we assume local/cluster use.

### 1.2 Newton‑X and Needed files

* Install **Newton‑X (NX)** and set the environment variable so scripts like `initcond.pl`/`makedir.pl` are on `PATH`:

  ```bash
  export NX=/path/to/Newton-X/bin
  ```
* Prepare the **input files** per the ML‑NEA tutorial’s “Needed files” list: [http://mlatom.com/tutorial/tutorial-mlnea-original/#Needed\_files](http://mlatom.com/tutorial/tutorial-mlnea-original/#Needed_files)

  Typical files expected by this repo:

  * `x.dat`: descriptors/input vectors (same length/order as the label table).
  * `state2.index.E.f`: the main label table containing indices, **E2** and **f2** for S1.
  * `itrain.dat`: training indices; optionally `isubtrain.dat` and `ivalidate.dat` if you split 80/20.
  * **For the reference spectrum**: `data/emission-rate.dat` (energy–rate table; col 1 = E \[eV], col 3 = `diff_rate`).

---

## 2) Input data format

**`data/emission-rate.dat`** should be a plain text table with **at least 3 columns**:

```
DE/eV    lambda/nm    diff_rate        +/-error
 2.9318   4.2289E+02   5.51056646E-09   1.99673392E-09
 2.9368   4.2217E+02   5.66272486E-09   1.95287277E-09
 ...
```

* Scripts read **E** from column 1 and **rate/intensity** from column 3.
* A header line is fine; it will be skipped automatically.
* If you only have **two columns** (E, I), most scripts also accept that — see flags below.

---

## 3) Build a reference spectrum (NEA convolution)

This produces two files at the project root:

* `emission_spectrum_ref_eV.dat` (E vs normalized intensity)
* `emission_spectrum_ref_full.dat` (E, λ, I, error=0)

```bash
python script/emission-rate-TD.py data/emission-rate.dat \
  --delta 0.06 --eps 0.002 --kappa 3.0
# → emission_spectrum_ref_eV.dat, emission_spectrum_ref_full.dat
```

**Notes**

* *δ (delta)* is the Gaussian broadening in eV (NEA line shape). Use smaller δ to avoid artificial over‑broadening; increase only to suppress stochastic noise if needed.
* `--no-smooth` and `--no-norm` are available for debugging.

---

## 4) Your ML spectra: where to put them

For each training size **N** you tested (e.g., 50, 100, 500, 40000), create:

```
train/N/spectrum/emission/emission_spectrum_full.dat
```

This is a two‑column file (E, normalized I). If your code outputs an energy‑domain spectrum under a different name, just rename/copy it here. A matching `emission_spectrum_eV.dat` (two columns) in the same folder is also used by plotting commands.

> If, instead, you have **predicted point‑wise rates** on the original grid, convolve them with `script/emission_spectrum.py`:

```bash
python script/emission_spectrum.py path/to/your_emission-rate.dat \
  --delta 0.06 --eps 0.002
# writes emission_spectrum_eV.dat + emission_spectrum_full.dat in CWD
```

---

## 5) Evaluate accuracy with RIC

Two complementary evaluations are provided.

### 5.1 TD subsets vs full reference

Assesses **statistical sampling error**: how much a TD/NEA spectrum built from only `N` points deviates from the **full reference**.

```bash
# Ensure you already built the full reference (Section 3)
# Prepare train_nums with e.g. 50, 100, 500, 2000, ...

bash script/TD-RIC.sh
# → TD-RIC.result (one RIC per N)
```

The script internally:

* Extracts the first `N` rows from `data/emission-rate.dat` (keeping header).
* Convolves them into `emission_spectrum_ref_${N}_full.dat`.
* Compares against the full reference using `disparity_emission.py` (no pre‑normalization).

### 5.2 ML spectra vs reference

Assesses **ML prediction error** (on a dense ensemble) against the **same reference**.

```bash
bash script/ML-RIC.sh
# → ML-RIC.result (one RIC per N with an existing ML spectrum)
```

The script expects ML spectra at `train/N/spectrum/emission/emission_spectrum_full.dat`.

> **What is RIC here?** We use $\int|Δ|\,dE / \int A\,dE$ on a common grid — a robust, unit‑free measure analogous to the “relative integral change”. See also additional metrics below.

---

## 6) Plot ML vs REF (energy & wavelength)

```bash
python script/plot_emission_compare.py \
  --ml-ev train/500/spectrum/emission/emission_spectrum_eV.dat \
  --ref-rate data/emission-rate.dat --delta 0.06 --eps 0.002
# writes compare_eV.png, compare_nm.png
```

* Use `--ref-ev emission_spectrum_ref_eV.dat` if you already built the reference.
* Intensities are normalized to their own maxima for shape comparison.

---

## 7) Extra: full set of disparity metrics

Besides RIC, you can compute multiple shape/position metrics on an automatically aligned energy grid:

```bash
python script/disparity_emission.py \
  emission_spectrum_ref_full.dat \
  train/500/spectrum/emission/emission_spectrum_full.dat \
  --norm none --eps 0.002
# prints: L1_norm_area, Relative_change, RMSE, Cosine, Overlap, PeakShift_eV
# and saves aligned curves to disparity_aligned.dat
```

Normalization modes: `none|max|area`.

---

## 8) (Optional) Preparing ML training labels

If your ML workflow needs clean label files from a combined state table (e.g., `state2.index.E.f`):

```bash
# 80/20 split of itrain.dat (first K indices are the training set)
bash script/training_set_generator.sh 500  # → isubtrain.dat, ivalidate.dat

# Make full-length y files where training indices hold truth and others are NaN
python script/make_train_labels.py state2.index.E.f itrain.dat y.E.train.dat --target E --fill nan
python script/make_train_labels.py state2.index.E.f itrain.dat y.f.train.dat --target f --fill nan

# Extract compact training-only vectors (E, f)
python script/extract_train_labels.py state2.index.E.f itrain.dat --prefix S1_ --suffix _train
# → S1_E_train.dat, S1_f_train.dat

# After prediction, merge truth-on-train with ML for the full 1..N range
python script/mix_labels.py state2.index.E.f ml_pred_E.dat y.E.em.dat --itrain itrain.dat --n_train 500 --target E --N 50000
python script/mix_labels.py state2.index.E.f ml_pred_f.dat y.f.em.dat --itrain itrain.dat --n_train 500 --target f --N 50000
```

These helpers keep indices **1‑based** to match common NEA/ensemble conventions.

---

## 9) Reproducible demo (end‑to‑end)

### 9.0 One‑click training & spectrum (`ML_train_emission.sh`)

This helper script automates **training E/f → building a rate table → convolving the ML emission spectrum**.

* Key parameters (editable near the top of the script):

  * `EM_STATE=2` (emission S1→S0 uses state2 labels **E2/f2**)
  * `WEIGHT`: `f`, `E*r`, or `E3` (default `E3`, i.e., `e^3 * f` weighting for `diff_rate`)
  * `DELTA`, `EPS`: line broadening and grid step for the convolution
* Expected inputs/paths (relative to repo root):

  * `data/x.dat`, `data/state2.index.E.f`, `data/ml.E.inp`, `data/ml.f.inp`
  * `data/itrain.dat` (the script can also create `isubtrain.dat`/`ivalidate.dat`)
* Run:

```bash
bash ML_train_emission.sh
```

Outputs under `train/<N>/spectrum/emission/`:

* `emission-rate-ML.dat` (from E2/f2 + weighting rule)
* `emission_spectrum_eV.dat` and `emission_spectrum_full.dat`

> Prefer the `script/` utilities (`make_train_labels.py`, `extract_train_labels.py`, `mix_labels.py`)? You can swap them in equivalently.

### 9.1 Reference

```bash
python script/emission-rate-TD.py data/emission-rate.dat --delta 0.06 --eps 0.002
```

### 9.2 Place an ML spectrum

```bash
mkdir -p train/500/spectrum/emission
cp emission_spectrum_ref_full.dat train/500/spectrum/emission/emission_spectrum_full.dat
cp emission_spectrum_ref_eV.dat   train/500/spectrum/emission/emission_spectrum_eV.dat
```

### 9.3 Evaluate & plot

```bash
printf "50\n100\n500\n" > train_nums
bash script/TD-RIC.sh
bash script/ML-RIC.sh
python script/plot_emission_compare.py --ml-ev train/500/spectrum/emission/emission_spectrum_eV.dat --ref-ev emission_spectrum_ref_eV.dat
```

---

## 10) How to cite

If you build ML‑accelerated NEA spectra, please cite the foundational ML‑NEA paper and your QC/ML stack. Example (APA/RSC snippets in your style):

* Xue, B.‑X.; Barbatti, M.; Dral, P. O. *Machine Learning for Absorption Cross Sections*. **J. Phys. Chem. A** (2020) 124, 7199‑7210. DOI: 10.1021/acs.jpca.0c05310.
* Crespo‑Otero, R.; Barbatti, M. *Spectrum Simulation and Decomposition with Nuclear Ensemble*. **Theor. Chem. Acc.** (2012) 131, 1237.

You may also cite the scripts/repo (add a `CITATION.cff` and/or archive a release on Zenodo for a DOI).

---

## 11) License

MIT is recommended for maximal reuse. Replace `LICENSE` if you prefer GPL/BSD/Apache.

---

## 12) (Optional) Turn this repo into a web tutorial

* **GitHub Pages (quick)**: enable Pages → *Deploy from branch* → `/docs` folder; copy sections of this README into `docs/index.md`.
* **MkDocs**: add `mkdocs.yml` and a `docs/` tree with short pages (Setup, Data, Reference, ML, Metrics). Run `pip install mkdocs mkdocs-material` locally and `mkdocs gh-deploy`.

### Minimal CI to run the demo and upload plots

Save as `.github/workflows/ci.yml`:

```yaml
name: demo
on: [push, pull_request]
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: '3.11' }
    - run: python -m pip install numpy matplotlib
    - run: |
        python script/emission-rate-TD.py data/emission-rate.dat --delta 0.06 --eps 0.002
        mkdir -p train/50/spectrum/emission
        cp emission_spectrum_ref_full.dat train/50/spectrum/emission/emission_spectrum_full.dat
        cp emission_spectrum_ref_eV.dat   train/50/spectrum/emission/emission_spectrum_eV.dat
        echo 50 > train_nums
        bash script/TD-RIC.sh
        bash script/ML-RIC.sh
        python script/plot_emission_compare.py --ml-ev train/50/spectrum/emission/emission_spectrum_eV.dat --ref-ev emission_spectrum_ref_eV.dat
    - uses: actions/upload-artifact@v4
      with:
        name: figures
        path: |
          compare_*.png
          *_RIC.result
```

This CI reuses the reference as a dummy ML spectrum for smoke‑testing (replace with a real ML output later).

---

## 13) Troubleshooting

* **“Parsing failed / No rows parsed”** → check that `emission-rate.dat` has numeric data under the header and columns 1/3 are E/rate.
* **Different grids** → plotting/metrics scripts auto‑interpolate to a common grid; you can force step with `--eps`.
* **All‑zero ML spectrum** → normalize only after checking max(I) > 0; the scripts handle this but upstream ML may need fixes.

---

## Acknowledgements

Thanks to the authors of NEA/ML‑NEA methods and open‑source tooling. Replace with your specific grants/credits.

---

*Happy spectrums!*
