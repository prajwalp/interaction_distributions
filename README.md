Code and data accompanying:

**Metabolic processes shape microbial interaction distributions**

Prajwal Padmanabha, Sara Mitri

## Overview

This repository reproduces the analyses and figures used/shown in the paper. The workflow has three stages — **generate** raw simulations, **process** them into summary arrays, and **plot** the figures. All processed data is included, so the figures can be reproduced without re-running the compute-heavy simulations.

## Repository structure

```
.
├── scripts/                      # Importable modules (the model + analysis engine)
│   ├── cr_model.py               # CR dynamics, EO conversion, gLV dynamics, interaction rescaling
│   ├── informed_glv.py           # Fitting interaction distributions, sampling matrices, gLV evaluation
│   └── utils.py                  # Compressed-pickle I/O, lognormal reparameterisation, diversity, helpers
│
├── generate/                     # Scripts that produce the data (run from inside this directory)
│   ├── 01_random_communities.py  # Random CR communities       → data/simulation_data/random_communities/
│   ├── 02_informed_glv.py        # Informed-gLV predictions     → data/simulation_data/predictability/
│   ├── 03_analytical_skew.py     # Analytical + simulated skew  → data/figures/fig2/
│   ├── 04_example_matrices.py    # Example interaction matrices → data/figures/fig2/
│   └── 05_predicting_data.py     # Diversity predictions        → data/figures/fig4/
│
├── figures/                      # Plotting notebooks and rendered output (run from inside this directory)
│   ├── figures.ipynb             # Figures 1, 3, 4
│   ├── data_analysis.ipynb       # Figure 2 and the nine-dataset analysis
│   ├── generated/                # Individual rendered panels (SVG / PNG), incl. supplementary (SM/)
│   └── collated/                 # Assembled main-text figures
│
└── data/
    ├── interaction_data/         # The nine empirical datasets
    ├── simulation_data/          # Raw CR and gLV simulation output (.pbz2)
    └── figures/                  # Processed arrays that the notebooks plot from
```

## Installation

Developed with **Python 3.9**. Install the dependencies with:

```bash
pip install numpy scipy pandas matplotlib seaborn numba numbalsoda colormaps statsmodels jupyter
```

## Quickstart — reproduce the figures

The processed data needed for plotting is already in `data/`, so the figures can
be regenerated directly:

```bash
cd figures
jupyter lab        # then run figures.ipynb and data_analysis.ipynb
```

Run the notebooks from inside `figures/` — their data paths are relative to that directory.

## Full pipeline

### 1. Generate

Run the generation scripts **from inside `generate/`** (their data paths are relative to that folder; the `scripts/` import is resolved automatically):

```bash
cd generate
python 01_random_communities.py
python 02_informed_glv.py
python 03_analytical_skew.py
python 04_example_matrices.py
python 05_predicting_data.py
```

`01` and `02` are parallelised and compute-heavy; they write raw simulation dumps (`.pbz2`) into `data/simulation_data/`. `03`–`05` write processed arrays directly into `data/figures/`.

`02` runs one growth-rate heterogeneity value per invocation and `04` runs one growth-rate CV per invocation. The shipped data covers ``gamma`` = 1–2 and CV = 1;

### 2. Process

Aggregation steps turn the raw `.pbz2` files into the summary arrays the that can be used to plot the figures. Each lives in a commented "run once" cell near the top of a notebook, and its output is already included:

- **`data_analysis.ipynb`** — aggregates `simulation_data/random_communities/` to `data/figures/fig1/interaction_measures.npy` (per-community skewness and  kurtosis, plus the normality test used for Figure 2B).
- **`figures.ipynb`** — aggregates `simulation_data/predictability/` to `data/figures/fig3/predictability_data_highSupply.npz`.

Uncomment and run these once if you have regenerated the raw simulations. Not to change the ``dateTimeStamp`` variable if you do so.

### 3. Plot

| Figure | Notebook | Reads from |
|--------|----------|------------|
| **Fig 1** — interaction skew and its drivers | `figures.ipynb` | `data/figures/fig1/`, `fig2/` |
| **Fig 2** — nine datasets, skew–kurtosis relationship | `data_analysis.ipynb` | `data/interaction_data/`, `fig1/` |
| **Fig 3** — informed-gLV predictions | `figures.ipynb` | `data/figures/fig3/` |
| **Fig 4** — diversity prediction (duckweed, gut) | `figures.ipynb` | `data/figures/fig4/` |

Rendered panels are written to `figures/generated/` — the `savefig` calls are commented out by default; uncomment them to overwrite the shipped panels.

## The model (`scripts/`)

- **`cr_model.py`** — the consumer–resource model (Monod uptake with linear cross-feeding), the Environment–Organism (EO) conversion from a CR steady state to effective gLV growth rates and interactions, the canonical rescaling `a_ij = b_ij * r_j / (b_jj * r_i)`, and the gLV integrator.
- **`informed_glv.py`** — fits lognormal / Gaussian / correlated interaction distributions to a matrix and samples new matrices from them, plus the routine that integrates an informed gLV community and measures its diversity.
- **`utils.py`** — compressed-pickle I/O, the normal to lognormal parameter conversion, Shannon diversity, and small helpers.

## Datasets

The nine empirical interaction datasets in `data/interaction_data/` are derived from previously published studies (Stein et al., Kehe et al., Schäfer et al., Ishizawa et al., Ho et al., Weiss et al., Clark et al., Arias-Sánchez et al., and Merz et al.). Please cite the original sources if you reuse them; the paper gives the full references and describes the processing applied to each.

## License

See MIT Licence.txt. Note that the redistributed datasets in `data/interaction_data/` retain the licenses of their original publications.
