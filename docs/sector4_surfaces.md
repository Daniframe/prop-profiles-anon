# Sector 4: Propensity Surfaces

Package: `src/propensity/surfaces/` · CLI: `scripts/derive_propensity_curves.py` · Config: `config/surfaces.yaml`

Fits a single propensity point `theta`, an empirical curve, and an empirical 2D
surface for one model from item-level (demand interval, success/failure) data
-- paper Eq. 2-6. This is the only sector with no
credentials/GPU dependency: everything here is local numpy/scipy/pandas math,
fully unit-tested in `tests/test_data.py`, `tests/test_mle.py`, `tests/test_model.py`,
and `tests/test_curves_surfaces.py`.

## Pipeline

```
demand annotations (JSONL/CSV)  ─┐
                                  ├─► build_arrays ─► fit_theta ─► theta_hat, CI95, GoF
outcomes (wide CSV)              ─┘        │
                                            ├─► build_empirical_curve  ─► plot_prop_curve
                                            └─► build_empirical_surface ─► plot_prop_surface
```

## Modules

| Module | Responsibility |
|---|---|
| `model.py` | `two_sided_sigma` -- the canonical Eq. 5 propensity model (a normalized product of two logistic sigmoids, peaking at 1.0 at the demand interval's midpoint). |
| `variants.py` | Superseded exploratory curve formulations, kept only for reference. Not used elsewhere. |
| `data.py` | I/O: load demand annotations (single- or multi-dimension, JSONL/CSV) and outcomes (wide CSV), join them into fit-ready arrays. |
| `mle.py` | Eq. 6 maximum-likelihood fit of `theta` via `scipy.optimize.minimize` (BFGS). |
| `curves.py` | Bins observed success by interval-center and LOWESS-smooths it ("PROP CURVE" data). |
| `surfaces.py` | Aggregates success onto an integer `(b_l, b_u)` grid ("PROP HEATMAP" data). |
| `plotting.py` | Matplotlib rendering of both, decoupled from the fitting logic. |

## Usage

### Loading data

```python
from src.propensity.surfaces import data

# Single propensity dimension per file (columns default to
# propensity_lower/propensity_upper, renamed to lower/upper):
demands = data.load_demands_single_dim("data/benchmarks/annotated/RA/RA_RA_annotations_GPT-4.1.jsonl")

# Multiple dimensions in one file (e.g. a cross-benchmark annotation file with
# RA_l/RA_u, Ex_l/Ex_u columns) -- pull one dimension back out for fitting:
multi = data.load_demands_multi_dim(
    "data/benchmarks/annotated/TimeMenatQA/TimeMenatQA_Ul_annotations_GPT-4.1.jsonl",
    dim_names=["Ul"],
)
demands = data.select_dim_demands(multi, dim="Ul")

outcomes = data.load_outcomes("data/inference/outcomes/RA/RA_complete.csv")
demand_array, success_array = data.build_arrays(demands, outcomes, model_name="4o_RA")
```

### Fitting theta (Eq. 6 MLE)

```python
from src.propensity.surfaces import mle

fit = mle.fit_theta(demand_array, success_array, k=1.0)
# {'theta_hat': -1.34, 'se': 0.01, 'ci95_lower': -1.36, 'ci95_upper': -1.32,
#  'convergence': 1.0, 'reference_ll': ..., 'gof': ..., 'pseudo_r2': ...}
```

### Curve and surface + plotting

```python
from src.propensity.surfaces import curves, surfaces, plotting
import matplotlib.pyplot as plt

curve = curves.build_empirical_curve(demand_array, success_array, n_bins=20, lowess_frac=0.4)
plotting.plot_prop_curve(curve, fit["theta_hat"], fit["ci95_lower"], fit["ci95_upper"], r1=-3, r2=3)
plt.savefig("PC_4o_RA.png", dpi=100)

grid_demands = demand_array.round().astype(int)  # surface needs an integer grid
surface = surfaces.build_empirical_surface(grid_demands, success_array, r1=-3, r2=3)
plotting.plot_prop_surface(surface, fit["theta_hat"], fit["ci95_lower"], fit["ci95_upper"], model_name="4o_RA")
plt.savefig("HM_4o_RA.png", dpi=100)
```

### The bare model (no fitting)

```python
from src.propensity.surfaces.model import two_sided_sigma

p_success = two_sided_sigma(x=0.5, x1=-2, x2=3, k1=1.0, k2=1.0)  # peaks at 1.0 at x=(x1+x2)/2
```

### CLI

```bash
python scripts/derive_propensity_curves.py \
  --demands_filename data/benchmarks/annotated/RA/RA_RA_annotations_GPT-4.1.jsonl \
  --outcomes_filename data/inference/outcomes/RA/RA_complete.csv \
  --model_name 4o_RA \
  --plot_folder /tmp/plots
```

Prints one CSV line to stdout (`true_x,fitted_x,ci95_lower,ci95_upper,convergence,reference_ll,gof`)
and writes `PC_{model_name}.png` / `HM_{model_name}.png` into `--plot_folder`.
Reads `n_bins`/`lowess_frac`/`maxiter`/`k_default` from `config/surfaces.yaml`.

No credentials needed -- this sector only touches local files.
