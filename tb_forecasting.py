# %% [markdown]
# # Forecasting monthly tuberculosis notifications in Nepal — hybrid SARIMA–CNNAR
#
# **Analysis notebook (S1 File)** for *“Data-Driven Hybrid Model of SARIMA-CNNAR For Tuberculosis Incidence Time Series Analysis in Nepal”* (Public Health Challenges, manuscript 3049755).
#
# This notebook reproduces **every table, figure and number** in the manuscript and supplementary tables S1–S9 from the monthly data embedded in Section 2.
#
# ## How to run
# 1. **One-time setup** of a dedicated environment. Do not use the Anaconda *base* environment. In Terminal, in the folder with these files:
#    ```
#    bash setup_environment.sh
#    ```
#    The script creates the conda environment `tb_forecast` (Python 3.12), installs compatible versions of all packages, registers the Jupyter kernel and tests the imports. To do it by hand instead:
#    ```
#    conda create -n tb_forecast -c conda-forge python=3.12 -y
#    conda activate tb_forecast
#    python -m pip install -r requirements.txt
#    python -m ipykernel install --user --name tb_forecast --display-name "Python (tb_forecast)"
#    ```
#    Then select the kernel **Python (tb_forecast)** in Jupyter (*Kernel → Change kernel*). Cell 1.0 checks the environment and says exactly what to fix if something is missing or broken.
#    The published results were produced with Python 3.12.16, numpy 2.5.5, pandas 3.1.1, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1, tensorflow 2.21.0, xgboost 3.4.1, prophet 1.5.0, ruptures 1.1.10, matplotlib 3.11.2, seaborn 0.13.2 (macOS, Apple silicon).
# 2. Optionally edit **Section 1 (Configuration)**: output folder, which stages to run, quick-test mode.
# 3. Run all cells (*Kernel → Restart & Run All*), or from a terminal:
#    ```
#    jupyter nbconvert --to notebook --execute TB_SARIMA_CNNAR_Analysis.ipynb --ExecutePreprocessor.timeout=-1
#    ```
#
# ## Run time and caching
# | Stage | Approx. time (2 CPU cores) | Cached? |
# |---|---|---|
# | Data, EDA, tests, SARIMA, breaks | < 1 min | – |
# | SARIMA grid search (729 models) | ~15 min | yes |
# | 5-fold cross-validation, all models × 5 seeds | ~35 min | yes |
# | Over-fitting, regularised networks, ablation, lockdown indicator (all folds) | ~35 min | yes |
# | Sensitivity analyses | ~30 min | yes |
# | 2025 forecasts | ~5 min | yes |
#
# Expensive stages are saved to `OUTPUT_DIR/_cache/<mode>_<data-hash>/` the first time they run. Later runs load them in seconds, so you can re-run any section **on demand**. Set `USE_CACHE = False`, or delete the cache folder, to recompute everything from scratch. `QUICK_TEST = True` runs a fast smoke test (fewer epochs, one seed); its numbers are **not** the manuscript numbers.
#
# ## Reproducibility
# * All random seeds are fixed, and TensorFlow deterministic operations are enabled. Results are identical on the same platform and library versions.
# * Statistical models (SARIMA, SARIMAX, tests, Bai–Perron, Prophet) are deterministic. Neural networks and XGBoost can differ slightly (typically < 0.5 MAPE points) on other hardware or library versions.
# * Section 16 compares the key results with the published values and reports PASS/CHECK for each.
#
# ## Input and outputs
# **Input:** `TB_Incidence.csv` in the project folder (set `DATA_FILE` in Section 1). It is checked against the published dataset before any analysis runs.
#
# **Outputs** go to `…/Final after revision Submitted files/Outputs of Model` (set `OUTPUT_DIR` in Section 1):
# ```
# Outputs of Model/
# ├── 01_Manuscript_Tables/          Table01 … Table18 (CSV) + Manuscript_Tables.xlsx (one sheet per table)
# ├── 02_Supplementary_Tables/       S1 … S9 (CSV + XLSX)
# ├── 03_Model_Outputs/              2025 forecasts, cross-validation forecasts, fitted-model summaries
# ├── 04_Exploratory_and_Tests/      descriptive statistics and statistical tests
# ├── 05_Figures_600dpi_TIFF/        Fig01 … Fig22 (.tif, 600 dpi, RGB, LZW)
# ├── 06_Logs_and_Reproducibility/   environment, input-data check, SARIMA fit log, reproducibility check
# ├── Outputs_index.csv              list of every output file
# └── _cache/                        saved results of the expensive stages (safe to delete)
# ```
#
# ## Notebook map
# | Section | Content | Manuscript |
# |---|---|---|
# | 1 | Configuration | – |
# | 2 | Data and back-casting | S1 Table, Section 2.3 |
# | 3 | Model and statistics library | Sections 2.5–2.12 |
# | 4 | Exploratory analysis | Table 1, Figs 4–10, 13–17 |
# | 5 | Seasonality tests and structural breaks | Tables 2–3, Figs 11–12 |
# | 6 | SARIMA: grid search, benchmark, residual tests | Tables 4–6, Fig 18, S2 Table |
# | 7 | Cross-validation of all models | Tables 8, 10, 12, 13, Fig 20, S3 Table |
# | 8 | Structural breaks as model variables | Table 11, S8 Table |
# | 9 | Over-fitting, regularisation, ablation, lockdown indicator | Table 15, Fig 21, S9 Table |
# | 10 | Forecast comparison: DM tests, bootstrap CIs, interval coverage | Tables 14, 16, Fig 19, S6 Table |
# | 11 | Sensitivity analyses | Tables 17–18, S4 Table |
# | 12 | 2025 forecasts | Tables 4, 7, 9, Fig 22, S7 Table |
# | 13 | Residual diagnostics of the test-year forecasts | S5 Table |
# | 14 | Method figures | Figs 1–3 |
# | 15 | Export of manuscript and supplementary tables | Tables 1–18, S2–S9 |
# | 16 | Reproducibility check | – |
# | 17 | Final model summaries and index of all outputs | – |

# %% [markdown]
# ## 1. Configuration
# Edit this cell only. All other cells read their settings from here.

# %%
from pathlib import Path
import os

PROJECT_DIR = Path("/Users/dipsingh/Documents/Data Driven Hybrid Model of SARIMACNNAR For Tuberculosis Incidence Time Series Analysis in Nepal")

# Input dataset (CSV). Wide format: one column of month names + one column per year (2015...2024 or 2018...2024).
# Long format (a date column + a value column) is also accepted. Environment variable TB_DATA_FILE overrides it.
DATA_FILE = Path(os.environ.get("TB_DATA_FILE", PROJECT_DIR / "TB_Incidence.csv"))

# All outputs (tables, model results, figures) go here, organised in sub-folders (created if missing).
# Environment variable TB_OUTPUT_DIR overrides it.
OUTPUT_DIR = Path(os.environ.get("TB_OUTPUT_DIR", PROJECT_DIR / "Final after revision Submitted files" / "Outputs of Model"))

# Figures: publication-ready TIFF
FIG_DPI    = 600      # resolution of every saved figure
FIG_FORMAT = "tif"    # file extension (TIFF, LZW-compressed, RGB, white background)

# Run mode
QUICK_TEST = False   # True = fast smoke test (few epochs, 1 seed); False = manuscript results
USE_CACHE  = True    # reuse saved results of expensive stages; False = recompute everything

# Stages to run on demand (a stage that is switched off is skipped, together with the tables and figures that need it)
RUN = dict(
    eda             = True,   # Section 4
    tests_breaks    = True,   # Section 5
    sarima          = True,   # Section 6 (benchmark, residual tests)
    grid_search     = True,   # Section 6 (729-model grid, ~15 min first time)
    cross_validation= True,   # Section 7 (needed by Sections 9-13 and 15)
    extended        = True,   # Section 9 (regularised networks, ablation, lockdown indicator)
    sensitivity     = True,   # Section 11
    forecast_2025   = True,   # Section 12
    figures         = True,   # save figures
)

# Study design
FOLD_YEARS   = [2020, 2021, 2022, 2023, 2024]   # expanding-window CV: test years
H            = 12        # forecast horizon (months)
ALPHA        = 0.05      # 95% intervals
BURN_IN      = 13        # SARIMA start-up values (d + D*s = 1 + 12) excluded from residuals/training metrics
N_SEEDS      = 5         # training runs per neural model
N_SEEDS_SENS = 2         # seeds per configuration in the sensitivity analysis
MC_SAMPLES   = 100       # Monte-Carlo dropout trajectories
N_BOOT       = 2000      # block-bootstrap replicates
EPOCH_SCALE  = 1.0       # multiplies all epoch numbers (quick mode only)

# Calendar of the input file. NTCC/DHIS2 reports by Nepali (Bikram Sambat) month; "BS" means each value is a Nepali
# month labelled by the Gregorian month in which it starts (e.g. Shrawan 2077 -> July 2020). The values are converted
# to Gregorian calendar months in Section 2 before any analysis. Use "AD" if the file already holds Gregorian months.
INPUT_CALENDAR = "BS"

# A-priori COVID-19 restriction periods (dates inclusive): nationwide lockdown 24 Mar-21 Jul 2020; district prohibitory
# orders of the second wave, in force in 74-75 of 77 districts, from 29 Apr 2021 until public transport resumed on 29 Jun 2021.
# The lockdown regressor of a month is the share of its days under restriction (Section 2.9).
LOCKDOWN_PERIODS = [("2020-03-24", "2020-07-21"), ("2021-04-29", "2021-06-28")]

if QUICK_TEST:
    N_SEEDS, N_SEEDS_SENS, MC_SAMPLES, N_BOOT, EPOCH_SCALE = 1, 1, 20, 200, 0.05

# %% [markdown]
# ### 1.0 Environment check
# Imports each required package and stops with a clear message if one is missing or broken. If this cell fails, fix the environment (see the message and the README), **restart the kernel**, and run again. Running further cells would only give follow-on errors such as `NameError`.

# %%
import importlib, sys, platform, os, logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")          # silence TensorFlow start-up messages
logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)  # optional plotly not needed

REQUIRED = [  # (import name, pip name, minimum version)
    ("numpy", "numpy", "1.26"), ("pandas", "pandas", "2.1"), ("scipy", "scipy", "1.11"),
    ("PIL.Image", "pillow", "10.0"), ("matplotlib.pyplot", "matplotlib", "3.8"), ("seaborn", "seaborn", "0.13"),
    ("statsmodels.api", "statsmodels", "0.14"), ("sklearn", "scikit-learn", "1.3"), ("tensorflow", "tensorflow", "2.16"),
    ("xgboost", "xgboost", "2.0"), ("prophet", "prophet", "1.1"), ("ruptures", "ruptures", "1.1.9"), ("openpyxl", "openpyxl", "3.1"),
]

def _hint(pip_name, err):
    msg = str(err)
    if "NumPy 1.x" in msg or "numpy.core" in msg or "_multiarray_umath" in msg:
        return (f"'{pip_name}' was built for NumPy 1.x but this environment has NumPy {np_version}. "
                "Fix: create the dedicated environment with setup_environment.sh (see README), which installs matching "
                "versions automatically. In an existing environment: pip install -U tensorflow  (Apple-silicon Mac) "
                "or  pip install \"numpy<2\"  (Intel Mac, where TensorFlow stops at 2.16), then restart the kernel.")
    if "Symbol not found" in msg or "dlopen" in msg or "DLL load failed" in msg or "undefined symbol" in msg:
        return (f"the compiled library of '{pip_name}' does not match this environment (common in Anaconda on macOS). "
                f"Fix: run  pip install --force-reinstall --no-cache-dir {pip_name}  in this environment "
                f"(conda users: or  conda install -c conda-forge {pip_name} libcxx), then restart the kernel. "
                f"Most reliable: a fresh environment as described in the README.")
    if "libomp" in msg or "OpenMP" in msg:
        return ("the OpenMP runtime (libomp.dylib) is missing. Fix (macOS): in Terminal run  "
                "CONDA_SOLVER=classic conda install -n tb_forecast -c conda-forge llvm-openmp -y  "
                "(or  brew install libomp), then restart the kernel.")
    if isinstance(err, ModuleNotFoundError):
        return f"not installed. Fix: pip install {pip_name}, then restart the kernel."
    return f"import failed ({type(err).__name__}). Fix: pip install --force-reinstall {pip_name}, then restart the kernel."

def _version_tuple(v):
    out = []
    for part in str(v).split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)

_DIRS = {"PIL": "pillow", "numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "matplotlib": "matplotlib",
         "sklearn": "scikit-learn", "tensorflow": "tensorflow", "xgboost": "xgboost", "statsmodels": "statsmodels"}
try:
    import numpy as _np
    np_version = _np.__version__
except Exception:
    np_version = "?"
in_base_env = os.environ.get("CONDA_DEFAULT_ENV") == "base" or ((Path(sys.prefix) / "conda-meta").exists() and "envs" not in Path(sys.prefix).parts)
print(f"Python {platform.python_version()} | {sys.executable}")
if in_base_env:
    print("Note: this kernel runs in the Anaconda *base* environment. A dedicated environment is strongly recommended "
          "(run setup_environment.sh, then choose the kernel 'Python (tb_forecast)').")
problems = []
for mod, pip_name, min_v in REQUIRED:
    try:
        m = importlib.import_module(mod)
        top = importlib.import_module(mod.split(".")[0])
        ver = getattr(top, "__version__", "?")
        old = ver != "?" and _version_tuple(ver) < _version_tuple(min_v)
        print(f"  {'OLD ' if old else 'OK  '} {pip_name:<13} {ver}" + (f"  (needs >= {min_v})" if old else ""))
        if old:
            problems.append(f"{pip_name} {ver} is older than {min_v}. Fix: pip install -U {pip_name}")
    except Exception as err:            # ImportError, OSError from broken binaries, etc.
        # a package can fail because a dependency is broken (e.g. matplotlib -> Pillow): name the real culprit
        abi = "NumPy 1.x" in str(err) or "numpy.core" in str(err)
        culprit = pip_name if abi else next((p for d, p in _DIRS.items() if f"/{d}/" in str(err) or f"\\{d}\\" in str(err)), pip_name)
        print(f"  FAIL {pip_name:<13} {type(err).__name__}: {str(err).splitlines()[0][:110]}"
              + (f"  [caused by {culprit}]" if culprit != pip_name else ""))
        if not any(p.startswith(f"{culprit}:") for p in problems):
            problems.append(f"{culprit}: {_hint(culprit, err)}")
if sys.version_info < (3, 10):
    problems.append("Python 3.10 or newer is required.")
if problems:
    raise RuntimeError("Environment not ready:\n  - " + "\n  - ".join(problems))
print("Environment OK.")

# %% [markdown]
# ### 1.1 Environment, folders and helpers for saving results

# %%
import sys, time, random, pickle, logging, warnings, importlib, platform
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import scipy
from scipy import stats
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, bds
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.nonparametric.smoothers_lowess import lowess
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import xgboost as xgb
import ruptures as rpt
from prophet import Prophet

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
for _lg in ("cmdstanpy", "prophet"):
    logging.getLogger(_lg).setLevel(logging.WARNING)
try:
    tf.config.experimental.enable_op_determinism()   # identical neural-network results on the same platform
except Exception as e:
    print("TensorFlow op determinism not available:", e)

import io
from PIL import Image

MODE = "quick" if QUICK_TEST else "full"
# Output folder structure
DIRS = {
    "manuscript":    OUTPUT_DIR / "01_Manuscript_Tables",          # Table01_..., Table18_... (CSV) + Manuscript_Tables.xlsx
    "supplementary": OUTPUT_DIR / "02_Supplementary_Tables",       # S1-S9 (CSV + XLSX)
    "models":        OUTPUT_DIR / "03_Model_Outputs",              # forecasts, fitted models, cross-validation results
    "analysis":      OUTPUT_DIR / "04_Exploratory_and_Tests",      # descriptive statistics and statistical tests
    "figures":       OUTPUT_DIR / f"05_Figures_{FIG_DPI}dpi_TIFF", # Fig01 ... Fig22
    "logs":          OUTPUT_DIR / "06_Logs_and_Reproducibility",   # environment, data check, fit log, reproducibility check
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

def _folder_for(name):
    """Choose the output sub-folder from the file name prefix."""
    if name.startswith("Table"):
        return DIRS["manuscript"]
    if name.startswith(("S1_", "S2_", "S3_", "S4_", "S5_", "S6_", "S7_", "S8_", "S9_")):
        return DIRS["supplementary"]
    if name.startswith(("Model_", "Forecast")):
        return DIRS["models"]
    if name.startswith(("00_", "Data_", "SARIMA_fit_log", "reproducibility", "Run_")):
        return DIRS["logs"]
    return DIRS["analysis"]

def save_table(df, name, index=False, folder=None):
    """Save a DataFrame as <sub-folder>/<name>.csv (sub-folder chosen from the name) and return it."""
    (folder or _folder_for(name)).mkdir(parents=True, exist_ok=True)
    df.to_csv((folder or _folder_for(name)) / f"{name}.csv", index=index)
    return df

# Publication style for all figures
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": FIG_DPI, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "axes.linewidth": 0.8, "lines.linewidth": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white",
})

def save_figure(fig, name):
    """Save a figure as <OUTPUT_DIR>/05_Figures_600dpi_TIFF/<name>.tif:
    600 dpi, RGB (no transparency), white background, LZW compression, then close it."""
    if RUN["figures"]:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        buf.seek(0)
        with Image.open(buf) as im:
            im.convert("RGB").save(DIRS["figures"] / f"{name}.{FIG_FORMAT}", format="TIFF",
                                   compression="tiff_lzw", dpi=(FIG_DPI, FIG_DPI))
    plt.close(fig)

def cached(name, compute):
    """Run `compute()` once and store its result in the cache; later calls load it.
    The cache folder depends on the run mode and on the input data (CACHE_DIR, set in Section 2),
    so a changed dataset is never combined with old results."""
    path = CACHE_DIR / f"{name}.pkl"
    if USE_CACHE and path.exists():
        print(f"[cache] loaded {name}")
        return pickle.load(open(path, "rb"))
    t0 = time.time()
    result = compute()
    pickle.dump(result, open(path, "wb"))
    print(f"[computed] {name} in {(time.time() - t0) / 60:.1f} min")
    return result

def ep(n):
    """Number of training epochs (scaled down only in quick mode)."""
    return max(5, int(round(n * EPOCH_SCALE)))

versions = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
            "scikit-learn": sklearn.__version__, "tensorflow": tf.__version__, "xgboost": xgb.__version__,
            "ruptures": getattr(rpt, "__version__", "?"), "prophet": importlib.import_module("prophet").__version__,
            "matplotlib": matplotlib.__version__, "seaborn": sns.__version__}
print(f"Mode: {MODE} | output: {OUTPUT_DIR}")
print(" | ".join(f"{k} {v}" for k, v in versions.items()))
save_table(pd.DataFrame([versions]), "00_environment")
print(f"Outputs: {OUTPUT_DIR}")

# %% [markdown]
# ## 2. Data
# Monthly national TB notifications (NTCC Nepal, DHIS2), **observed January 2018–December 2024**.
# January 2015–December 2017 are **back-cast**, separately for each calendar month *m*, with the compound annual growth rate between 2018 and 2024:
#
# $r_m = (y_{2024,m}/y_{2018,m})^{1/6} - 1, \qquad y_{2018-j,m} = y_{2018,m}/(1+r_m)^j,\; j = 1,2,3$ (rounded to whole cases).

# %%
import hashlib, re

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
# Published dataset (NTCC monthly notifications, January..December): used to VALIDATE the input file
PUBLISHED = {
    2018: [1812, 2406, 2679, 2798, 2835, 2813, 2718, 2493, 2305, 2098, 2094, 2098],
    2019: [2008, 2396, 2627, 2979, 3043, 2735, 2676, 2692, 2030, 2145, 2400, 2255],
    2020: [2475, 2675, 1881, 1772, 2069, 2472, 2782, 2093, 2500, 1865, 2016, 2457],
    2021: [2293, 2739, 3370, 2450, 1562, 2475, 3503, 3136, 2572, 2593, 2992, 2944],
    2022: [2292, 3208, 3775, 3269, 3835, 3669, 3562, 3197, 2480, 2613, 2789, 2524],
    2023: [2599, 2977, 3620, 3483, 4150, 3453, 3608, 2982, 3037, 2027, 3012, 2784],
    2024: [2991, 3531, 4173, 4132, 4635, 3863, 3839, 3082, 2592, 2619, 3030, 2882],
}

def backcast(observed, first=2018, last=2024, n_back=3):
    """Month-specific CAGR back-cast of the n_back years before `first` (Section 2.3)."""
    y0, y1 = np.array(observed[first], float), np.array(observed[last], float)
    r = (y1 / y0) ** (1 / (last - first)) - 1
    return {first - j: list(np.round(y0 / (1 + r) ** j).astype(int)) for j in range(n_back, 0, -1)}

def load_tb_csv(path):
    """Read the TB dataset and return {year: [12 monthly values, Jan..Dec]}.
    Accepts (a) wide format: a column with month names (or numbers 1-12) + one column per year, or
            (b) long format: a date column + one numeric value column."""
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    year_cols = [c for c in raw.columns if re.fullmatch(r"(19|20)\d\d(\.0)?", c)]
    if len(year_cols) >= 2:                                             # ---- wide format
        month_col = None
        for c in raw.columns:
            if c in year_cols:
                continue
            vals = raw[c].astype(str).str.strip().str.capitalize()
            if vals.isin(MONTHS).sum() >= 12 or vals.str[:3].isin([m[:3] for m in MONTHS]).sum() >= 12:
                month_col = c
                break
        if month_col is not None:
            names = raw[month_col].astype(str).str.strip().str[:3].str.capitalize()
            order = [m[:3] for m in MONTHS]
            raw = raw[names.isin(order)].assign(_m=names.map({m: k for k, m in enumerate(order)})).sort_values("_m")
        else:                                                           # no month names: assume rows are Jan..Dec
            raw = raw.head(12)
        if len(raw) != 12:
            raise ValueError(f"{path}: expected 12 monthly rows, found {len(raw)}")
        return {int(float(c)): [float(v) for v in pd.to_numeric(raw[c], errors="coerce")] for c in year_cols}
    date_col = next((c for c in raw.columns if pd.to_datetime(raw[c], errors="coerce").notna().mean() > 0.9
                     and not pd.api.types.is_numeric_dtype(raw[c])), None)   # ---- long format
    val_col = next((c for c in raw.columns if c != date_col and pd.api.types.is_numeric_dtype(raw[c])), None)
    if date_col is None or val_col is None:
        raise ValueError(f"{path}: could not recognise the layout (need month + year columns, or date + value columns)")
    ser = pd.Series(pd.to_numeric(raw[val_col], errors="coerce").values, index=pd.to_datetime(raw[date_col])).sort_index()
    return {yr: [float(v) for v in ser[ser.index.year == yr].values] for yr in sorted(set(ser.index.year))}

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Input dataset not found:\n  {DATA_FILE}\nSet DATA_FILE in Section 1 to the correct path.")
DATA = load_tb_csv(DATA_FILE)
print(f"Input: {DATA_FILE}\nYears in file: {sorted(DATA)}")

# --- checks: completeness, then agreement with the published data -----------------------
problems = [f"{y}: {len(v)} months" for y, v in DATA.items() if len(v) != 12] + \
           [f"{y}: missing/non-numeric values" for y, v in DATA.items() if np.isnan(v).any()]
if problems:
    raise ValueError("Input data incomplete: " + "; ".join(problems))
missing_obs = [y for y in range(2018, 2025) if y not in DATA]
if missing_obs:
    raise ValueError(f"Observed years missing from the input file: {missing_obs}")

# Validation against the published dataset (reported values, before preprocessing)
BACKCAST = backcast({y: [int(round(v)) for v in DATA[y]] for y in range(2018, 2025)})

# %% [markdown]
# ### 2.1 Preprocessing: Kalman-filter imputation and conversion to Gregorian months
# NTCC/DHIS2 reports notifications by **Nepali (Bikram Sambat) month**, which runs from mid-month to mid-month
# (e.g. Shrawan 2077 = 16 July–16 August 2020). The input file labels each Nepali month by the Gregorian month in which
# it starts. All analyses use **Gregorian calendar months**, so that the series can be compared with dated events
# (lockdown, break dates) and with other data. The steps are:
# 1. **Complete the Nepali-month series.** A Gregorian month needs both Nepali months that overlap it. January 2018 also
#    needs Poush 2074 (16 December 2017–14 January 2018), which is not in the observed data, and any missing value in the
#    input file is treated the same way. Missing Nepali months are estimated with the **Kalman smoother** of a structural
#    time-series model (local linear trend + stochastic 12-month seasonal component) fitted by maximum likelihood to all
#    reported Nepali months; the Kalman smoother uses the observations before and after each gap.
# 2. **Convert to Gregorian months.** Each Nepali month's count is spread uniformly over its days and the daily values
#    are summed within each Gregorian month (day-weighted apportioning), then rounded to whole cases.
# 3. **Back-cast** January 2015–December 2017 from the Gregorian 2018–2024 series (month-specific CAGR, above).
#
# Every model is fitted to this final, complete Gregorian series. S1 Table gives the reported Nepali-month values,
# the imputed values and the converted series.

# %%
BS_MONTHS = ["Baishakh", "Jestha", "Asar", "Shrawan", "Bhadra", "Ashwin", "Kartik", "Mangsir", "Poush", "Magh", "Falgun", "Chaitra"]
# First day (Gregorian) of each Nepali month, Poush 2074 to Magh 2081 (Nepal Government calendar)
BS_START = {"2074-09": "2017-12-16", "2074-10": "2018-01-15", "2074-11": "2018-02-13", "2074-12": "2018-03-15", "2075-01": "2018-04-14", "2075-02": "2018-05-15", "2075-03": "2018-06-15", "2075-04": "2018-07-17", "2075-05": "2018-08-17", "2075-06": "2018-09-17", "2075-07": "2018-10-18", "2075-08": "2018-11-17", "2075-09": "2018-12-16", "2075-10": "2019-01-15", "2075-11": "2019-02-13", "2075-12": "2019-03-15", "2076-01": "2019-04-14", "2076-02": "2019-05-15", "2076-03": "2019-06-16", "2076-04": "2019-07-17", "2076-05": "2019-08-18", "2076-06": "2019-09-18", "2076-07": "2019-10-18", "2076-08": "2019-11-17", "2076-09": "2019-12-17", "2076-10": "2020-01-15", "2076-11": "2020-02-13", "2076-12": "2020-03-14", "2077-01": "2020-04-13", "2077-02": "2020-05-14", "2077-03": "2020-06-15", "2077-04": "2020-07-16", "2077-05": "2020-08-17", "2077-06": "2020-09-17", "2077-07": "2020-10-17", "2077-08": "2020-11-16", "2077-09": "2020-12-16", "2077-10": "2021-01-14", "2077-11": "2021-02-13", "2077-12": "2021-03-14", "2078-01": "2021-04-14", "2078-02": "2021-05-15", "2078-03": "2021-06-15", "2078-04": "2021-07-16", "2078-05": "2021-08-17", "2078-06": "2021-09-17", "2078-07": "2021-10-18", "2078-08": "2021-11-17", "2078-09": "2021-12-16", "2078-10": "2022-01-15", "2078-11": "2022-02-13", "2078-12": "2022-03-15", "2079-01": "2022-04-14", "2079-02": "2022-05-15", "2079-03": "2022-06-15", "2079-04": "2022-07-17", "2079-05": "2022-08-17", "2079-06": "2022-09-17", "2079-07": "2022-10-18", "2079-08": "2022-11-17", "2079-09": "2022-12-16", "2079-10": "2023-01-15", "2079-11": "2023-02-13", "2079-12": "2023-03-15", "2080-01": "2023-04-14", "2080-02": "2023-05-15", "2080-03": "2023-06-16", "2080-04": "2023-07-17", "2080-05": "2023-08-18", "2080-06": "2023-09-18", "2080-07": "2023-10-18", "2080-08": "2023-11-17", "2080-09": "2023-12-17", "2080-10": "2024-01-15", "2080-11": "2024-02-13", "2080-12": "2024-03-14", "2081-01": "2024-04-13", "2081-02": "2024-05-14", "2081-03": "2024-06-15", "2081-04": "2024-07-16", "2081-05": "2024-08-17", "2081-06": "2024-09-17", "2081-07": "2024-10-17", "2081-08": "2024-11-16", "2081-09": "2024-12-16", "2081-10": "2025-01-14"}

def kalman_impute(series):
    """Fill NaNs with the Kalman smoother of a local-linear-trend + stochastic seasonal (12) structural model."""
    mod = sm.tsa.UnobservedComponents(series.values.astype(float), level="local linear trend", seasonal=12, stochastic_seasonal=True)
    res = mod.fit(disp=False, maxiter=2000)
    smoothed = res.smoother_results.smoothed_forecasts[0]
    filled = series.copy(); miss = series.isna()
    filled[miss] = smoothed[miss.values]
    return filled, miss, res

def to_gregorian(nepali, starts):
    """Day-weighted conversion of Nepali-month totals (indexed by BS 'YYYY-MM') to Gregorian calendar months."""
    keys = sorted(starts)
    parts = []
    for k0, k1 in zip(keys[:-1], keys[1:]):
        if k0 in nepali.index:
            a, b = pd.Timestamp(starts[k0]), pd.Timestamp(starts[k1])
            parts.append(pd.Series(nepali[k0] / (b - a).days, index=pd.date_range(a, b - pd.Timedelta(days=1), freq="D")))
    return pd.concat(parts).resample("MS").sum()

if INPUT_CALENDAR == "BS":
    # BS month of each label: the Nepali month that starts within that Gregorian month
    label_to_bs = {}
    for k, d in BS_START.items():
        d = pd.Timestamp(d); label_to_bs[pd.Timestamp(d.year, d.month, 1)] = k
    labels = pd.date_range("2017-12-01", "2024-12-01", freq="MS")
    reported = pd.Series([DATA[t.year][t.month - 1] if (t.year >= 2018 and t.year in DATA) else np.nan for t in labels],
                         index=[label_to_bs[t] for t in labels], dtype=float)
    nepali_full, imputed_mask, kalman_res = kalman_impute(reported)
    greg = to_gregorian(nepali_full, BS_START)["2018-01-01":"2024-12-01"]
    OBSERVED = {y: [int(round(v)) for v in greg[str(y)].values] for y in range(2018, 2025)}
    # share of each Gregorian month's days that come from imputed Nepali months
    imp_days = to_gregorian(pd.Series(np.where(imputed_mask, 1.0, 0.0), index=reported.index) *
                            pd.Series({k: (pd.Timestamp(BS_START[n]) - pd.Timestamp(BS_START[k])).days
                                       for k, n in zip(sorted(BS_START)[:-1], sorted(BS_START)[1:]) if k in reported.index}), BS_START)
    IMPUTED_SHARE = (imp_days["2018-01-01":"2024-12-01"] / greg.index.days_in_month).round(3)
    preprocessing = pd.DataFrame({
        "Nepali month": [f"{BS_MONTHS[int(k[5:]) - 1]} {k[:4]}" for k in reported.index], "BS year-month": reported.index,
        "Start date": [BS_START[k] for k in reported.index],
        "End date": [(pd.Timestamp(BS_START[n]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d") for n in
                     [sorted(BS_START)[sorted(BS_START).index(k) + 1] for k in reported.index]],
        "Label in input file": [t.strftime("%Y-%m") for t in labels],
        "Reported notifications": reported.values, "Imputed (Kalman smoother)": imputed_mask.values,
        "Value used": nepali_full.round(1).values})
    print(f"Kalman smoother (log-likelihood {kalman_res.llf:.1f}, converged {kalman_res.mle_retvals.get('converged')}): "
          f"imputed {int(imputed_mask.sum())} Nepali month(s): " +
          ", ".join(f"{r['Nepali month']} = {r['Value used']:.0f}" for _, r in preprocessing[preprocessing['Imputed (Kalman smoother)']].iterrows()))
    print(f"Reported 2018-2024 Nepali-month total {np.nansum(reported.values):,.0f}; Gregorian Jan 2018-Dec 2024 total {sum(map(sum, OBSERVED.values())):,}")
    display(preprocessing)
else:
    OBSERVED = {y: [int(round(v)) for v in DATA[y]] for y in range(2018, 2025)}
    IMPUTED_SHARE = pd.Series(0.0, index=pd.date_range("2018-01-01", "2024-12-01", freq="MS"))
    preprocessing = pd.DataFrame({"Note": ["Input already in Gregorian calendar months; no conversion or imputation was needed."]})
check_rows = []
for y in range(2015, 2025):
    ref = BACKCAST[y] if y < 2018 else PUBLISHED[y]
    src = DATA.get(y)
    diff = None if src is None else int(np.abs(np.array(src) - np.array(ref)).max())
    check_rows.append({"Year": y, "In input file": src is not None,
                       "Reference": "CAGR back-cast" if y < 2018 else "published (S1 Table)",
                       "Max abs difference": diff, "Matches": (diff == 0) if diff is not None else None})
data_check = save_table(pd.DataFrame(check_rows), "Data_check_input_vs_published")
display(data_check)

# Back-cast years: use the file's values if present (warn if they differ from the CAGR back-cast), otherwise back-cast
WIDE = {}
BACKCAST_FINAL = backcast(OBSERVED)          # back-cast from the final (Gregorian) 2018-2024 series
for y in range(2015, 2018):
    WIDE[y] = [int(round(v)) for v in DATA[y]] if (y in DATA and INPUT_CALENDAR == "AD") else BACKCAST_FINAL[y]
WIDE.update(OBSERVED)
DATA_MATCHES_PUBLISHED = bool(data_check["Matches"].fillna(True).all())
if DATA_MATCHES_PUBLISHED:
    print("Input data identical to the published dataset (reported Nepali-month values).")
else:
    print("WARNING: input data differ from the published dataset (see Data_check_input_vs_published.csv). "
          "All analyses will run on the input data, but results will not match the manuscript.")

wide = pd.DataFrame(WIDE, index=MONTHS)                       # months x years
y_all = pd.Series(np.concatenate([WIDE[y] for y in sorted(WIDE)]).astype(float),
                  index=pd.date_range("2015-01-01", periods=120, freq="MS"), name="TB notifications")
OBSERVED_START = pd.Timestamp("2018-01-01")
assert len(y_all) == 120 and y_all.isna().sum() == 0

# Cache folder specific to this run mode AND this dataset
DATA_HASH = hashlib.sha256(y_all.values.astype(np.int64).tobytes()).hexdigest()[:10]
CACHE_DIR = OUTPUT_DIR / "_cache" / f"{MODE}_{DATA_HASH}"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Cache: {CACHE_DIR}")

s1 = pd.DataFrame({"Date": y_all.index.strftime("%Y-%m"), "Year": y_all.index.year, "Month": y_all.index.month_name(),
                   "TB notifications": y_all.values.astype(int),
                   "Data type": np.where(y_all.index < OBSERVED_START, "Back-cast (month-specific CAGR)",
                                         "Observed (NTCC), converted to Gregorian month" if INPUT_CALENDAR == "BS" else "Observed (NTCC)"),
                   "Share of days from imputed Nepali months": IMPUTED_SHARE.reindex(y_all.index).values})
save_table(s1, "S1_dataset")
print(wide)

# %% [markdown]
# ## 3. Model and statistics library
# All model code is in this section; later sections only call these functions.
#
# **Conventions.** Scalers are fitted on the training window only. Multi-step forecasts are recursive (12 months from the forecast origin, no refitting). SARIMA training metrics exclude the 13 start-up values.

# %% [markdown]
# ### 3.1 Model settings (as described in Sections 2.6–2.10 and 2.12)

# %%
SARIMA_ORDER, SARIMA_SORDER = (1, 1, 1), (1, 1, 1, 12)   # benchmark SARIMA
HYB_ORDER,    HYB_SORDER    = (1, 1, 1), (1, 1, 0, 12)   # linear component of the hybrid
CNN_BASE = dict(look_back=12, filters=(128, 64), kernel=3, dropout=0.2, dense=64,
                lr=0.001, epochs=300, batch=24, l2=0.0, early_stop=False)        # stand-alone CNNAR
HYB_CNN  = dict(CNN_BASE, epochs=500, batch=16)                                  # residual CNNAR of the hybrid
CNN_REG  = dict(look_back=12, filters=(32, 16), kernel=3, dropout=0.3, dense=16,
                lr=0.001, epochs=300, batch=24, l2=1e-3, early_stop=True)        # regularised variant
HYB_REG  = dict(CNN_REG, epochs=500, batch=16)
LSTM_BASE = dict(look_back=16, units=64, dropout=0.2, dense=32, lr=0.001, epochs=1000, batch=16, patience=40)
PROPHET_BASE = dict(changepoint_prior_scale=0.05, seasonality_prior_scale=10, seasonality_mode="multiplicative",
                    yearly_fourier=10, monthly_fourier=5)
XGB_BASE = dict(look_back=12, n_estimators=1000, max_depth=2, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8)
MODEL_ORDER = ["SARIMA", "SARIMAX-breaks", "CNNAR", "LSTM", "Prophet", "XGBoost",
               "Hybrid SARIMA-CNNAR", "Hybrid SARIMAX(breaks)-CNNAR"]
MAIN = "Hybrid SARIMA-CNNAR"

# %% [markdown]
# ### 3.2 General utilities

# %%
def set_seed(seed):
    """Fix the Python, NumPy and TensorFlow random generators."""
    np.random.seed(seed); random.seed(seed); tf.keras.utils.set_random_seed(seed)

def metrics(actual, pred):
    """MAE, RMSE, MAPE (%) and R² of a forecast."""
    a = np.asarray(actual, float); p = np.asarray(pred, float)
    return {"MAE": mean_absolute_error(a, p), "RMSE": float(np.sqrt(mean_squared_error(a, p))),
            "MAPE (%)": float(np.mean(np.abs((a - p) / a)) * 100), "R²": r2_score(a, p)}

def future_index(train, h=H):
    """Monthly dates of the h months after the training window."""
    return pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")

def fold_split(year, series=None):
    """Expanding-window split: train = Jan 2015..Dec (year-1), test = the 12 months of `year`."""
    series = y_all if series is None else series
    return series.loc[:pd.Timestamp(f"{year - 1}-12-01")], series.loc[f"{year}-01-01":f"{year}-12-01"]

def make_supervised(arr, look_back):
    """Sliding windows of `look_back` values (inputs) and the next value (target)."""
    arr = np.asarray(arr, float).ravel()
    X = np.array([arr[i:i + look_back] for i in range(len(arr) - look_back)])
    return X[..., None].astype("float32"), arr[look_back:].astype("float32")

def recursive_forecast(model, last_window, h, training=False):
    """h-step forecast by feeding each one-step forecast back as input (training=True keeps dropout on)."""
    seq = np.asarray(last_window, "float32").reshape(1, -1, 1); out = []
    for _ in range(h):
        v = float(model(seq, training=training).numpy()[0, 0]); out.append(v)
        seq = np.concatenate([seq[:, 1:, :], np.array([[[v]]], "float32")], axis=1)
    return np.array(out)

def fmt(mean, sd, nd=2):
    """'mean ± SD' (or 'mean' for a single run)."""
    return f"{mean:.{nd}f}" if (sd is None or np.isnan(sd) or sd == 0) else f"{mean:.{nd}f} ± {sd:.{nd}f}"

# %% [markdown]
# ### 3.3 Intervention variables (structural breaks and lockdown indicator)

# %%
def step_dummies(index, break_dates):
    """Level-shift dummies: 1 from the first month of each new regime onwards (Section 2.9)."""
    if not break_dates:
        return None
    return pd.DataFrame({f"Break_{d:%Y_%m}": (index >= d).astype(float) for d in break_dates}, index=index)

_LD_DAYS = pd.DatetimeIndex(np.concatenate([pd.date_range(a, b, freq="D") for a, b in LOCKDOWN_PERIODS]))

def lockdown_share(index):
    """Share of the days of each month under COVID-19 restrictions (LOCKDOWN_PERIODS)."""
    return np.array([((_LD_DAYS >= m) & (_LD_DAYS < m + pd.offsets.MonthBegin(1))).sum() / m.days_in_month for m in index])

def lockdown_dummy(index, future=False):
    """A-priori lockdown regressor (share of days under restriction); always 0 over a forecast horizon."""
    v = np.zeros(len(index)) if future else lockdown_share(index)
    return pd.DataFrame({"Lockdown": v}, index=index)

def exog_for(train, test_index, kind, breaks):
    """Exogenous regressors for training and forecast periods.
    kind: 'steps' (Bai-Perron step dummies), 'lockdown', or 'steps+lockdown'.
    The lockdown indicator is omitted when the training window has no lockdown month."""
    tr, te = [], []
    if "steps" in kind and breaks:
        tr.append(step_dummies(train.index, breaks)); te.append(step_dummies(test_index, breaks))
    if "lockdown" in kind and lockdown_dummy(train.index)["Lockdown"].sum() > 0:
        tr.append(lockdown_dummy(train.index)); te.append(lockdown_dummy(test_index, future=True))
    return (pd.concat(tr, axis=1), pd.concat(te, axis=1)) if tr else (None, None)

# %% [markdown]
# ### 3.4 SARIMA / SARIMAX
# Maximum likelihood with stationarity and invertibility constraints relaxed (up to 500 iterations). If that likelihood is **degenerate** (innovation variance ≈ 0 or an ARMA coefficient larger than 10 in absolute value), the model is refitted with the constraints enforced. Every fit is logged in `FIT_LOG`.

# %%
FIT_LOG = []

def fit_sarimax(y, exog=None, order=SARIMA_ORDER, sorder=SARIMA_SORDER):
    """Fit SARIMA(X); refit with constraints if the unconstrained likelihood is degenerate."""
    kw = dict(exog=exog, order=order, seasonal_order=sorder)
    res = SARIMAX(y, enforce_stationarity=False, enforce_invertibility=False, **kw).fit(disp=False, maxiter=500)
    exog_cols = list(exog.columns) if exog is not None else []
    arma = res.params.drop(["sigma2"] + [c for c in res.params.index if c in exog_cols])
    degenerate = (res.params["sigma2"] < 1e-6 * np.var(np.diff(np.asarray(y, float)))) or (np.abs(arma).max() > 10)
    if degenerate:
        res = SARIMAX(y, enforce_stationarity=True, enforce_invertibility=True, **kw).fit(disp=False, maxiter=500)
    FIT_LOG.append({"Train end": y.index[-1].strftime("%Y-%m"), "Order": f"{order}{sorder}", "Exog": ", ".join(exog_cols),
                    "Degenerate unconstrained fit": bool(degenerate), "Refitted with constraints": bool(degenerate),
                    "Converged": bool(res.mle_retvals.get("converged", True)), "sigma2": float(res.params["sigma2"])})
    return res

def run_sarima(train, h=H, order=SARIMA_ORDER, sorder=SARIMA_SORDER, exog_train=None, exog_future=None):
    """Fit on `train` and forecast h months. Returns forecast, 95% interval, fitted values (after start-up) and residuals."""
    res = fit_sarimax(train, exog_train, order, sorder)
    fc = res.get_forecast(h, exog=exog_future); ci = fc.conf_int(alpha=ALPHA)
    return dict(pred=fc.predicted_mean.values, lower=ci.iloc[:, 0].values, upper=ci.iloc[:, 1].values,
                fitted=res.fittedvalues.iloc[BURN_IN:], resid_all=train - res.fittedvalues,
                res=res, n_params=len(res.params))

# %% [markdown]
# ### 3.5 CNNAR and hybrid SARIMA–CNNAR
# Hybrid forecast = SARIMA(X) forecast + recursive CNNAR forecast of the SARIMA residuals.
# Model-based hybrid interval = hybrid forecast ± (SARIMA 95% half-width + 1.96 × SD of the SARIMA residuals after the start-up values).

# %%
def build_cnn(look_back, filters, kernel, dropout, dense, lr, l2, **_):
    """Two Conv1D layers (ReLU) with dropout, flatten, dense ReLU layer, linear output (Fig 1)."""
    reg = regularizers.l2(l2) if l2 > 0 else None
    m = Sequential([Input(shape=(look_back, 1)),
                    Conv1D(filters[0], kernel, activation="relu", kernel_regularizer=reg), Dropout(dropout),
                    Conv1D(filters[1], kernel, activation="relu", kernel_regularizer=reg), Dropout(dropout),
                    Flatten(), Dense(dense, activation="relu", kernel_regularizer=reg), Dense(1)])
    m.compile(optimizer=Adam(lr), loss="mse")
    return m

def fit_keras(model, X, y, epochs, batch, early_stop=False, patience=30, val_size=12, track_val=False):
    """Train a Keras model; with early stopping (or track_val) the last `val_size` windows are a chronological validation set."""
    if early_stop or track_val:
        cb = [EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)] if early_stop else []
        return model.fit(X[:-val_size], y[:-val_size], validation_data=(X[-val_size:], y[-val_size:]),
                         epochs=ep(epochs), batch_size=batch, verbose=0, callbacks=cb)
    return model.fit(X, y, epochs=ep(epochs), batch_size=batch, verbose=0)

def run_cnnar(train, h=H, seed=0, mc=False, track_val=False, **cfg):
    """Stand-alone CNNAR. mc=True adds a Monte-Carlo-dropout 95% interval."""
    cfg = {**CNN_BASE, **cfg}; set_seed(seed); lb = cfg["look_back"]
    scaler = MinMaxScaler().fit(train.values.reshape(-1, 1))
    z = scaler.transform(train.values.reshape(-1, 1)).ravel()
    X, y = make_supervised(z, lb)
    model = build_cnn(**cfg)
    hist = fit_keras(model, X, y, cfg["epochs"], cfg["batch"], cfg["early_stop"], track_val=track_val)
    inv = lambda v: scaler.inverse_transform(np.asarray(v).reshape(-1, 1)).ravel()
    out = dict(pred=inv(recursive_forecast(model, z[-lb:], h)),
               fitted=pd.Series(inv(model(X, training=False).numpy()), index=train.index[lb:]),
               n_params=model.count_params(), history=hist.history, n_samples=len(X), lower=None, upper=None)
    if mc:
        sims = np.array([recursive_forecast(model, z[-lb:], h, training=True) for _ in range(MC_SAMPLES)])
        out["lower"] = inv(np.percentile(sims, 100 * ALPHA / 2, axis=0))
        out["upper"] = inv(np.percentile(sims, 100 * (1 - ALPHA / 2), axis=0))
    return out

def run_hybrid(train, h=H, seed=0, order=HYB_ORDER, sorder=HYB_SORDER, exog_train=None, exog_future=None, cnn_cfg=None):
    """Hybrid SARIMA(X)-CNNAR: CNNAR trained on the SARIMA residuals after the 13 start-up values."""
    cfg = {**HYB_CNN, **(cnn_cfg or {})}; lb = cfg["look_back"]
    s = run_sarima(train, h, order, sorder, exog_train, exog_future)
    resid = s["resid_all"].iloc[BURN_IN:]
    set_seed(seed)
    scaler = MinMaxScaler().fit(resid.values.reshape(-1, 1))
    z = scaler.transform(resid.values.reshape(-1, 1)).ravel()
    X, y = make_supervised(z, lb)
    model = build_cnn(**cfg)
    fit_keras(model, X, y, cfg["epochs"], cfg["batch"], cfg["early_stop"])
    inv = lambda v: scaler.inverse_transform(np.asarray(v).reshape(-1, 1)).ravel()
    fitted = s["res"].fittedvalues.loc[resid.index[lb:]] + inv(model(X, training=False).numpy())
    resid_fc = inv(recursive_forecast(model, z[-lb:], h))
    half = (s["upper"] - s["lower"]) / 2 + 1.96 * np.std(resid.values)
    pred = s["pred"] + resid_fc
    return dict(pred=pred, fitted=fitted, sarima_pred=s["pred"], resid_fc=resid_fc, half=half,
                lower=pred - half, upper=pred + half, n_params=model.count_params() + s["n_params"], n_samples=len(X))

# %% [markdown]
# ### 3.6 Benchmark models: LSTM, Prophet, XGBoost

# %%
def run_lstm(train, h=H, seed=0, mc=False, **cfg):
    """LSTM(64) → dropout → dense(32) → output; early stopping on the last 10% of windows."""
    cfg = {**LSTM_BASE, **cfg}; set_seed(seed); lb = cfg["look_back"]
    scaler = MinMaxScaler().fit(train.values.reshape(-1, 1))
    z = scaler.transform(train.values.reshape(-1, 1)).ravel()
    X, y = make_supervised(z, lb)
    model = Sequential([Input(shape=(lb, 1)), LSTM(cfg["units"], activation="tanh"),
                        Dropout(cfg["dropout"]), Dense(cfg["dense"], activation="relu"), Dense(1)])
    model.compile(optimizer=Adam(cfg["lr"]), loss="mse")
    model.fit(X, y, epochs=ep(cfg["epochs"]), batch_size=cfg["batch"], validation_split=0.1, verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", patience=cfg["patience"], restore_best_weights=True)])
    inv = lambda v: scaler.inverse_transform(np.asarray(v).reshape(-1, 1)).ravel()
    out = dict(pred=inv(recursive_forecast(model, z[-lb:], h)),
               fitted=pd.Series(inv(model(X, training=False).numpy()), index=train.index[lb:]),
               n_params=model.count_params(), n_samples=len(X), lower=None, upper=None)
    if mc:
        sims = np.array([recursive_forecast(model, z[-lb:], h, training=True) for _ in range(MC_SAMPLES)])
        out["lower"] = inv(np.percentile(sims, 100 * ALPHA / 2, axis=0))
        out["upper"] = inv(np.percentile(sims, 100 * (1 - ALPHA / 2), axis=0))
    return out

def run_prophet(train, h=H, **cfg):
    """Prophet with yearly and 30.5-day Fourier seasonality and a COVID-19 period effect (Mar 2020-Dec 2022)."""
    cfg = {**PROPHET_BASE, **cfg}
    covid = pd.DataFrame({"holiday": "covid", "ds": pd.date_range("2020-03-01", "2022-12-01", freq="MS"),
                          "lower_window": 0, "upper_window": 0})
    m = Prophet(growth="linear", seasonality_mode=cfg["seasonality_mode"], changepoint_prior_scale=cfg["changepoint_prior_scale"],
                seasonality_prior_scale=cfg["seasonality_prior_scale"], holidays=covid, yearly_seasonality=False,
                weekly_seasonality=False, daily_seasonality=False, interval_width=1 - ALPHA)
    m.add_seasonality(name="yearly", period=365.25, fourier_order=cfg["yearly_fourier"])
    m.add_seasonality(name="monthly", period=30.5, fourier_order=cfg["monthly_fourier"])
    m.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    fc = m.predict(pd.DataFrame({"ds": list(train.index) + list(future_index(train, h))})); n = len(train)
    return dict(pred=fc["yhat"].values[n:], lower=fc["yhat_lower"].values[n:], upper=fc["yhat_upper"].values[n:],
                fitted=pd.Series(fc["yhat"].values[:n], index=train.index), n_params=None)

def run_xgb(train, h=H, seed=42, **cfg):
    """XGBoost on the 12 previous (scaled) values; recursive forecast."""
    cfg = {**XGB_BASE, **cfg}; lb = cfg.pop("look_back")
    scaler = MinMaxScaler().fit(train.values.reshape(-1, 1))
    z = scaler.transform(train.values.reshape(-1, 1)).ravel()
    X = np.array([z[i - lb:i] for i in range(lb, len(z))]); y = z[lb:]
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=seed, **cfg).fit(X, y)
    inv = lambda v: scaler.inverse_transform(np.asarray(v).reshape(-1, 1)).ravel()
    hist = list(z[-lb:]); preds = []
    for _ in range(h):
        v = float(model.predict(np.array(hist[-lb:]).reshape(1, lb))[0]); preds.append(v); hist.append(v)
    return dict(pred=inv(preds), fitted=pd.Series(inv(model.predict(X)), index=train.index[lb:]),
                lower=None, upper=None, n_params=None)

# %% [markdown]
# ### 3.7 Bai–Perron structural-break test
# Regime-wise intercept and trend on the STL-seasonally adjusted series; dynamic programming; minimum segment 15% of the sample; number of breaks chosen by BIC. Break date = first month of the new regime.

# %%
def bai_perron(series, max_breaks=5, trim=0.15):
    s = series.astype(float) - STL(series.astype(float), period=12, robust=True).fit().seasonal
    yv = s.values; n = len(yv); h_min = max(int(np.floor(trim * n)), 6)
    X = np.column_stack([np.ones(n), np.arange(n, dtype=float)]); k = X.shape[1]
    algo = rpt.Dynp(model="linear", min_size=h_min, jump=1).fit(np.column_stack([yv, X]))
    def ssr_of(bkps):
        tot, start = 0.0, 0
        for b in bkps:
            beta = np.linalg.lstsq(X[start:b], yv[start:b], rcond=None)[0]
            tot += float(np.sum((yv[start:b] - X[start:b] @ beta) ** 2)); start = b
        return tot
    rows = []
    for m in range(0, max_breaks + 1):
        try:
            bk = [n] if m == 0 else algo.predict(n_bkps=m)
        except Exception:
            break
        ssr = ssr_of(bk); p = (m + 1) * k + m
        rows.append({"n_breaks": m, "break_index": bk[:-1], "break_dates": [s.index[b].strftime("%Y-%m") for b in bk[:-1]],
                     "SSR": ssr, "BIC": n * np.log(ssr / n) + p * np.log(n)})
    tab = pd.DataFrame(rows); best = tab.loc[tab["BIC"].idxmin()]; m = int(best["n_breaks"])
    q = m * (k + 1); dfd = n - ((m + 1) * k + m)
    F = ((tab.loc[0, "SSR"] - best["SSR"]) / q) / (best["SSR"] / dfd) if m > 0 else np.nan
    return dict(table=tab, n_breaks=m, break_index=list(best["break_index"]),
                break_dates=[s.index[b] for b in best["break_index"]], F=F,
                p_nominal=stats.f.sf(F, q, dfd) if m > 0 else np.nan, adjusted=s, n=n, min_size=h_min)

# %% [markdown]
# ### 3.8 Forecast comparison and uncertainty

# %%
def dm_test(e1, e2, h=1, power=2):
    """Diebold-Mariano test with Harvey-Leybourne-Newbold correction (Student-t, df = n-1).
    power=2: squared-error loss; power=1: absolute-error loss. Negative statistic = model 1 better."""
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    d = np.abs(e1) ** power - np.abs(e2) ** power; T = len(d); dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[:T - k] - dbar)) / T for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / T
    if var <= 0:
        return np.nan, np.nan
    stat = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T) * dbar / np.sqrt(var)
    return stat, 2 * stats.t.sf(abs(stat), df=T - 1)

def block_bootstrap_ci(actual, pred, n_boot=None, block=3, seed=0):
    """95% circular block-bootstrap CIs (block = 3 months) of MAE, RMSE, MAPE and R²."""
    n_boot = n_boot or N_BOOT
    a = np.asarray(actual, float); p = np.asarray(pred, float); n = len(a)
    rng = np.random.default_rng(seed); nb = int(np.ceil(n / block)); draws = []
    for _ in range(n_boot):
        idx = np.concatenate([(st + np.arange(block)) % n for st in rng.integers(0, n, nb)])[:n]
        draws.append(metrics(a[idx], p[idx]))
    d = pd.DataFrame(draws)
    return d.quantile(0.025), d.quantile(0.975)

def residual_diagnostics(res, lag=12):
    """Shapiro-Wilk, Ljung-Box (lag, at most n-1), ARCH LM and mean residual."""
    res = pd.Series(np.asarray(res, float)).dropna(); lag = min(lag, len(res) - 1)
    lb = acorr_ljungbox(res, lags=[lag], return_df=True); arch = het_arch(res); sw = stats.shapiro(res)
    return pd.DataFrame({"Statistic": [sw.statistic, lb.iloc[0, 0], arch[0], res.mean()],
                         "p-value": [sw.pvalue, lb.iloc[0, 1], arch[1], np.nan]},
                        index=["Shapiro-Wilk", f"Ljung-Box (lag {lag})", "ARCH LM", "Mean residual"])

print("Library ready.")

# %% [markdown]
# ## 4. Exploratory analysis (Table 1; Figs 4–10, 13–17)

# %%
if RUN["eda"]:
    # Table 1 - annual descriptive statistics
    table1 = wide.describe().T.rename(columns={"25%": "Q1", "50%": "Median", "75%": "Q3"}).round(1)
    table1.index.name = "Year"; save_table(table1, "Table01_descriptives", index=True); display(table1)

    annual = y_all.groupby(y_all.index.year).sum()
    growth = pd.DataFrame([
        {"Comparison": "2018 -> 2024 (observed)", "Change (%)": (annual[2024] / annual[2018] - 1) * 100},
        {"Comparison": "2019 -> 2024 (pre-COVID observed)", "Change (%)": (annual[2024] / annual[2019] - 1) * 100},
        {"Comparison": "2015 -> 2024 (back-cast baseline)", "Change (%)": (annual[2024] / annual[2015] - 1) * 100},
        {"Comparison": "2018 -> 2019", "Change (%)": (annual[2019] / annual[2018] - 1) * 100}])
    save_table(growth.round(1), "Growth"); display(growth.round(1))
    print("Observed-year monthly means:", y_all[OBSERVED_START:].groupby(y_all[OBSERVED_START:].index.month).mean().round(0).to_dict())

    # Normality of the raw series (Figs 5-6)
    yv = y_all.values
    norm = pd.DataFrame({"Statistic": [stats.shapiro(yv).statistic, stats.kstest(yv, stats.norm(yv.mean(), yv.std()).cdf).statistic,
                                       stats.jarque_bera(yv).statistic, stats.skew(yv)],
                         "p-value": [stats.shapiro(yv).pvalue, stats.kstest(yv, stats.norm(yv.mean(), yv.std()).cdf).pvalue,
                                     stats.jarque_bera(yv).pvalue, np.nan]},
                        index=["Shapiro-Wilk", "Kolmogorov-Smirnov", "Jarque-Bera", "Skewness"])
    save_table(norm, "Normality_tests", index=True); display(norm.round(4))

    # Fig 4 - series with LOESS
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axvspan(y_all.index[0], pd.Timestamp("2017-12-31"), color="grey", alpha=0.12, label="Back-cast years (2015–2017)")
    ax.plot(y_all.index, yv, "o-", color="#1f77b4", ms=4, label="Monthly TB notifications")
    ax.plot(y_all.index, lowess(yv, np.arange(120), frac=0.25, return_sorted=False), color="red", lw=2.2, label="LOESS smoothing (span 0.25)")
    ax.set(title="Monthly TB notifications in Nepal (2015–2024)", xlabel="Date", ylabel="Notifications per month")
    ax.grid(alpha=0.3); ax.legend(loc="upper left"); save_figure(fig, "Fig04_Monthly_TB_series")
    # Fig 5 - histogram, Fig 6 - Q-Q plot
    fig, ax = plt.subplots(figsize=(8, 5)); sns.histplot(yv, kde=True, ax=ax); ax.set_title("Histogram of monthly TB notifications")
    save_figure(fig, "Fig05_Histogram")
    fig, ax = plt.subplots(figsize=(6, 6)); stats.probplot(yv, dist="norm", plot=ax); ax.set_title("Q-Q plot"); save_figure(fig, "Fig06_QQ_plot")
    # Fig 7 - correlation of yearly profiles
    corr = wide.corr(); save_table(corr.round(3), "Correlation_years", index=True)
    fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation of monthly TB notification profiles between years"); save_figure(fig, "Fig07_Correlation_heatmap")
    # Fig 8 - within-year IQR screening (values are retained in the main analysis)
    def iqr_cap_by_year(series):
        s = series.astype(float).copy()
        for yr in sorted(set(s.index.year)):
            m = s.index.year == yr; q1, q3 = s[m].quantile([0.25, 0.75]); iqr = q3 - q1
            s[m] = s[m].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
        return s
    capped = iqr_cap_by_year(y_all)
    outliers = pd.DataFrame({"Raw": y_all, "Capped": capped})[y_all != capped]; display(outliers)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, s, t in [(axes[0], y_all, "Before capping"), (axes[1], capped, "After IQR capping")]:
        sns.boxplot(x=s.index.year, y=s.values, ax=ax); ax.set_title(t)
    save_figure(fig, "Fig08_Outlier_boxplots")
    # Fig 9 - ACF / PACF
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_acf(y_all, lags=36, ax=axes[0]); plot_pacf(y_all, lags=36, ax=axes[1], method="ywm")
    axes[0].set_title("(A) Autocorrelation function"); axes[1].set_title("(B) Partial autocorrelation function")
    save_figure(fig, "Fig09_ACF_PACF")
    # Fig 10 - classical additive decomposition
    dec = seasonal_decompose(y_all, model="additive", period=12)
    print(f"Decomposition residuals: {dec.resid.min():.0f} to {dec.resid.max():.0f}")
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for ax, comp, col, lab in zip(axes, [dec.observed, dec.trend, dec.seasonal, dec.resid], ["blue", "green", "red", "orange"],
                                  ["(A) Observed", "(B) Trend", "(C) Seasonal", "(D) Residual"]):
        ax.plot(comp, color=col); ax.set_title(lab)
    save_figure(fig, "Fig10_Decomposition")
    # Figs 13-17 - stationarity (ADF)
    adf_rows = []
    for name, s, fig_name, col in [("Original series", y_all, "Fig13_Original_series", "C0"),
                                   ("First difference", y_all.diff().dropna(), "Fig14_First_difference", "orange"),
                                   ("Seasonal difference (lag 12)", y_all.diff(12).dropna(), "Fig15_Seasonal_difference", "green"),
                                   ("Log, first difference", np.log(y_all).diff().dropna(), "Fig16_Log_difference", "red"),
                                   ("First and seasonal difference (d = 1, D = 1)", y_all.diff().diff(12).dropna(), None, "purple")]:
        r = adfuller(s); adf_rows.append({"Series": name, "ADF statistic": r[0], "p-value": r[1]})
        if fig_name is None:
            continue
        fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(s, marker="o", color=col); ax.set_title(f"{name} (ADF p = {r[1]:.3g})")
        ax.grid(alpha=0.3); save_figure(fig, fig_name)
    adf_tab = save_table(pd.DataFrame(adf_rows), "ADF_tests"); display(adf_tab)
    roll_sd = y_all.rolling(12).std(); print(f"Rolling 12-month SD: {roll_sd.min():.0f} to {roll_sd.max():.0f}")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_all, label="Original"); ax.plot(y_all.rolling(12).mean(), color="red", label="Rolling mean (12 months)")
    ax.plot(roll_sd, color="green", label="Rolling SD (12 months)"); ax.legend(); ax.grid(alpha=0.3); save_figure(fig, "Fig17_Rolling_mean_SD")

# %% [markdown]
# ## 5. Seasonality tests and structural breaks (Tables 2–3; Figs 11–12)

# %%
def qs_test(series):
    """QS test: positive autocorrelation at lags 12 and 24 of the first-differenced series (chi-square, 2 df)."""
    x = series.diff().dropna(); n = len(x); r = acf(x, nlags=24, fft=False)
    qs = n * (n + 2) * sum(max(0.0, r[k]) ** 2 / (n - k) for k in (12, 24))
    return qs, stats.chi2.sf(qs, 2)

def seasonality_tests(series, label):
    """Kruskal-Wallis and Friedman tests on STL-detrended values, QS test, STL seasonal/trend strength."""
    stl = STL(series, period=12, robust=True).fit(); T, S, R = stl.trend, stl.seasonal, stl.resid
    detr = series - T; month = detr.index.month
    kw = stats.kruskal(*[detr[month == m].values for m in range(1, 13)])
    piv = pd.DataFrame({"year": detr.index.year, "month": month, "v": detr.values}).pivot(index="year", columns="month", values="v")
    fr = stats.friedmanchisquare(*[piv[m].values for m in range(1, 13)]); qs, qs_p = qs_test(series)
    return pd.DataFrame({"Test": ["Kruskal-Wallis", "Friedman", "QS test", "STL seasonal strength F_S", "STL trend strength F_T"],
                         "Statistic": [kw.statistic, fr.statistic, qs, max(0, 1 - np.var(R) / np.var(S + R)), max(0, 1 - np.var(R) / np.var(T + R))],
                         "p-value": [kw.pvalue, fr.pvalue, qs_p, np.nan, np.nan], "Sample": label})

if RUN["tests_breaks"]:
    table2 = save_table(pd.concat([seasonality_tests(y_all, "2015-2024"), seasonality_tests(y_all[OBSERVED_START:], "2018-2024 (observed)")]),
                        "Table02_seasonality_tests")
    display(table2.round(4))
    obs = y_all[OBSERVED_START:]; detr_obs = obs - STL(obs, period=12, robust=True).fit().trend
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot([detr_obs[detr_obs.index.month == m].values for m in range(1, 13)]); ax.set_xticks(range(1, 13))
    ax.set_xticklabels([m[:3] for m in MONTHS]); ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set(title="Detrended TB notifications by calendar month (2018–2024)", ylabel="Deviation from STL trend (cases)")
    save_figure(fig, "Fig11_Seasonality_by_month")

    bp_full = bai_perron(y_all)
    table3 = bp_full["table"].assign(break_dates=lambda d: d["break_dates"].apply(", ".join),
                                     selected=lambda d: d["n_breaks"] == bp_full["n_breaks"]).drop(columns="break_index")
    save_table(table3, "Table03_bai_perron"); display(table3.round(1))
    print(f"BIC-selected breaks: {[d.strftime('%b %Y') for d in bp_full['break_dates']]}; F = {bp_full['F']:.2f} (nominal p = {bp_full['p_nominal']:.2g})")
    adj = bp_full["adjusted"]; tt = np.arange(120, dtype=float); segs = [0] + bp_full["break_index"] + [120]; fit = np.zeros(120)
    for a, b in zip(segs[:-1], segs[1:]):
        Xs = np.column_stack([np.ones(b - a), tt[a:b]]); fit[a:b] = Xs @ np.linalg.lstsq(Xs, adj.values[a:b], rcond=None)[0]
        print(f"Regime {adj.index[a]:%b %Y}-{adj.index[b - 1]:%b %Y}: mean {y_all.iloc[a:b].mean():.0f}, trend {np.polyfit(tt[a:b], adj.values[a:b], 1)[0]:.1f} cases/month")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_all, color="lightgrey", label="Observed"); ax.plot(adj, color="steelblue", marker="o", ms=3, label="Seasonally adjusted")
    ax.plot(adj.index, fit, color="red", lw=2, label="Regime-wise linear fit")
    for i, d in enumerate(bp_full["break_dates"]):
        ax.axvline(d, color="green", ls="--", lw=2, label="Bai–Perron break" if i == 0 else None)
    ax.axvspan(y_all.index[0], pd.Timestamp("2017-12-01"), color="orange", alpha=0.08, label="Back-cast years")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("Bai–Perron structural breaks (BIC-selected)"); save_figure(fig, "Fig12_BaiPerron_breaks")

# %% [markdown]
# ## 6. SARIMA: grid search, benchmark model and residual tests (Tables 4–6; Fig 18; S2 Table)

# %%
def sarima_grid(series):
    """All 729 SARIMA(p,d,q)(P,D,Q)12 models with orders in {0,1,2} (default optimiser settings)."""
    rows = []
    for o in [(p, d, q) for p in range(3) for d in range(3) for q in range(3)]:
        for so in [(P, D, Q, 12) for P in range(3) for D in range(3) for Q in range(3)]:
            try:
                r = SARIMAX(series, order=o, seasonal_order=so, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                rows.append({"Order (p,d,q)": str(o), "Seasonal (P,D,Q,12)": str(so), "AIC": r.aic, "BIC": r.bic, "HQIC": r.hqic, "Log-Likelihood": r.llf})
            except Exception:
                pass
    return pd.DataFrame(rows)

if RUN["grid_search"]:
    grid = cached("sarima_grid", lambda: sarima_grid(y_all))
    grid["Degenerate fit"] = grid["Log-Likelihood"] > -100
    grid_d1D1 = (grid[grid["Order (p,d,q)"].str.match(r"^\(\d, 1, \d\)$") & grid["Seasonal (P,D,Q,12)"].str.match(r"^\(\d, 1, \d, 12\)$")]
                 .sort_values("AIC").reset_index(drop=True))
    grid_d1D1.insert(0, "AIC rank (d=1, D=1)", range(1, len(grid_d1D1) + 1))
    save_table(grid, "S2_grid_all"); save_table(grid_d1D1, "S2_grid_d1D1")
    rank = lambda o, s: int(grid_d1D1.query("`Order (p,d,q)` == @o and `Seasonal (P,D,Q,12)` == @s")["AIC rank (d=1, D=1)"].iloc[0])
    print(f"{len(grid)} models fitted, {int(grid['Degenerate fit'].sum())} degenerate (excluded).")
    print(f"Within d=1, D=1 ({len(grid_d1D1)} models): SARIMA(1,1,1)(1,1,1)12 rank {rank('(1, 1, 1)', '(1, 1, 1, 12)')}, "
          f"SARIMA(1,1,1)(1,1,0)12 rank {rank('(1, 1, 1)', '(1, 1, 0, 12)')}; AIC-best {grid_d1D1.iloc[0, 1]}{grid_d1D1.iloc[0, 2]}")

# %%
if RUN["sarima"]:
    train24, test24 = fold_split(2024)
    sar = run_sarima(train24)
    t5 = pd.DataFrame([metrics(train24.iloc[BURN_IN:], sar["fitted"]), metrics(test24, sar["pred"])], index=["Training (2015-2023)", "Test (2024)"])
    save_table(t5.round(2), "Table05_sarima_metrics", index=True); display(t5.round(2))
    print(f"Training-data AIC {sar['res'].aic:.1f}, BIC {sar['res'].bic:.1f}")
    # Table 6 - residual diagnostics (training residuals include the start-up values, as in the manuscript)
    t6 = pd.concat({"Train": residual_diagnostics(sar["resid_all"], 12), "Test": residual_diagnostics(test24.values - sar["pred"], 11)})
    save_table(t6, "Table06_sarima_residual_diagnostics", index=True); display(t6.round(6))
    # BDS test on SARIMA residuals (justification of the residual learner)
    bds_rows = []
    for label, o, so in [("SARIMA(1,1,1)(1,1,1)12", SARIMA_ORDER, SARIMA_SORDER), ("SARIMA(1,1,1)(1,1,0)12 (hybrid component)", HYB_ORDER, HYB_SORDER)]:
        r = run_sarima(train24, order=o, sorder=so)["resid_all"].iloc[BURN_IN:]; st_, pv = bds(r.values, max_dim=5)
        bds_rows += [{"Residuals of": label, "Dimension": d, "BDS": a, "p-value": b} for d, a, b in zip(range(2, 6), st_, pv)]
    display(save_table(pd.DataFrame(bds_rows), "BDS_test").round(4))

# %% [markdown]
# ## 7. Cross-validation of all models (Tables 8, 10, 12, 13; Fig 20; S3 Table)
# Five expanding-window folds (test years 2020–2024); fold 2024 is the primary hold-out split. Break dates are re-estimated within each fold from its training data. Neural models are trained with `N_SEEDS` seeds, and their fold forecast is the seed average.

# %%
def summarise_runs(outs, train, test):
    """Collect forecasts and metrics of one model in one fold (all seeds)."""
    preds = np.array([o["pred"] for o in outs]); mean_pred = preds.mean(axis=0)
    half = outs[0].get("half")
    return dict(pred=mean_pred, seed_preds=preds, actual=test.values, index=test.index,
                test_metrics=[metrics(test.values, p) for p in preds],
                train_metrics=[metrics(train.loc[o["fitted"].index].values, o["fitted"].values) for o in outs],
                lower=(mean_pred - half) if half is not None else outs[0].get("lower"),
                upper=(mean_pred + half) if half is not None else outs[0].get("upper"),
                n_params=outs[0].get("n_params"), n_samples=outs[0].get("n_samples"))

def cross_validate():
    results, breaks = {}, {}
    for yr in FOLD_YEARS:
        t0 = time.time(); train, test = fold_split(yr)
        breaks[yr] = bai_perron(train)["break_dates"]; xs = exog_for(train, test.index, "steps", breaks[yr])
        results[(yr, "SARIMA")] = summarise_runs([run_sarima(train)], train, test)
        results[(yr, "SARIMAX-breaks")] = summarise_runs([run_sarima(train, exog_train=xs[0], exog_future=xs[1])], train, test)
        results[(yr, "CNNAR")] = summarise_runs([run_cnnar(train, seed=s, mc=(s == 0)) for s in range(N_SEEDS)], train, test)
        results[(yr, "LSTM")] = summarise_runs([run_lstm(train, seed=s, mc=(s == 0)) for s in range(N_SEEDS)], train, test)
        results[(yr, "Prophet")] = summarise_runs([run_prophet(train)], train, test)
        results[(yr, "XGBoost")] = summarise_runs([run_xgb(train)], train, test)
        results[(yr, MAIN)] = summarise_runs([run_hybrid(train, seed=s) for s in range(N_SEEDS)], train, test)
        results[(yr, "Hybrid SARIMAX(breaks)-CNNAR")] = summarise_runs(
            [run_hybrid(train, seed=s, exog_train=xs[0], exog_future=xs[1]) for s in range(N_SEEDS)], train, test)
        print(f"fold {yr}: breaks {[d.strftime('%Y-%m') for d in breaks[yr]]} ({time.time() - t0:.0f}s)")
    return results, breaks

if RUN["cross_validation"]:
    REV, FOLD_BREAKS = cached("cross_validation", cross_validate)

    def errors(name, years, res=None):
        res = res or REV
        return np.concatenate([res[(y, name)]["actual"] - res[(y, name)]["pred"] for y in years])

    def pooled_metrics(name, years, res=None):
        res = res or REV
        return metrics(np.concatenate([res[(y, name)]["actual"] for y in years]), np.concatenate([res[(y, name)]["pred"] for y in years]))

    # Every fold x model x seed (S3 Table A)
    cv_long = pd.DataFrame([{"Fold (test year)": yr, "Model": name, "Seed": s, **{f"Train {k}": v for k, v in mtr.items()},
                             **{f"Test {k}": v for k, v in mte.items()}}
                            for (yr, name), r in REV.items() for s, (mtr, mte) in enumerate(zip(r["train_metrics"], r["test_metrics"]))])
    save_table(cv_long, "S3_cv_all_runs")
    fold_fc = pd.DataFrame([{"Fold (test year)": yr, "Model": name, "Date": d.strftime("%Y-%m"), "Actual": a, "Forecast": p}
                            for (yr, name), r in REV.items() for d, a, p in zip(r["index"], r["actual"], r["pred"])])
    save_table(fold_fc, "S3_fold_forecasts"); save_table(fold_fc, "Model_CV_fold_forecasts_all_models")

    # Table 12 / S6 Table C - 2024 hold-out comparison
    rows = []
    for name in MODEL_ORDER:
        r = REV[(2024, name)]; te = pd.DataFrame(r["test_metrics"]); avg = metrics(r["actual"], r["pred"])
        rows.append({"Model": name, "Runs": len(te), "MAPE of runs (mean ± SD)": fmt(te["MAPE (%)"].mean(), te["MAPE (%)"].std(ddof=1) if len(te) > 1 else np.nan),
                     **{f"{k} (seed-avg)": v for k, v in avg.items()}})
    hc = run_sarima(fold_split(2024)[0], order=HYB_ORDER, sorder=HYB_SORDER)
    rows.append({"Model": "Hybrid's SARIMA(1,1,1)(1,1,0) component", "Runs": 1, "MAPE of runs (mean ± SD)": "",
                 **{f"{k} (seed-avg)": v for k, v in metrics(fold_split(2024)[1], hc["pred"]).items()}})
    table12 = save_table(pd.DataFrame(rows).sort_values("MAPE (%) (seed-avg)"), "Table12_test2024")
    display(table12.round(2))

    # Table 13 - MAPE by fold and pooled
    per_fold = pd.DataFrame([{"Model": n, "Test year": y, **metrics(REV[(y, n)]["actual"], REV[(y, n)]["pred"])}
                             for n in MODEL_ORDER for y in FOLD_YEARS])
    save_table(per_fold, "S3_cv_per_fold")
    t13 = per_fold.pivot(index="Model", columns="Test year", values="MAPE (%)").loc[MODEL_ORDER]
    t13["Pooled MAPE (n=60)"] = [pooled_metrics(n, FOLD_YEARS)["MAPE (%)"] for n in MODEL_ORDER]
    t13["Pooled RMSE (n=60)"] = [pooled_metrics(n, FOLD_YEARS)["RMSE"] for n in MODEL_ORDER]
    t13["Pooled MAPE excl. 2020 (n=48)"] = [pooled_metrics(n, FOLD_YEARS[1:])["MAPE (%)"] for n in MODEL_ORDER]
    table13 = save_table(t13, "Table13_cross_validation", index=True); display(table13.round(2))

    # Fig 20 - all forecasts for 2024
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(y_all["2022":], "k-o", lw=2.5, label="Observed")
    for name in MODEL_ORDER:
        ax.plot(REV[(2024, name)]["index"], REV[(2024, name)]["pred"], ls="--", lw=2 if "Hybrid" in name else 1.2, label=name)
    ax.axvline(pd.Timestamp("2024-01-01"), color="grey", ls=":"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    ax.set(title="Forecasts of all models for 2024 (issued December 2023; neural models: mean of seeds)", ylabel="TB notifications")
    save_figure(fig, "Fig20_All_models_2024")

# %% [markdown]
# ## 8. Structural breaks as model variables (Table 11; S8 Table)

# %%
if RUN["tests_breaks"]:
    rows, coefs = [], []
    for label, series in [("Training 2015-2023", y_all[:"2023-12-01"]), ("Full 2015-2024", y_all)]:
        bp = bai_perron(series); base = run_sarima(series, h=1)["res"]
        for kind in ["steps", "lockdown", "steps+lockdown"]:
            xtr, xte = exog_for(series, future_index(series, 1), kind, bp["break_dates"])
            res = run_sarima(series, h=1, exog_train=xtr, exog_future=xte)["res"]; lr = 2 * (res.llf - base.llf)
            rows.append({"Sample": label, "Specification": kind, "Breaks": ", ".join(d.strftime("%Y-%m") for d in bp["break_dates"]),
                         "AIC SARIMA": base.aic, "AIC SARIMAX": res.aic, "BIC SARIMA": base.bic, "BIC SARIMAX": res.bic,
                         "LogLik SARIMA": base.llf, "LogLik SARIMAX": res.llf, "LR": lr, "df": xtr.shape[1], "LR p": stats.chi2.sf(max(lr, 0), xtr.shape[1])})
            coefs += [{"Sample": label, "Specification": kind, "Variable": c, "Coefficient": res.params[c], "SE": res.bse[c], "p": res.pvalues[c]}
                      for c in xtr.columns]
    table11 = save_table(pd.DataFrame(rows), "Table11_sarimax_interventions"); coef_tab = save_table(pd.DataFrame(coefs), "S8_intervention_coefficients")
    display(table11.round(3)); display(coef_tab.round(3))

# %% [markdown]
# ## 9. Over-fitting, regularisation, ablation and lockdown indicator in all folds (Table 15; Fig 21; S9 Table)
# * Regularised CNNAR and hybrids (3,761 parameters, L2, dropout 0.3, early stopping) in every fold.
# * **Ablation:** each hybrid against its own linear component, which shows what the CNNAR residual correction adds.
# * **Lockdown indicator** (Section 3.3) in SARIMAX and in the hybrid, alone and together with the Bai–Perron step dummies.

# %%
COMP, COMP_B = "SARIMA(1,1,1)(1,1,0) [hybrid component]", "SARIMAX(1,1,1)(1,1,0)-breaks [break-hybrid component]"
EXT_MODELS = [COMP, COMP_B, "CNNAR (regularised)", "Hybrid SARIMA-CNNAR (regularised)", "Hybrid SARIMAX(breaks)-CNNAR (regularised)",
              "SARIMAX-lockdown", "SARIMAX-breaks+lockdown", "Hybrid SARIMAX(lockdown)-CNNAR", "Hybrid SARIMAX(breaks+lockdown)-CNNAR"]

def extended_cv():
    res = {}
    for yr in FOLD_YEARS:
        t0 = time.time(); train, test = fold_split(yr); br = FOLD_BREAKS[yr]; xs = exog_for(train, test.index, "steps", br)
        res[(yr, COMP)] = summarise_runs([run_sarima(train, order=HYB_ORDER, sorder=HYB_SORDER)], train, test)
        res[(yr, COMP_B)] = summarise_runs([run_sarima(train, order=HYB_ORDER, sorder=HYB_SORDER, exog_train=xs[0], exog_future=xs[1])], train, test)
        res[(yr, "CNNAR (regularised)")] = summarise_runs([run_cnnar(train, seed=s, **CNN_REG) for s in range(N_SEEDS)], train, test)
        res[(yr, "Hybrid SARIMA-CNNAR (regularised)")] = summarise_runs([run_hybrid(train, seed=s, cnn_cfg=HYB_REG) for s in range(N_SEEDS)], train, test)
        res[(yr, "Hybrid SARIMAX(breaks)-CNNAR (regularised)")] = summarise_runs(
            [run_hybrid(train, seed=s, cnn_cfg=HYB_REG, exog_train=xs[0], exog_future=xs[1]) for s in range(N_SEEDS)], train, test)
        for kind, lab in [("lockdown", "lockdown"), ("steps+lockdown", "breaks+lockdown")]:
            xtr, xte = exog_for(train, test.index, kind, br)
            res[(yr, f"SARIMAX-{lab}")] = summarise_runs([run_sarima(train, exog_train=xtr, exog_future=xte)], train, test)
            res[(yr, f"Hybrid SARIMAX({lab})-CNNAR")] = summarise_runs(
                [run_hybrid(train, seed=s, exog_train=xtr, exog_future=xte) for s in range(N_SEEDS)], train, test)
        print(f"fold {yr} ({time.time() - t0:.0f}s)")
    return res

if RUN["cross_validation"] and RUN["extended"]:
    REV_X = cached("extended_cv", extended_cv)
    ALLR = {**REV, **REV_X}
    ext = []
    for name in MODEL_ORDER + EXT_MODELS:
        a = np.concatenate([ALLR[(y, name)]["actual"] for y in FOLD_YEARS]); p = np.concatenate([ALLR[(y, name)]["pred"] for y in FOLD_YEARS])
        lo, hi = block_bootstrap_ci(a, p)
        ext.append({"Model": name, **{f"MAPE {y}": metrics(ALLR[(y, name)]["actual"], ALLR[(y, name)]["pred"])["MAPE (%)"] for y in FOLD_YEARS},
                    "Pooled MAPE (n=60)": pooled_metrics(name, FOLD_YEARS, ALLR)["MAPE (%)"], "95% CI low": lo["MAPE (%)"], "95% CI high": hi["MAPE (%)"],
                    "Pooled MAPE excl. 2020": pooled_metrics(name, FOLD_YEARS[1:], ALLR)["MAPE (%)"]})
    s9_ext = save_table(pd.DataFrame(ext), "S9_extended_cv"); display(s9_ext.round(2))

    # Over-fitting in every fold
    of_fold = pd.DataFrame([{"Model": n, "Fold": y, "Parameters": ALLR[(y, n)]["n_params"],
                             "Train RMSE": np.mean([m["RMSE"] for m in ALLR[(y, n)]["train_metrics"]]),
                             "Test RMSE": np.mean([m["RMSE"] for m in ALLR[(y, n)]["test_metrics"]]),
                             "Test MAPE SD (seeds)": np.std([m["MAPE (%)"] for m in ALLR[(y, n)]["test_metrics"]], ddof=1)}
                            for n in ["CNNAR", "CNNAR (regularised)", MAIN, "Hybrid SARIMA-CNNAR (regularised)",
                                      "Hybrid SARIMAX(breaks)-CNNAR", "Hybrid SARIMAX(breaks)-CNNAR (regularised)"] for y in FOLD_YEARS])
    of_fold["Test/Train"] = of_fold["Test RMSE"] / of_fold["Train RMSE"]
    save_table(of_fold, "S9_overfitting_by_fold"); display(of_fold.groupby("Model", sort=False).mean(numeric_only=True).round(2))

    # Additional Diebold-Mariano tests
    pairs = [(MAIN, COMP), ("Hybrid SARIMAX(breaks)-CNNAR", COMP_B), (MAIN, "Hybrid SARIMA-CNNAR (regularised)"),
             ("Hybrid SARIMAX(breaks)-CNNAR", "Hybrid SARIMAX(breaks)-CNNAR (regularised)"), ("SARIMAX-breaks", "SARIMA"),
             ("SARIMAX-lockdown", "SARIMA"), ("SARIMAX-breaks+lockdown", "SARIMA"), ("Hybrid SARIMAX(breaks)-CNNAR", MAIN),
             ("Hybrid SARIMAX(lockdown)-CNNAR", MAIN), ("Hybrid SARIMAX(breaks+lockdown)-CNNAR", MAIN), ("SARIMAX-lockdown", "LSTM")]
    dm_x = []
    for a_, b_ in pairs:
        for scope, yrs in [("2024", [2024]), ("All folds", FOLD_YEARS), ("Excl. 2020", FOLD_YEARS[1:])]:
            ea, eb = errors(a_, yrs, ALLR), errors(b_, yrs, ALLR); d2, p2 = dm_test(ea, eb, power=2); d1, p1 = dm_test(ea, eb, power=1)
            dm_x.append({"Model A": a_, "Model B": b_, "Scope": scope, "DM (squared)": d2, "p (squared)": p2,
                         "DM (absolute)": d1, "p (absolute)": p1, "Wilcoxon p": stats.wilcoxon(np.abs(ea), np.abs(eb)).pvalue})
    display(save_table(pd.DataFrame(dm_x), "S9_dm_additional").round(3))

# %%
if RUN["cross_validation"]:
    # Table 15 - training vs test in 2024 and model complexity (regularised networks: 2024 fold)
    def reg_2024():
        tr, te = fold_split(2024)
        return {"CNNAR (regularised)": summarise_runs([run_cnnar(tr, seed=s, **CNN_REG) for s in range(N_SEEDS)], tr, te),
                "Hybrid SARIMA-CNNAR (regularised)": summarise_runs([run_hybrid(tr, seed=s, cnn_cfg=HYB_REG) for s in range(N_SEEDS)], tr, te)}
    REG24 = cached("regularised_2024", reg_2024)
    rows = []
    for name in MODEL_ORDER + list(REG24):
        r = REV[(2024, name)] if name in MODEL_ORDER else REG24[name]
        tr = pd.DataFrame(r["train_metrics"]); te = pd.DataFrame(r["test_metrics"]); sd = lambda d, k: d[k].std(ddof=1) if len(d) > 1 else np.nan
        if name in MODEL_ORDER:
            fold_ratio = np.mean([np.mean([m["RMSE"] for m in REV[(y, name)]["test_metrics"]]) / np.mean([m["RMSE"] for m in REV[(y, name)]["train_metrics"]]) for y in FOLD_YEARS])
        else:
            fold_ratio = of_fold[of_fold.Model == name]["Test/Train"].mean() if RUN["extended"] else np.nan
        rows.append({"Model": name, "Parameters": r["n_params"], "Training samples": r["n_samples"],
                     "Train RMSE": fmt(tr["RMSE"].mean(), sd(tr, "RMSE"), 1), "Test RMSE": fmt(te["RMSE"].mean(), sd(te, "RMSE"), 1),
                     "Train MAPE": fmt(tr["MAPE (%)"].mean(), sd(tr, "MAPE (%)")), "Test MAPE": fmt(te["MAPE (%)"].mean(), sd(te, "MAPE (%)")),
                     "Test/Train RMSE 2024": te["RMSE"].mean() / tr["RMSE"].mean(), "Test/Train RMSE mean of folds": fold_ratio})
    table15 = save_table(pd.DataFrame(rows), "Table15_overfitting"); display(table15.round(2))

    # Fig 21 - learning curves (2024 fold, seed 0), validation = last 12 training windows
    tr24, _ = fold_split(2024)
    curves = cached("learning_curves", lambda: (run_cnnar(tr24, seed=0, track_val=True)["history"], run_cnnar(tr24, seed=0, **CNN_REG)["history"]))
    print(f"Published CNNAR: validation-loss minimum at epoch {int(np.argmin(curves[0]['val_loss'])) + 1}; "
          f"regularised CNNAR stopped after {len(curves[1]['loss'])} epochs")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, h_, t in [(axes[0], curves[0], "CNNAR as published (128/64 filters, fixed epochs)"), (axes[1], curves[1], "Regularised CNNAR (32/16 filters, L2, early stopping)")]:
        ax.plot(h_["loss"], label="Training loss"); ax.plot(h_["val_loss"], label="Validation loss (last 12 training windows)")
        ax.set_yscale("log"); ax.set(title=t, xlabel="Epoch"); ax.legend(); ax.grid(alpha=0.3)
    axes[0].set_ylabel("MSE (scaled)"); save_figure(fig, "Fig21_Learning_curves")

# %% [markdown]
# ## 10. Forecast comparison: Diebold–Mariano tests, bootstrap CIs, interval coverage (Tables 14, 16; Fig 19; S6 Table)

# %%
if RUN["cross_validation"]:
    SCOPES = {"Test year 2024 (n=12)": [2024], "All folds (n=60)": FOLD_YEARS, "Excl. 2020 (n=48)": FOLD_YEARS[1:]}
    # Table 14 - hybrid vs each comparator
    dm_rows = []
    for scope, yrs in SCOPES.items():
        e_main = errors(MAIN, yrs)
        for other in [m for m in MODEL_ORDER if m != MAIN]:
            e_o = errors(other, yrs); d2, p2 = dm_test(e_main, e_o, power=2); d1, p1 = dm_test(e_main, e_o, power=1)
            dm_rows.append({"Scope": scope, "Hybrid vs": other, "DM (squared)": d2, "p (squared)": p2, "DM (absolute)": d1, "p (absolute)": p1,
                            "Wilcoxon p": stats.wilcoxon(np.abs(e_main), np.abs(e_o)).pvalue})
    table14 = save_table(pd.DataFrame(dm_rows), "Table14_dm_tests"); display(table14.round(3))

    # Bootstrap CIs for all metrics and scopes (Table 12 CIs, Table 13 pooled CI)
    boot = []
    for scope, yrs in SCOPES.items():
        for name in MODEL_ORDER:
            a = np.concatenate([REV[(y, name)]["actual"] for y in yrs]); p = np.concatenate([REV[(y, name)]["pred"] for y in yrs])
            point = metrics(a, p); lo, hi = block_bootstrap_ci(a, p)
            boot.append({"Scope": scope, "Model": name, **{k: f"{point[k]:.{3 if k == 'R²' else 2}f} [{lo[k]:.{3 if k == 'R²' else 2}f}, {hi[k]:.{3 if k == 'R²' else 2}f}]" for k in point}})
    boot_ci = save_table(pd.DataFrame(boot), "S6_bootstrap_ci"); display(boot_ci)

    # Table 16 - coverage of model-based and conformal intervals, test years 2022-2024
    EVAL_YEARS = [2022, 2023, 2024]
    def conformal_q(name, calib_years, alpha=ALPHA):
        """Split-conformal quantile of the absolute out-of-sample errors of the calibration folds."""
        e = np.sort(np.abs(errors(name, calib_years))); k = int(np.ceil((len(e) + 1) * (1 - alpha)))
        return e[min(k, len(e)) - 1]
    cov = []
    for name in MODEL_ORDER:
        for method in ["Model-based", "Conformal"]:
            hits, widths, rel, per = [], [], [], {}
            for yr in EVAL_YEARS:
                r = REV[(yr, name)]
                if method == "Model-based":
                    if r["lower"] is None:
                        break
                    lo, hi = np.asarray(r["lower"]), np.asarray(r["upper"])
                else:
                    q = conformal_q(name, [y for y in FOLD_YEARS if y < yr]); lo, hi = r["pred"] - q, r["pred"] + q
                inside = (r["actual"] >= lo) & (r["actual"] <= hi); per[yr] = inside.mean() * 100
                hits += list(inside); widths += list(hi - lo); rel += list((hi - lo) / r["pred"] * 100)
            if hits:
                cov.append({"Model": name, "Interval": method, **{f"Coverage {y} (%)": v for y, v in per.items()},
                            "Coverage (%)": np.mean(hits) * 100, "Mean width": np.mean(widths), "Mean width (% of forecast)": np.mean(rel)})
    table16 = save_table(pd.DataFrame(cov), "Table16_interval_coverage"); display(table16.round(1))

    # Fig 19 - hybrid 2024 forecast with both intervals
    r = REV[(2024, MAIN)]; q = conformal_q(MAIN, [y for y in FOLD_YEARS if y < 2024])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_all["2023":], "k-o", label="Observed"); ax.plot(r["index"], r["pred"], color="purple", lw=2, label="Hybrid forecast (mean of seeds)")
    ax.fill_between(r["index"], r["lower"], r["upper"], color="red", alpha=0.15, label="Model-based 95% interval")
    ax.fill_between(r["index"], r["pred"] - q, r["pred"] + q, color="purple", alpha=0.25, label="Conformal 95% interval")
    ax.set(title="Hybrid SARIMA–CNNAR: 2024 forecast and 95% prediction intervals", ylabel="TB notifications"); ax.legend(); ax.grid(alpha=0.3)
    save_figure(fig, "Fig19_Hybrid_2024_PI")

# %% [markdown]
# ## 11. Sensitivity analyses (Tables 17–18; S4 Table)
# Configurations are changed one at a time from the published setting and evaluated on the 2024 test year. Neural models use the mean of `N_SEEDS_SENS` seeds.

# %%
def sensitivity():
    tr, te = fold_split(2024)
    nn = lambda fn: pd.DataFrame([metrics(te.values, fn(s)["pred"]) for s in range(N_SEEDS_SENS)]).mean().to_dict()
    one = lambda out: metrics(te.values, out["pred"])
    jobs = [("SARIMA", "Order", f"{o}{so[:3]}".replace(" ", ""), lambda o=o, so=so: one(run_sarima(tr, order=o, sorder=so)))
            for o, so in [((0, 1, 1), (1, 1, 1, 12)), ((2, 1, 1), (1, 1, 1, 12)), ((1, 1, 0), (1, 1, 1, 12)), ((1, 1, 2), (1, 1, 1, 12)),
                          ((1, 1, 1), (0, 1, 1, 12)), ((1, 1, 1), (1, 1, 0, 12)), ((1, 1, 1), (2, 1, 1, 12))]]
    for p, vals in [("filters", [(32, 16), (64, 32), (256, 128)]), ("kernel", [2, 4]), ("lr", [0.0005, 0.002]), ("dropout", [0.1, 0.3]), ("look_back", [6, 24])]:
        jobs += [("CNNAR", p, str(v), lambda p=p, v=v: nn(lambda s: run_cnnar(tr, seed=s, **{p: v}))) for v in vals]
    jobs += [(MAIN, "SARIMA order", f"{o}{so[:3]}".replace(" ", ""), lambda o=o, so=so: nn(lambda s: run_hybrid(tr, seed=s, order=o, sorder=so)))
             for o, so in [((0, 1, 1), (1, 1, 0, 12)), ((2, 1, 1), (1, 1, 0, 12)), ((1, 1, 1), (1, 1, 1, 12)), ((1, 1, 1), (0, 1, 1, 12))]]
    for p, vals in [("filters", [(32, 16), (64, 32), (256, 128)]), ("kernel", [2, 4]), ("lr", [0.0005, 0.002]), ("dropout", [0.1, 0.3])]:
        jobs += [(MAIN, f"CNNAR {p}", str(v), lambda p=p, v=v: nn(lambda s: run_hybrid(tr, seed=s, cnn_cfg={p: v}))) for v in vals]
    jobs += [(MAIN, "Training window start", st[:7], lambda st=st: nn(lambda s: run_hybrid(tr[st:], seed=s))) for st in ["2016-01-01", "2018-01-01"]]
    for p, vals in [("units", [32, 128]), ("look_back", [12, 24]), ("dropout", [0.1, 0.3]), ("lr", [0.0005, 0.002])]:
        jobs += [("LSTM", p, str(v), lambda p=p, v=v: nn(lambda s: run_lstm(tr, seed=s, **{p: v}))) for v in vals]
    for p, vals in [("changepoint_prior_scale", [0.01, 0.5]), ("seasonality_mode", ["additive"]), ("yearly_fourier", [5, 15]), ("seasonality_prior_scale", [1, 20])]:
        jobs += [("Prophet", p, str(v), lambda p=p, v=v: one(run_prophet(tr, **{p: v}))) for v in vals]
    for p, vals in [("max_depth", [3, 4]), ("n_estimators", [500, 2000]), ("learning_rate", [0.005, 0.05]), ("subsample", [0.6, 1.0]), ("look_back", [6, 24])]:
        jobs += [("XGBoost", p, str(v), lambda p=p, v=v: one(run_xgb(tr, **{p: v}))) for v in vals]
    rows = [{"Model": m, "Parameter": p, "Value": v, **fn()} for m, p, v, fn in jobs]
    # data-related sensitivity: IQR-capped training data, observed data only
    def iqr_cap(series):
        s = series.copy()
        for yr in sorted(set(s.index.year)):
            m = s.index.year == yr; q1, q3 = s[m].quantile([0.25, 0.75]); s[m] = s[m].clip(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
        return s
    data_rows = []
    for label, series in [("2015-2023, IQR-capped", iqr_cap(tr)), ("2018-2023, observed only", tr[OBSERVED_START:])]:
        for name, outs in [("SARIMA", [run_sarima(series)]), ("CNNAR", [run_cnnar(series, seed=s) for s in range(N_SEEDS)]),
                           (MAIN, [run_hybrid(series, seed=s) for s in range(N_SEEDS)])]:
            m = pd.DataFrame([metrics(te.values, o["pred"]) for o in outs])
            data_rows.append({"Training data": label, "Model": name, **{k: fmt(m[k].mean(), m[k].std(ddof=1) if len(m) > 1 else np.nan) for k in m}})
    return pd.DataFrame(rows), pd.DataFrame(data_rows)

if RUN["sensitivity"] and RUN["cross_validation"]:
    sens_detail, sens_data = cached("sensitivity", sensitivity)
    base = pd.DataFrame([{"Model": n, "Parameter": "Baseline", "Value": "published", **pd.DataFrame(REV[(2024, n)]["test_metrics"]).mean().to_dict()}
                         for n in ["SARIMA", "CNNAR", MAIN, "LSTM", "Prophet", "XGBoost"]])
    sens_all = pd.concat([base, sens_detail], ignore_index=True)
    sens_all["Model"] = pd.Categorical(sens_all["Model"], ["SARIMA", "CNNAR", MAIN, "LSTM", "Prophet", "XGBoost"]); sens_all = sens_all.sort_values("Model", kind="stable")
    save_table(sens_all, "S4_sensitivity_detail")
    table17 = save_table(sens_all.groupby("Model", observed=True).agg(Configurations=("MAPE (%)", "size"), MAPE_min=("MAPE (%)", "min"), MAPE_max=("MAPE (%)", "max"),
                         RMSE_min=("RMSE", "min"), RMSE_max=("RMSE", "max"), R2_min=("R²", "min"), R2_max=("R²", "max"),
                         CV_of_MAPE=("MAPE (%)", lambda v: v.std(ddof=1) / v.mean())).reset_index(), "Table17_sensitivity")
    display(table17.round(3))
    raw = pd.DataFrame([{"Training data": "2015-2023, raw (primary)", "Model": n, **{k: fmt(np.mean(v), np.std(v, ddof=1) if len(v) > 1 else np.nan)
                         for k, v in pd.DataFrame(REV[(2024, n)]["test_metrics"]).items()}} for n in ["SARIMA", "CNNAR", MAIN]])
    table18 = save_table(pd.concat([raw, sens_data], ignore_index=True), "Table18_outlier_backcast"); display(table18)

# %% [markdown]
# ## 12. Forecasts for 2025 (Tables 4, 7, 9; Fig 22; S7 Table)
# All models are refitted on January 2015–December 2024. Conformal intervals use the absolute errors of the five cross-validation folds.

# %%
def forecasts_2025():
    idx = future_index(y_all); bp = bai_perron(y_all)
    ex_a, ex_f = step_dummies(y_all.index, bp["break_dates"]), step_dummies(idx, bp["break_dates"])
    out = {"SARIMA": run_sarima(y_all), "SARIMAX-breaks": run_sarima(y_all, exog_train=ex_a, exog_future=ex_f),
           MAIN: [run_hybrid(y_all, seed=s) for s in range(N_SEEDS)],
           "Hybrid SARIMAX(breaks)-CNNAR": [run_hybrid(y_all, seed=s, exog_train=ex_a, exog_future=ex_f) for s in range(N_SEEDS)],
           "CNNAR": [run_cnnar(y_all, seed=s, mc=(s == 0)) for s in range(N_SEEDS)]}
    for k in out:   # drop statsmodels result objects before caching
        out[k] = [{kk: vv for kk, vv in o.items() if kk not in ("res",)} for o in out[k]] if isinstance(out[k], list) else {kk: vv for kk, vv in out[k].items() if kk != "res"}
    return idx, out

if RUN["forecast_2025"] and RUN["cross_validation"]:
    idx25, F25 = cached("forecast_2025", forecasts_2025)
    rows = []
    for name, o in F25.items():
        runs = o if isinstance(o, list) else [o]
        pred = np.mean([r["pred"] for r in runs], axis=0); sdv = np.std([r["pred"] for r in runs], axis=0, ddof=1) if len(runs) > 1 else np.zeros(H)
        lo, hi = (pred - runs[0]["half"], pred + runs[0]["half"]) if "half" in runs[0] else (runs[0]["lower"], runs[0]["upper"])
        q = conformal_q(name, FOLD_YEARS)
        rows += [{"Model": name, "Date": d.strftime("%Y-%m"), "Forecast": pred[i], "Seed SD": sdv[i],
                  "Model-based lower 95%": lo[i], "Model-based upper 95%": hi[i], "Conformal lower 95%": pred[i] - q, "Conformal upper 95%": pred[i] + q}
                 for i, d in enumerate(idx25)]
    fc2025 = save_table(pd.DataFrame(rows), "S7_forecast_2025"); save_table(fc2025, "Forecast_2025_all_models"); display(fc2025.round(0))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_all["2021":], "k-o", ms=3, label="Observed")
    for name in ["SARIMA", "SARIMAX-breaks", MAIN, "Hybrid SARIMAX(breaks)-CNNAR"]:
        f = fc2025[fc2025.Model == name]; ax.plot(idx25, f["Forecast"], lw=2 if "Hybrid" in name else 1.2, ls="-" if "Hybrid" in name else "--", label=name)
    f = fc2025[fc2025.Model == MAIN]; ax.fill_between(idx25, f["Conformal lower 95%"], f["Conformal upper 95%"], color="purple", alpha=0.2, label="Hybrid conformal 95% interval")
    ax.axvline(idx25[0], color="grey", ls=":"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set(title="Forecasts for 2025 (models refitted on 2015–2024)", ylabel="TB notifications"); save_figure(fig, "Fig22_Forecast_2025")

    # Fig 18 - SARIMA: training fit, 2024 test forecast, 2025 forecast
    tr24, te24 = fold_split(2024); s24 = run_sarima(tr24); s25 = F25["SARIMA"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tr24, color="blue", label="Training data"); ax.plot(s24["fitted"], "b--", alpha=0.7, label="Training predictions")
    ax.plot(te24, color="red", label="Test data (2024)"); ax.plot(te24.index, s24["pred"], "r--", label="Test predictions")
    ax.plot(idx25, s25["pred"], "g--", label="2025 forecast"); ax.fill_between(idx25, s25["lower"], s25["upper"], color="lightgreen", alpha=0.4, label="95% prediction interval")
    ax.legend(); ax.grid(alpha=0.3); ax.set(title="SARIMA(1,1,1)(1,1,1)12 forecasts", ylabel="TB notifications"); save_figure(fig, "Fig18_SARIMA_forecast")

# %% [markdown]
# ## 13. Residual diagnostics of the 2024 test-year forecasts (S5 Table)
# n = 12 multi-step errors from one forecast origin; serial correlation is expected and the tests have low power.

# %%
if RUN["cross_validation"]:
    s5 = []
    for name in MODEL_ORDER + (list(REG24) if "REG24" in globals() else []):
        r = REV[(2024, name)] if name in MODEL_ORDER else REG24[name]
        d = residual_diagnostics(r["actual"] - r["pred"], 11)["Statistic"].to_dict(); p_ = residual_diagnostics(r["actual"] - r["pred"], 11)["p-value"].to_dict()
        s5.append({"Model": name, "Mean residual": d["Mean residual"], "Shapiro-Wilk W": d["Shapiro-Wilk"], "Shapiro-Wilk p": p_["Shapiro-Wilk"],
                   "Ljung-Box Q (lag 11)": d["Ljung-Box (lag 11)"], "Ljung-Box p": p_["Ljung-Box (lag 11)"], "ARCH LM": d["ARCH LM"], "ARCH p": p_["ARCH LM"]})
    display(save_table(pd.DataFrame(s5), "S5_residual_diagnostics").round(4))

# %% [markdown]
# ## 14. Method figures (Figs 1–3)

# %%
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def _box(ax, x, y, w, h, txt, fc, ec, fs=9.5, bold=False, color="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=color)

def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color="#4A5568", lw=1.3))

if RUN["figures"]:
    # Fig 1 - CNNAR architecture (parameter count taken from the implemented model)
    n_par = build_cnn(**CNN_BASE).count_params()
    fig, ax = plt.subplots(figsize=(13, 3.6)); ax.set_xlim(0, 13.6); ax.set_ylim(0, 3.6); ax.axis("off")
    blocks = [("Input window\n12 lagged values\n(min–max scaled)\nshape 12 × 1", "#FFF2CC", "#B7950B", 1.55),
              ("Conv1D\n128 filters, kernel 3\nReLU, valid padding\n→ 10 × 128", "#DCE6F2", "#1F4E79", 1.75), ("Dropout 0.2", "#EDEDED", "#7F7F7F", 1.15),
              ("Conv1D\n64 filters, kernel 3\nReLU, valid padding\n→ 8 × 64", "#DCE6F2", "#1F4E79", 1.75), ("Dropout 0.2", "#EDEDED", "#7F7F7F", 1.15),
              ("Flatten\n→ 512", "#E2EFDA", "#548235", 1.05), ("Dense\n64 units, ReLU", "#FCE4D6", "#C55A11", 1.3), ("Dense\n1 unit, linear\n→ ŷ(t)", "#FCE4D6", "#C55A11", 1.3)]
    x = 0.1
    for t, fc, ec, w in blocks:
        _box(ax, x, 1.2, w, 1.6, t, fc, ec)
        if x > 0.2: _arrow(ax, x - 0.25, 2.0, x - 0.02, 2.0)
        x += w + 0.27
    ax.text(6.8, 3.35, "CNNAR architecture (stand-alone model and residual learner of the hybrid)", ha="center", fontsize=12, fontweight="bold")
    ax.text(6.8, 0.55, f"{n_par:,} trainable parameters · Adam (learning rate 0.001), MSE loss · stand-alone CNNAR: 300 epochs, batch 24 · "
            "hybrid residual model: 500 epochs, batch 16 · no early stopping, no pooling\nMulti-step forecasts are recursive; 95% intervals by Monte-Carlo dropout "
            "(100 trajectories, dropout active at prediction)", ha="center", fontsize=8.8, color="#333")
    save_figure(fig, "Fig01_CNNAR_architecture")
    # Fig 2 - hybrid framework
    fig, ax = plt.subplots(figsize=(13, 4.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.text(6.5, 4.35, "Hybrid SARIMA–CNNAR framework", ha="center", fontsize=13, fontweight="bold")
    _box(ax, 0.1, 2.4, 1.9, 1.1, "Monthly TB\nnotifications y(t)\n(training window)", "#FFF2CC", "#B7950B")
    _box(ax, 2.5, 2.4, 2.4, 1.1, "SARIMA(1,1,1)(1,1,0)$_{12}$\n[break-adjusted variant:\nSARIMAX + step dummies]", "#DCE6F2", "#1F4E79")
    _box(ax, 5.4, 2.4, 2.3, 1.1, "Residuals\ne(t) = y(t) − ŷ$^{L}$(t)\n(13 start-up values dropped)", "#EDEDED", "#7F7F7F")
    _box(ax, 8.2, 2.4, 2.2, 1.1, "CNNAR on residuals\n(12 lagged residuals,\nmin–max scaled)", "#FCE4D6", "#C55A11")
    for a_, b_ in [(2.0, 2.5), (4.9, 5.4), (7.7, 8.2)]: _arrow(ax, a_, 2.95, b_, 2.95)
    _box(ax, 2.5, 0.6, 2.4, 1.0, "SARIMA(X) 12-month\nforecast ŷ$^{L}$(t+h)", "#DCE6F2", "#1F4E79")
    _box(ax, 8.2, 0.6, 2.2, 1.0, "Recursive CNNAR residual\nforecast ê(t+h)", "#FCE4D6", "#C55A11")
    _arrow(ax, 3.7, 2.4, 3.7, 1.6); _arrow(ax, 9.3, 2.4, 9.3, 1.6)
    _box(ax, 10.85, 0.45, 2.05, 1.3, "Hybrid forecast\nŷ(t+h) =\nŷ$^{L}$(t+h) + ê(t+h)", "#1F4E79", "#1F4E79", fs=10, bold=True, color="white")
    _arrow(ax, 10.4, 1.1, 10.85, 1.1); ax.plot([3.7, 3.7, 11.875], [0.6, 0.2, 0.2], color="#4A5568", lw=1.3); _arrow(ax, 11.875, 0.2, 11.875, 0.45)
    ax.text(7.3, 0.27, "+", fontsize=14, ha="center", color="#4A5568"); save_figure(fig, "Fig02_Hybrid_framework")
    # Fig 3 - cross-validation design
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for k, yr in enumerate(FOLD_YEARS, 1):
        tr, te = fold_split(yr)
        ax.barh(k, (tr.index[-1] - tr.index[0]).days + 31, left=tr.index[0], color="steelblue", height=0.6, label="Training (expanding)" if k == 1 else None)
        ax.barh(k, (te.index[-1] - te.index[0]).days + 31, left=te.index[0], color="darkorange", height=0.6, label="Test (12 months)" if k == 1 else None)
    ax.axvspan(y_all.index[0], OBSERVED_START, color="grey", alpha=0.12, label="Back-cast years")
    ax.set_yticks(range(1, 6)); ax.set_yticklabels([f"Fold {k}" for k in range(1, 6)]); ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Expanding-window time-series cross-validation"); ax.legend(loc="lower left", fontsize=8); save_figure(fig, "Fig03_CV_design")

# %% [markdown]
# ## 15. Export: manuscript tables (one workbook) and supplementary tables S2–S9
# A table is exported only if the stage that produces it was run in this session.

# %%
def export_workbook(path, sheets):
    with pd.ExcelWriter(path) as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    print("written:", path)

g = globals()
manuscript = {k: v.reset_index() if k in ("Table01", "Table05", "Table06", "Table13") else v for k, v in {
    "Table01": g.get("table1"), "Table02": g.get("table2"), "Table03": g.get("table3"), "Table05": g.get("t5"), "Table06": g.get("t6"),
    "Table11": g.get("table11"), "Table12": g.get("table12"), "Table13": g.get("table13"), "Table14": g.get("table14"),
    "Table15": g.get("table15"), "Table16": g.get("table16"), "Table17": g.get("table17"), "Table18": g.get("table18"),
    "Tables04_07_09_2025": g.get("fc2025")}.items() if v is not None}
export_workbook(DIRS["manuscript"] / "Manuscript_Tables.xlsx", manuscript)

fit_log = save_table(pd.DataFrame(FIT_LOG).drop_duplicates(), "SARIMA_fit_log")
print("SARIMA(X) fits refitted with constraints in this session:", int(fit_log["Refitted with constraints"].sum()) if len(fit_log) else 0)
# ---- Supporting-information workbooks (S1–S9 Tables): titles, captions, sheets ----
# The same captions are printed in the manuscript's "Supporting Information" list.

SI_CAPTIONS = {
    "S1 File": "Analysis notebook (Jupyter, IPYNB). Notebook that reproduces, from S1 Table, every number, table and figure in this article and S2–S9 Tables; the first cells list the software environment and the run instructions.",
    "S1 Table": "Monthly TB notification dataset, January 2015–December 2024 (XLSX). Final, complete Gregorian-month series used in all analyses (sheet A): NTCC notifications reported by Nepali (Bikram Sambat) month for 2018–2024, converted to Gregorian calendar months by day-weighted apportioning after Kalman-smoother imputation of missing Nepali months, and month-specific compound-annual-growth-rate back-cast values for 2015–2017; reported Nepali-month values with their dates and the imputed months flagged (sheet B).",
    "S2 Table": "SARIMA grid search (XLSX). AIC, BIC, HQIC and log-likelihood of the candidate models with d = 1 and D = 1, ranked by AIC (sheet A), and of all candidate models, including the excluded degenerate fits (sheet B).",
    "S3 Table": "Cross-validation results (XLSX). Training and test metrics for every fold, model and random seed (sheet A), fold-level metrics of the seed-averaged forecasts (B), monthly forecasts of every fold (C) and the SARIMA(X) fit log, including the constrained refits of the 2020 fold (D).",
    "S4 Table": "Sensitivity analyses (XLSX). 2024 test-year metrics for every model configuration (sheet A), summary by model family (B; Table 17) and the outlier and back-cast sensitivity analyses (C; Table 18).",
    "S5 Table": "Residual diagnostics of the 2024 test-year forecasts (XLSX). Mean residual, Shapiro–Wilk, Ljung–Box (lag 11) and ARCH tests for all models, including the regularised networks.",
    "S6 Table": "Bootstrap confidence intervals and forecast-comparison tests (XLSX). Circular block-bootstrap 95% confidence intervals of MAE, RMSE, MAPE and R² for the 2024 test year, all folds and all folds except 2020 (sheet A); Diebold–Mariano tests with squared- and absolute-error loss and Wilcoxon tests (B); 2024 metrics of all models (C); training versus test performance (D); and prediction-interval coverage (E).",
    "S7 Table": "2025 forecasts (XLSX). Monthly 2025 forecasts of all models refitted on 2015–2024, with the between-seed SD and model-based and conformal 95% prediction intervals.",
    "S8 Table": "Structural breaks and intervention variables (XLSX). Bai–Perron break dates estimated from the training data of each cross-validation fold (sheet A) and from the full series (B), SARIMA versus SARIMAX comparisons for the step dummies, the lockdown indicator and both combined (C), and the estimated coefficients (D).",
    "S9 Table": "Additional cross-validation analyses (XLSX). MAPE by fold and pooled for the hybrids’ own linear components, the regularised networks and the lockdown-indicator models (sheet A), test/train RMSE ratios by fold (B), and the corresponding Diebold–Mariano and Wilcoxon tests (C).",
}

SI_NOTES = {
    "S1 Table": "Back-cast values (2015–2017) are not observations; they were reconstructed as described in Section 2.3. Share of days from imputed Nepali months: fraction of the Gregorian month covered by a Nepali month whose value was imputed with the Kalman smoother.",
    "S2 Table": "Degenerate fit: log-likelihood ≈ 0 (a numerical artefact of D = 2 with few usable observations); these models were excluded.",
    "S3 Table": "Folds are labelled by test year; each fold forecasts January–December of that year from December of the previous year. Seed: random seed of the neural network (0–4); single-run models have seed 0. Train metrics of SARIMA-type models exclude the 13 start-up values.",
    "S4 Table": "One setting changed at a time from the configuration in Sections 2.6–2.10; neural models: mean of two seeds per configuration.",
    "S5 Table": "Mean residual = mean(observed − forecast) over January–December 2024; positive values indicate underprediction.",
    "S6 Table": "DM: Diebold–Mariano statistic with the Harvey–Leybourne–Newbold correction; negative values favour the standard hybrid. Bootstrap: circular block bootstrap, block length 3, 2,000 replicates.",
    "S7 Table": "Seed SD: standard deviation of the forecasts of the five networks (0 for single-run models). XGBoost has no model-based interval; CNNAR and LSTM model-based intervals are Monte-Carlo dropout intervals.",
    "S8 Table": "Lockdown indicator: 1 in April–July 2020 and May–June 2021, otherwise 0; set to 0 over forecast horizons. The Breaks column lists the Bai–Perron break dates of the sample; they enter the model only in the specifications with step dummies.",
    "S9 Table": "Regularised networks: 32/16 filters, 16-unit dense layer, dropout 0.3, L2 = 10⁻³, early stopping. Pooled MAPE over 60 test months (48 excluding 2020).",
}

SI_SHEETS = {  # table -> list of (sheet name, description, key of the data frame)
    "S1 Table": [("A Dataset", "Final Gregorian-month series, January 2015–December 2024", "s1"),
                 ("B Nepali months", "Reported Nepali-month values, dates and Kalman-imputed months", "preprocessing")],
    "S2 Table": [("A d=1 D=1 ranked", "Candidate models with d = 1 and D = 1, ranked by AIC", "grid_d1D1"),
                 ("B All candidates", "All candidate orders, including degenerate fits", "grid")],
    "S3 Table": [("A All runs", "Training and test metrics for every fold, model and seed", "cv_long"),
                 ("B Per fold", "Metrics of the seed-averaged forecast by fold (Table 13)", "per_fold"),
                 ("C Fold forecasts", "Monthly observed values and forecasts of every fold", "fold_fc"),
                 ("D SARIMA fit log", "Every SARIMA(X) fit; constrained refits flagged", "fit_log")],
    "S4 Table": [("A All configurations", "2024 metrics of every configuration", "sens_all"),
                 ("B Summary", "Range and CV of MAPE by model family (Table 17)", "table17"),
                 ("C Data sensitivity", "Outlier capping and observed-data-only training (Table 18)", "table18")],
    "S5 Table": [("A Residual diagnostics", "Residual tests of the 2024 forecasts", "s5")],
    "S6 Table": [("A Bootstrap CIs", "Estimate [95% CI] for each metric, model and scope", "boot_ci"),
                 ("B DM and Wilcoxon", "Standard hybrid versus each comparator (Table 14)", "table14"),
                 ("C 2024 comparison", "2024 metrics of all models (Table 12)", "table12"),
                 ("D Overfitting", "Training versus test performance (Table 15)", "table15"),
                 ("E PI coverage", "Coverage and width of 95% prediction intervals (Table 16)", "table16")],
    "S7 Table": [("A 2025 forecasts", "Monthly 2025 forecasts with 95% prediction intervals", "fc2025")],
    "S8 Table": [("A Breaks per fold", "Bai–Perron break dates from each fold's training data", "fold_breaks"),
                 ("B Full sample", "Bai–Perron results for 0–5 breaks on the full series (Table 3)", "table3"),
                 ("C SARIMAX comparison", "SARIMA versus SARIMAX: AIC, BIC and likelihood-ratio tests for all three specifications (Table 11 shows the step dummies)", "table11"),
                 ("D Coefficients", "Estimated step-dummy and lockdown coefficients", "coef_tab")],
    "S9 Table": [("A Extended CV", "MAPE by fold and pooled for the additional models", "s9_ext"),
                 ("B Overfitting by fold", "Test/train RMSE ratio by fold for the convolutional models", "of_fold"),
                 ("C DM tests", "Diebold–Mariano and Wilcoxon tests of the additional comparisons", "dm_x")],
}

ARTICLE = "Data-Driven Hybrid Model of SARIMA-CNNAR For Tuberculosis Incidence Time Series Analysis in Nepal"


def write_si_workbook(path, label, frames, environment=""):
    """Write one supporting table: a 'Read me' sheet (title, caption, sheet list, notes) followed by the data sheets."""
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter
    title, _, desc = SI_CAPTIONS[label].partition(" (XLSX). ")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame().to_excel(xw, sheet_name="Read me", index=False)
        for name, _, key in SI_SHEETS[label]:
            df = frames[key].copy()
            if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
                df = df.reset_index()
            df.to_excel(xw, sheet_name=name[:31], index=False)
        wb = xw.book
        rm = wb["Read me"]
        rows = [(f"{label}. {title}", "title"), (desc, "text"), ("", None), (f"Article: {ARTICLE}", "text")]
        if environment:
            rows.append((f"Produced by S1 File ({environment}).", "text"))
        rows += [("", None), ("Sheets", "head")] + [(f"{n} – {d}", "text") for n, d, _ in SI_SHEETS[label]]
        if label in SI_NOTES:
            rows += [("", None), ("Notes", "head"), (SI_NOTES[label], "text")]
        for i, (txt, kind) in enumerate(rows, start=1):
            c = rm.cell(row=i, column=1, value=txt)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            if kind in ("title", "head"):
                c.font = Font(bold=True, size=12 if kind == "title" else 11)
        rm.column_dimensions["A"].width = 110
        for name, _, key in SI_SHEETS[label]:
            ws = wb[name[:31]]
            ws.freeze_panes = "A2"
            for c in ws[1]:
                c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="DDEBF7")
                c.alignment = Alignment(wrap_text=True, vertical="top")
            for j, col in enumerate(ws.iter_cols(min_row=1, max_row=min(ws.max_row, 200)), start=1):
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[get_column_letter(j)].width = min(max(10, width + 2), 45)
                for c in col[1:]:
                    if isinstance(c.value, float):
                        v = abs(c.value)
                        c.number_format = "0.00E+00" if (0 < v < 1e-4 or v >= 1e7) else ("0.0000" if v < 1 else "#,##0.00")
    return path

fold_breaks = pd.DataFrame([{"Fold (test year)": y, "Breaks (training data)": ", ".join(d.strftime("%Y-%m") for d in b)}
                            for y, b in g.get("FOLD_BREAKS", {}).items()])
frames = {"s1": g.get("s1"), "preprocessing": g.get("preprocessing"), "grid_d1D1": g.get("grid_d1D1"), "grid": g.get("grid"), "cv_long": g.get("cv_long"),
          "per_fold": g.get("per_fold"), "fold_fc": g.get("fold_fc"), "fit_log": fit_log, "sens_all": g.get("sens_all"),
          "table17": g.get("table17"), "table18": g.get("table18"), "s5": pd.DataFrame(g.get("s5", [])),
          "boot_ci": g.get("boot_ci"), "table14": g.get("table14"), "table12": g.get("table12"), "table15": g.get("table15"),
          "table16": g.get("table16"), "fc2025": g.get("fc2025"), "fold_breaks": fold_breaks, "table3": g.get("table3"),
          "table11": g.get("table11"), "coef_tab": g.get("coef_tab"), "s9_ext": g.get("s9_ext"), "of_fold": g.get("of_fold"),
          "dm_x": pd.DataFrame(g.get("dm_x", []))}
ENV_TEXT = f"Python {platform.python_version()}, {platform.platform()}"
for label, sheets in SI_SHEETS.items():
    if all(frames.get(k) is not None and len(frames[k]) for _, _, k in sheets):
        write_si_workbook(DIRS["supplementary"] / f"{label.replace(' ', '_')}.xlsx", label, frames, ENV_TEXT)
        print("written:", f"{label.replace(' ', '_')}.xlsx")
    else:
        print(f"{label}: skipped (a stage it needs was not run in this session)")

# %% [markdown]
# ## 16. Reproducibility check
# Compares key results of this session with the published values. **Deterministic** quantities (statistical models and tests) must match the published value to its last reported digit. **Platform-dependent** quantities (neural networks, XGBoost, Prophet) are flagged CHECK if they differ by more than 0.5 MAPE points; that usually means a different library version or hardware, not an error.

# %%
REFERENCE = [  # (description, getter, published value, deterministic?)
    ("Back-cast Jan 2015", lambda: y_all.iloc[0], 1451, True),
    ("Growth 2018->2024 (%)", lambda: (y_all["2024"].sum() / y_all["2018"].sum() - 1) * 100, 43.0, True),
    ("Kruskal-Wallis H (2015-2024)", lambda: table2.iloc[0]["Statistic"], 60.99, True),
    ("Bai-Perron F", lambda: bp_full["F"], 17.88, True),
    ("SARIMA 2024 MAPE", lambda: metrics(REV[(2024, "SARIMA")]["actual"], REV[(2024, "SARIMA")]["pred"])["MAPE (%)"], 9.60, True),
    ("SARIMAX-breaks 2024 MAPE", lambda: metrics(REV[(2024, "SARIMAX-breaks")]["actual"], REV[(2024, "SARIMAX-breaks")]["pred"])["MAPE (%)"], 9.31, True),
    ("SARIMA 2020 MAPE (constrained refit)", lambda: table13.loc["SARIMA", 2020], 21.11, True),
    ("SARIMA pooled MAPE", lambda: table13.loc["SARIMA", "Pooled MAPE (n=60)"], 13.23, True),
    ("SARIMAX-lockdown pooled MAPE", lambda: s9_ext.set_index("Model").loc["SARIMAX-lockdown", "Pooled MAPE (n=60)"], 13.33, True),
    ("Lockdown coefficient (2015-2023)", lambda: coef_tab.query("Sample == 'Training 2015-2023' and Specification == 'lockdown'")["Coefficient"].iloc[0], -990, True),
    ("Prophet 2024 MAPE", lambda: metrics(REV[(2024, "Prophet")]["actual"], REV[(2024, "Prophet")]["pred"])["MAPE (%)"], 10.34, False),
    ("LSTM 2024 MAPE (seed-avg)", lambda: metrics(REV[(2024, "LSTM")]["actual"], REV[(2024, "LSTM")]["pred"])["MAPE (%)"], 13.97, False),
    ("CNNAR 2024 MAPE (seed-avg)", lambda: metrics(REV[(2024, "CNNAR")]["actual"], REV[(2024, "CNNAR")]["pred"])["MAPE (%)"], 10.14, False),
    ("XGBoost 2024 MAPE", lambda: metrics(REV[(2024, "XGBoost")]["actual"], REV[(2024, "XGBoost")]["pred"])["MAPE (%)"], 10.61, False),
    ("Hybrid 2024 MAPE (seed-avg)", lambda: metrics(REV[(2024, MAIN)]["actual"], REV[(2024, MAIN)]["pred"])["MAPE (%)"], 8.46, False),
    ("Break-hybrid 2024 MAPE (seed-avg)", lambda: metrics(REV[(2024, "Hybrid SARIMAX(breaks)-CNNAR")]["actual"], REV[(2024, "Hybrid SARIMAX(breaks)-CNNAR")]["pred"])["MAPE (%)"], 7.39, False),
    ("Hybrid pooled MAPE", lambda: table13.loc[MAIN, "Pooled MAPE (n=60)"], 15.40, False),
    ("LSTM pooled MAPE", lambda: table13.loc["LSTM", "Pooled MAPE (n=60)"], 13.63, False),
]
check = []
for desc, getter, ref, det in REFERENCE:
    try:
        val = float(getter())
        decimals = len(str(ref).split(".")[1]) if "." in str(ref) else 0
        tol = 0.5 * 10 ** -decimals + 1e-9 if det else 0.5   # deterministic: rounding of the published value; neural/XGBoost: 0.5 MAPE points
        status = "PASS" if abs(val - ref) <= tol else ("FAIL" if det else "CHECK")
    except Exception as e:      # stage not run in this session
        val, status = np.nan, "not run"
    check.append({"Quantity": desc, "Published": ref, "This run": val, "Type": "deterministic" if det else "platform-dependent", "Status": status})
check = save_table(pd.DataFrame(check), "reproducibility_check")
display(check.round(2))
print(check["Status"].value_counts().to_dict())
if not DATA_MATCHES_PUBLISHED:
    print("Note: the input data differ from the published dataset, so differences from the published values are expected.")

# %% [markdown]
# ## 17. Final model summaries and index of all outputs
# Final SARIMA and SARIMAX models fitted to January 2015–December 2024 (the models behind the 2025 forecasts), and a list of every file written.

# %%
bp_final = bai_perron(y_all)
for label, xa in [("SARIMA_111_111_12", None), ("SARIMAX_breaks_111_111_12", step_dummies(y_all.index, bp_final["break_dates"])),
                  ("SARIMA_111_110_12_hybrid_component", None)]:
    o, so = (HYB_ORDER, HYB_SORDER) if "hybrid" in label else (SARIMA_ORDER, SARIMA_SORDER)
    res = fit_sarimax(y_all, xa, o, so)
    (DIRS["models"] / f"Model_{label}_fit_2015_2024_summary.txt").write_text(str(res.summary()))
    save_table(pd.DataFrame({"Parameter": res.params.index, "Estimate": res.params.values, "SE": res.bse.values, "p-value": res.pvalues.values}),
               f"Model_{label}_parameters")
    save_table(pd.DataFrame({"Date": y_all.index.strftime("%Y-%m"), "Observed": y_all.values, "Fitted": res.fittedvalues.values,
                             "Residual": (y_all - res.fittedvalues).values, "Start-up value (excluded)": np.arange(120) < BURN_IN}),
               f"Model_{label}_fitted_values")
save_table(pd.DataFrame(FIT_LOG).drop_duplicates(), "SARIMA_fit_log")

index_rows = [{"Folder": p.parent.name, "File": p.name, "Size (KB)": round(p.stat().st_size / 1024, 1)}
              for p in sorted(OUTPUT_DIR.rglob("*")) if p.is_file() and "_cache" not in p.parts]
index_rows.append({"Folder": OUTPUT_DIR.name, "File": "Outputs_index.csv", "Size (KB)": None})
outputs_index = save_table(pd.DataFrame(index_rows), "Outputs_index", folder=OUTPUT_DIR)
print(f"All outputs are in: {OUTPUT_DIR}")
display(outputs_index.groupby("Folder").size().rename("files").to_frame())
