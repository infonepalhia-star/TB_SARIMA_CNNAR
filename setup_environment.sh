#!/bin/bash
# =====================================================================
# One-time setup of a clean Python environment for TB_SARIMA_CNNAR_Analysis.ipynb
#
# Usage (macOS Terminal), from the folder that contains this file:
#     bash setup_environment.sh
#
# It creates the conda environment "tb_forecast" (Python 3.12), installs mutually
# compatible versions of all packages (pip picks the right TensorFlow/NumPy pair
# for Apple-silicon or Intel Macs), registers the Jupyter kernel
# "Python (tb_forecast)" and tests the imports. Your base environment is not changed.
# =====================================================================
set -e
ENV_NAME="tb_forecast"
cd "$(dirname "$0")"

# locate conda (also works when 'conda init' has not been run)
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
for c in /Volumes/Untitled/opt/anaconda3/bin/conda "$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda" /opt/anaconda3/bin/conda /opt/miniconda3/bin/conda; do
    [ -z "$CONDA_BIN" ] && [ -x "$c" ] && CONDA_BIN="$c"
done
if [ -z "$CONDA_BIN" ]; then echo "conda not found - install Miniconda or Anaconda first."; exit 1; fi
echo "Using conda: $CONDA_BIN   |   Mac architecture: $(uname -m)"
eval "$("$CONDA_BIN" shell.bash hook 2>/dev/null)"

# The 'classic' solver avoids conda-libmamba-solver, which fails in some Anaconda installs
# ("Symbol not found ... libc++.1.0.dylib"). It only affects how conda resolves packages.
export CONDA_SOLVER="${CONDA_SOLVER:-classic}"

if conda env list 2>/dev/null | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "Environment '$ENV_NAME' exists - removing it to start clean."
    conda env remove -n "$ENV_NAME" -y || true
fi
# llvm-openmp provides libomp.dylib, which XGBoost needs on macOS
if conda create -n "$ENV_NAME" -c conda-forge python=3.12 llvm-openmp -y && conda activate "$ENV_NAME"; then
    echo "Created conda environment '$ENV_NAME'."
else
    # Fallback: a plain virtual environment next to this script (uses pip wheels only)
    echo "conda could not create the environment - using a Python virtual environment instead."
    BASE_PY="$(dirname "$CONDA_BIN")/python3"
    [ -x "$BASE_PY" ] || BASE_PY="$(command -v python3)"
    "$BASE_PY" -m venv "$PWD/.venv_tb_forecast"
    source "$PWD/.venv_tb_forecast/bin/activate"
    if [ "$(uname)" = "Darwin" ] && command -v brew >/dev/null; then brew install libomp || true; fi
fi
echo "Python in use: $(which python)  ($(python --version))"

python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

python - <<'PY'
import numpy, pandas, PIL.Image, matplotlib.pyplot, statsmodels, tensorflow as tf, xgboost, prophet, ruptures
print(f"\nAll packages import correctly: numpy {numpy.__version__} | pandas {pandas.__version__} | "
      f"tensorflow {tf.__version__} | statsmodels {statsmodels.__version__} | xgboost {xgboost.__version__}")
PY
echo ""
echo "Done. In Jupyter: open TB_SARIMA_CNNAR_Analysis.ipynb -> Kernel -> Change kernel -> 'Python ($ENV_NAME)'"
echo "      then Kernel -> Restart Kernel and Run All Cells."
