#!/usr/bin/env bash
# Fix numpy/scipy mismatch after installing ROS into perception env.
# Do NOT pip uninstall numpy — that deletes conda numpy without reinstalling files.
set -eo pipefail

source /root/miniforge3/etc/profile.d/conda.sh
set +u
conda activate perception
set -u

echo "Before:"
python -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null || echo "numpy missing/broken"

mamba install --force-reinstall -y -c conda-forge numpy=2.2.6 scipy=1.15.2
mamba install -y -c conda-forge pyparsing python-dateutil typing_extensions

echo "After:"
python -c "import numpy, scipy; from scipy import interpolate; print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('OK')"
