#!/data/data/com.termux/files/usr/bin/bash
set -u

echo "=== JE AI Audio Studio / Termux diagnostics ==="
echo

echo "[Python]"
python --version 2>&1 || true
python -c 'import sys; print(sys.executable); print(sys.version)' 2>&1 || true
echo

echo "[Architecture]"
uname -m 2>&1 || true
echo

echo "[NumPy]"
python -c 'import numpy; print(numpy.__version__); print(numpy.__file__)' 2>&1 || true
echo

echo "[SciPy]"
python -c 'import scipy; print(scipy.__version__); print(scipy.__file__)' 2>&1 || true
echo

echo "[Audio libraries]"
python -c 'import soundfile; print("soundfile", soundfile.__version__)' 2>&1 || true
python -c 'import librosa; print("librosa", librosa.__version__)' 2>&1 || true
echo

echo "[Termux packages]"
pkg list-installed 2>/dev/null | grep -E '^(python|python-numpy|python-scipy|libopenblas|libsndfile|libsoxr)' || true
echo

echo "=== End diagnostics ==="
