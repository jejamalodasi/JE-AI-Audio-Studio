# Termux setup notes

JE AI Audio Studio is primarily designed to run the Python/AI stack in Google Colab or a Linux server. Termux on Android is useful for repository work and lightweight checks, but Python packages with native scientific/ML extensions do not always have normal PyPI Android wheels.

## Why `pip install numpy` failed

If this command returns:

```
ERROR: Could not find a version that satisfies the requirement numpy
ERROR: No matching distribution found for numpy
```

do not keep retrying `pip install --only-binary=:all: numpy`. Termux is not a normal Linux wheel target.

Termux maintainers document that native packages such as NumPy are commonly supplied through the Termux package manager rather than PyPI wheels. Use:

```bash
pkg update
pkg install python python-numpy -y
python -c "import numpy; print(numpy.__version__)"
```

For SciPy, the package may be provided through the Termux User Repository (TUR):

```bash
pkg install tur-repo -y
pkg install python-scipy -y
python -c "import scipy; print(scipy.__version__)"
```

The exact package availability depends on the current Termux/Python version. In particular, Python major-version transitions can temporarily break native scientific packages.

## Important

Do **not** run the Colab notebook's shell cells directly as the Termux installation method. The notebook install block is intended for Google Colab/Linux.

The Colab workflow remains:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-ai.txt
python -m pip install -r requirements-api.txt
```

For Termux, first make sure the native NumPy/SciPy stack imports successfully before attempting audio packages such as librosa.

## Current project strategy

- **Colab / Linux server:** full Gradio + optional AI stack.
- **Android/Termux:** source control, lightweight development, API/client work, and diagnostics.
- **Heavy neural generation:** keep server-side; do not try to bundle Python/ML model weights into the Android client.

If Termux reports a native-extension error after a Python upgrade, capture these diagnostics before changing anything:

```bash
python --version
uname -m
pkg list-installed | grep -E '^(python|python-numpy|python-scipy|libopenblas|libsndfile)'
python -c "import sys; print(sys.executable); print(sys.version)"
```
