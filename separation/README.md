# Stem Separation

JE AI Audio Studio keeps source separation as an optional heavy backend so the lightweight audio editor can run without a GPU.

## Current backend

- Demucs-compatible separation engine
- Default model: `htdemucs`
- Automatic CUDA/CPU device selection
- Exports each detected source as a WAV file

## Colab/GPU installation

Install the project's normal requirements first, then install a compatible Demucs release in the runtime. The application will report a clear error if the optional backend is missing.

> Note: model packages are intentionally not part of the lightweight default `requirements.txt`; this keeps normal startup fast and avoids forcing a large ML download onto every installation.
