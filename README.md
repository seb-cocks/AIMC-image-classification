# AIMC-Spec: Spectrogram-based Image Classification Benchmarks

This repository contains code and notebooks to benchmark **automatic intra-pulse modulation classification (AIMC)** models on **AIMC-Spec**, using a **unified FFT-based spectrogram** representation.

- **Paper scope:** compare *architectures* (CNN, denoising pipelines, U-Net-like, pretrained backbones, transformer) on the same input representation.
- **Dataset:** 33 modulation types × 13 SNR levels (from +10 to −20 dB).
- **Reproducibility:** fixed spectrogram parameters, manifest-driven data loading, and seedable splits.

---

## Quick Start

```bash
# 1) Create environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Recommended) Verify repo with a quick smoke test
python - <<'PY'
print("Env OK"); import torch, numpy, PIL, matplotlib, pandas; print("Imports OK")
PY
