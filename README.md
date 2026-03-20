# AIMC-Spec Dataset (Updated Version)

### Spectrogram-Based Radar Intra-Pulse Modulation Classification Benchmark

This repository contains the experimental framework used to benchmark **Automatic Intra-Pulse Modulation Classification (AIMC)** models using a **unified spectrogram representation**.

The project evaluates multiple deep learning architectures on the **same spectrogram input pipeline**, allowing fair comparison of model performance.

This repository hosts the updated release of the AIMC-Spec dataset, a synthetic benchmark designed for reproducible evaluation of Automatic Intrapulse Modulation Classification (AIMC) algorithms.

This version introduces several important improvements over the initial AIMC-Spec release, including:

- redesigned storage pipeline reducing dataset size from >500 GB to <30 GB
- refined modulation set and standardized signal definitions
- streamlined dataset generation process for improved usability
- updated benchmark protocol and experimental evaluation

The updated dataset is fully described in the following paper:
AIMC-Spec: An Updated Benchmark Dataset for Automatic Intrapulse Modulation Classification
arXiv: https://arxiv.org/abs/2601.08265

### Users are strongly encouraged to consult the paper for:
- dataset design methodology
- signal generation details
- SNR definition and preprocessing
- benchmark evaluation protocol
- recommended train/validation/test usage

---

# Research Focus

The goal of this repository is to compare **model architectures** rather than input representations.

All models use the **same power spectrogram representation** generated from IQ radar data.

Architectures currently implemented include:

- **STFT-CNN**
- **LPI-Net**
- **LDC-UNet**
- **CDAE-DCNN**
- **Vision Transformer (ViT)**

Each architecture has its own training module but uses the same dataset pipeline and experiment framework.

---

# Dataset

The framework operates on the **AIMC-Spec dataset**, which contains radar emitter signals with different intra-pulse modulations.

## Modulation Classes

33 intra-pulse modulation types including:

### FM Family
- unmodulated
- LFM up/down
- NLFM variants
- multi-LFM
- dual-LFM

### Phase-Coded
- BPSK
- QPSK
- Barker codes
- P1–P4 polyphase codes

### Additional Modulations
- triangle
- exponential
- BFSK
- 4FSK
- EQFM
- SFM
- Costas

## SNR Levels

Experiments typically use:

```
+6, +3, 0, −3, −6 dB
```

These levels simulate progressively more challenging radar environments.

---

# Repository Structure

```
src/
│
├─ main.py
│
├─ build_iq_train_subset.py
├─ export_spectrogram_memmap.py
│
├─ common/
│   ├─ config.py
│   ├─ training.py
│   ├─ evaluation.py
│   ├─ metrics.py
│   ├─ spectrogram.py
│   ├─ memmap.py
│   └─ MultiHotStreamLoader.py
│
├─ STFT_CNN/
├─ LPI_Net/
├─ LDC_Unet/
├─ CDAE_DCNN/
└─ ViT/
```

### `common/`

Contains shared infrastructure used by all models:

- dataset loading
- spectrogram generation
- training loops
- evaluation utilities
- metrics
- checkpoint management

### Model Directories

Each model architecture has its own directory containing:

- model definition
- experiment configuration
- training entrypoint

---

# Dataset Preparation Pipeline

The training pipeline operates in **three stages**.

---

## 1️⃣ Build Training Subset

Creates a smaller training dataset from the full emitter dataset.

```
build_train_subset()
```

Input:

```
AIMC_Spec_v2_train_ALL/
```

Output:

```
AIMC_Spec_v2_train/
```

---

## 2️⃣ Export Spectrogram Dataset

Converts the HDF5 IQ dataset into **memmap spectrogram files**.

```
export_main()
```

Output directory example:

```
power_specs_memmap/
```

Files produced:

```
power_u8.dat
yend_u8.dat
snr_i16.dat
mod_i16.dat
stream_i32.dat
k_u8.dat
meta.json
```

These files contain spectrogram images and aligned labels for efficient training.

---

## 3️⃣ Train Models

Each model can then be trained on the exported dataset.

Example models:

```
stft_main()
ldc_main()
lpi_main()
cdae_main()
vit_main()
```

Outputs are written to:

```
out/<model_name>/
checkpoints/<model_name>/
```

---

# Running Experiments

Run the project entrypoint:

```bash
python -m src.main
```

Inside `main.py`, enable the stage you want to run.

Example:

```python
if __name__ == "__main__":

    stft_main()
```

Only **one stage should normally be enabled at a time**.

---

# Training Outputs

Training generates:

```
out/<model_name>/
    results
    confusion matrices
    logs

checkpoints/<model_name>/
    saved model weights
```

---

# Reproducibility

Experiments use:

- fixed spectrogram generation parameters
- deterministic dataset splits
- configurable training seeds
- consistent dataset loader across models

All architectures therefore operate on the **same dataset and input representation**.

---

# Requirements

Python **3.10+ recommended**.

Install dependencies:

```bash
pip install -r requirements.txt
```

Quick environment test:

```bash
python - <<'PY'
print("Env OK")
import torch, numpy, matplotlib, pandas
print("Imports OK")
PY
```

---

# Citation

If you use this codebase or the AIMC-Spec dataset, please cite the associated publication.

@misc{cocks2026,
  title         = {AIMC-Spec: A Benchmark Dataset for Automatic Intrapulse Modulation Classification under Variable Noise Conditions},
  author        = {Sebastian L. Cocks and Salvador Dreo and Brian Ng and Feras Dayoub},
  year          = {2026},
  eprint        = {2601.08265},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  note          = {arXiv:2601.08265v2}
}

---

# Notes

This repository is part of ongoing research on **radar intra-pulse modulation classification** and the development of **AIMC-Spec benchmarking frameworks**.
