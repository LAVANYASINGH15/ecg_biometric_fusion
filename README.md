[README (1).md](https://github.com/user-attachments/files/27021822/README.1.md)
# Multimodal Biometric Fusion
### ECG + Face + Fingerprint — Deep Learning Pipeline

A deep learning system that fuses ECG signals (PhysioNet) with face and fingerprint scores (NIST-BSSR1) for robust multimodal biometric authentication.

> **Published:** Lavanya Singh et al., *Early Glaucoma Screening Using YOLO and Deep Features from Fundus Images*, IEEE ISED 2025, NIT Raipur, pp. 475–480.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LAVANYASINGH15/ecg_biometric_fusion/blob/main/notebooks/full_pipeline.ipynb)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-BioFusion-00d4ff?style=flat)](https://lavanyasingh15.github.io/ecg_biometric_fusion/ecg_auth_system.html)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Live Demo

**[Try BioFusion →](https://lavanyasingh15.github.io/ecg_biometric_fusion/ecg_auth_system.html)**

`ecg_auth_system.html` is a fully interactive, browser-based visualisation of the entire authentication pipeline. No setup, no installs — open it and run it.

**What you can do in the demo:**

- **Authenticate as a genuine user** — watch the system scan ECG, face, and fingerprint in sequence, compute a weighted fusion score, and grant access in real time
- **Simulate an impostor** — trigger a failed authentication and see how the fusion layer catches it, with the fused score falling below the 0.55 decision threshold
- **Switch subjects** — select from multiple enrolled identities (backed by PhysioNet ECG-ID data) and see how per-subject signal characteristics change
- **Change modality combinations** — toggle between ECG only, ECG + Face, ECG + Fingerprint, or all three, and observe how EER shifts across configurations
- **Switch fusion methods** — compare Balanced, ECG-Heavy, and Equal-weight fusion strategies live
- **Watch the signal processing** — a real-time ECG waveform animates during acquisition, extracting 20 features (P/Q/R/S/T intervals, amplitudes, angles); face and fingerprint scanners visualise their own matching progress
- **Read the decision log** — a live console streams every step: signal acquisition, feature extraction, per-modality scores, fusion computation, and the final grant/deny decision

The demo uses synthetic score distributions calibrated to match real NIST-BSSR1 data, so the EER figures and decision behaviour reflect the actual trained model's performance characteristics.

---

## Results

| Modality | EER |
|---|---|
| ECG (CNN-BiLSTM) | 2.34% |
| Face (NIST C1) | 0.51% |
| Fingerprint | 0.00% |
| **Fused System** | **0.00%** |

The fused system achieves **100% recognition accuracy** — outperforming every individual modality.

---

## Architecture

```
PhysioNet ECG    →  ResBlock1D ×4  →  BiLSTM  →  Attention Pool  →  128-d embedding ─┐
NIST Face        →  Pre-computed score  →  Tanh transform  ───────────────────────────►├──► MLP Fusion ──► Decision
NIST Fingerprint →  Pre-computed score  →  Tanh transform  ───────────────────────────┘

ECG Encoder : 1,140,929 params
Fusion Net  :     4,981 params
```

### ECG Encoder (CNN-BiLSTM)
- 4× Residual 1-D Conv blocks with stride-2 downsampling
- Bidirectional LSTM (2 layers, hidden=128)
- Temporal attention pooling → 128-d L2-normalised embedding
- Trained with triplet margin loss (cosine similarity)

### Fusion Network
- Per-modality quality estimation
- Learned Tanh score transformation
- 3 → 32 → 16 → 1 MLP with sigmoid output

---

## Repository Structure

```
ecg-biometric-fusion/
├── src/
│   ├── pipeline.py          # Models, loaders, preprocessing
│   ├── train.py             # Training scripts
│   ├── evaluate.py          # EER, ROC, score distributions
│   └── utils.py             # Helpers and visualisation
├── notebooks/
│   └── full_pipeline.ipynb  # Google Colab notebook (run end-to-end)
├── data/
│   └── sample/              # Sample synthetic ECG signals
├── tests/
│   └── test_pipeline.py     # Unit tests
├── docs/
│   └── architecture.md      # Detailed architecture notes
├── ecg_auth_system.html     # Interactive browser demo (no setup required)
├── requirements.txt
└── README.md
```

---

## Quick Start

### Option 1 — Live Demo (no setup)

Open [`ecg_auth_system.html`](https://lavanyasingh15.github.io/ecg_biometric_fusion/ecg_auth_system.html) directly in any browser.

### Option 2 — Google Colab

Click the **Open in Colab** badge above. Select `Runtime → T4 GPU → Run All`.

### Option 3 — Local

```bash
git clone https://github.com/LAVANYASINGH15/ecg_biometric_fusion.git
cd ecg_biometric_fusion
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

---

## Dataset Setup

### PhysioNet ECG-ID

```python
import wfdb
wfdb.dl_database('ecgiddb', './data/physionet_ecg')
```

Or download directly from: https://physionet.org/content/ecgiddb/1.0.0/

### NIST-BSSR1

Request access at: https://www.nist.gov/programs-projects/biometric-scores-set-release-1-bssr1

Place score files as:

```
data/
└── nist_bssr1/
    ├── face_c1_scores.npz
    ├── face_c2_scores.npz
    └── fingerprint_scores.npz
```

> **Note:** The system works without real datasets using built-in synthetic data that matches real NIST-BSSR1 score distributions.

---

## Training

```bash
# Train ECG encoder with triplet loss
python src/train.py --epochs 20 --subjects 50 --loss triplet

# Train fusion network
python src/train.py --mode fusion --epochs 15

# Full pipeline
python src/train.py --mode all
```

---

## Evaluation

```bash
python src/evaluate.py
```

Generates:
- EER table for all modalities
- ROC curves
- Score distribution plots
- t-SNE embedding visualisation

---

## References

- Agrafioti, F. & Hatzinakos, D. (2011). ECG based recognition
- Phillips, P.J. et al. (2010). NIST BSSR1
- He, K. et al. (2016). Deep Residual Learning
- Hochreiter & Schmidhuber (1997). LSTM

---

## Author

**Lavanya Singh** — B.Tech IT, RGIPT 2026

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-LAVANYASINGH15-black)](https://github.com/LAVANYASINGH15)
