# Architecture Details

## ECG Encoder (CNN-LSTM)

```
Input: (batch, 1, 3600)   ← 10-second ECG at 360 Hz

ResidualBlock1D(1  → 32,  kernel=15, stride=2)  → (batch, 32,  1800)
ResidualBlock1D(32 → 64,  kernel=11, stride=2)  → (batch, 64,   900)
ResidualBlock1D(64 → 128, kernel=7,  stride=2)  → (batch, 128,  450)
ResidualBlock1D(128→ 128, kernel=5,  stride=2)  → (batch, 128,  225)

Permute → (batch, 225, 128)

BiLSTM(128 → 256, layers=2, dropout=0.3)        → (batch, 225, 256)

Temporal attention: softmax(Linear(256→1))       → (batch, 225, 1)
Weighted sum                                     → (batch, 256)

Linear(256 → 128) + LayerNorm                   → (batch, 128)
```

**Total params: 1,140,929**

---

## Fusion Network

```
ECG embedding (128-d)
    └─► Quality estimator: Linear(128→32)→GELU→Linear(32→1)→Sigmoid  → quality weight q
    └─► Score transform:   Linear(1→16)→Tanh→Linear(16→1)            → s_ecg

Face score (scalar)
    └─► Score transform:   Linear(1→16)→Tanh→Linear(16→1)            → s_face

Fingerprint score (scalar)
    └─► Score transform:   Linear(1→16)→Tanh→Linear(16→1)            → s_fp

Concat [s_ecg, s_face, s_fp]  → (batch, 3)
    └─► Linear(3→32)→GELU→Dropout(0.2)→Linear(32→16)→GELU→Linear(16→1)→Sigmoid

Output: fused score in [0, 1]
```

**Total params: 4,981**

---

## Training Strategy

### ECG Encoder
- **Loss**: Triplet margin loss (cosine similarity space, margin=0.2)
- **Optimiser**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Epochs**: 20
- **Batch size**: 32
- **Triplet mining**: online random — anchor/positive = same subject, negative = random different subject

### Fusion Network
- **Loss**: Binary Cross-Entropy
- **Optimiser**: AdamW (lr=1e-3)
- **Epochs**: 15
- **ECG encoder**: frozen during fusion training
- **NIST scores**: simulated from real BSSR1 score distributions

---

## Score Normalisation

Tanh normalisation maps raw matcher scores to [0, 1]:

```
normalised = 0.5 × (1 + tanh(0.01 × (score − μ) / σ))
```

where μ and σ are estimated from training genuine scores.

---

## Fusion Rule

Weighted sum rule:

```
fused = 0.35 × ECG_score + 0.35 × Face_score + 0.30 × Fingerprint_score
```

Weights are set by the learned fusion MLP in the deep learning version,
or manually tuned in the classical baseline.
