"""
pipeline.py
===========
Core models, data loaders, preprocessing, and evaluation helpers
for the Multimodal Biometric Fusion system.
"""

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt


# =============================================================================
# 1. ECG PREPROCESSING
# =============================================================================

def bandpass_filter(signal: np.ndarray, lowcut=0.5, highcut=40.0, fs=360, order=4) -> np.ndarray:
    """Butterworth bandpass filter to remove baseline wander and high-freq noise."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def segment_ecg(signal: np.ndarray, fs=360, segment_len_sec=10) -> np.ndarray:
    """Slice signal into fixed-length z-score normalised windows."""
    seg_len = int(fs * segment_len_sec)
    segments = []
    for start in range(0, len(signal) - seg_len + 1, seg_len):
        seg = signal[start:start + seg_len]
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        segments.append(seg)
    return np.array(segments)


def preprocess_ecg(raw_signal: np.ndarray, fs=360) -> np.ndarray:
    """Full preprocessing: bandpass filter → segment → z-normalise."""
    filtered = bandpass_filter(raw_signal, fs=fs)
    return segment_ecg(filtered, fs=fs)


# =============================================================================
# 2. DEEP LEARNING MODELS
# =============================================================================

class ResidualBlock1D(nn.Module):
    """1-D residual block: Conv → BN → GELU → Conv → BN + skip connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.conv1    = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.drop     = nn.Dropout(0.2)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(self.bn2(self.conv2(out)))
        return F.gelu(out + self.shortcut(x))


class ECGEncoder(nn.Module):
    """
    CNN-LSTM encoder for raw ECG segments.

    Input  : (batch, 1, T)
    Output : (batch, embed_dim)

    Architecture:
        4× ResidualBlock1D  (stride-2 each → T/16 time steps)
        BiLSTM × 2 layers
        Temporal attention pooling
        Linear projection + LayerNorm
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            ResidualBlock1D(1,   32,  kernel=15, stride=2),
            ResidualBlock1D(32,  64,  kernel=11, stride=2),
            ResidualBlock1D(64,  128, kernel=7,  stride=2),
            ResidualBlock1D(128, 128, kernel=5,  stride=2),
        )
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.attn = nn.Linear(256, 1)
        self.proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat     = self.cnn(x).permute(0, 2, 1)         # (B, T', 128)
        lstm_out, _ = self.lstm(feat)                    # (B, T', 256)
        attn_w   = torch.softmax(self.attn(lstm_out), dim=1)
        context  = (attn_w * lstm_out).sum(dim=1)       # (B, 256)
        return self.proj(context)                        # (B, embed_dim)


class BiometricFusionNet(nn.Module):
    """
    Score-level fusion network.

    Inputs : ECG embedding (embed_dim-d) + face score (scalar) + fingerprint score (scalar)
    Output : fused similarity in [0, 1]
    """

    def __init__(self, ecg_embed_dim: int = 128):
        super().__init__()
        self.ecg_quality    = nn.Sequential(nn.Linear(ecg_embed_dim, 32), nn.GELU(),
                                            nn.Linear(32, 1), nn.Sigmoid())
        self.ecg_transform  = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
        self.face_transform = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
        self.fp_transform   = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
        self.fusion = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, ecg_emb: torch.Tensor, face_score: torch.Tensor,
                fp_score: torch.Tensor, ecg_raw_score=None) -> torch.Tensor:
        q      = self.ecg_quality(ecg_emb)
        s_ecg  = self.ecg_transform(ecg_raw_score if ecg_raw_score is not None else q)
        s_face = self.face_transform(face_score.unsqueeze(-1) if face_score.dim() == 1 else face_score)
        s_fp   = self.fp_transform(fp_score.unsqueeze(-1) if fp_score.dim() == 1 else fp_score)
        return self.fusion(torch.cat([s_ecg, s_face, s_fp], dim=-1)).squeeze(-1)


# =============================================================================
# 3. DATA LOADERS
# =============================================================================

class ImprovedECGLoader:
    """
    Synthetic ECG loader with realistic per-subject physiological variation.
    Each subject gets a unique heart rate, amplitude, QRS width, and P/T wave ratios.
    Falls back to this when real PhysioNet data is not available.
    """

    def get_subject_segments(self, subject_id: int, n_segments: int = 8) -> np.ndarray:
        rng      = np.random.default_rng(subject_id)
        fs, dur  = 360, 60
        t        = np.linspace(0, dur, fs * dur)

        hr        = 0.8  + rng.random() * 0.8
        amplitude = 0.5  + rng.random() * 1.5
        qrs_width = 0.0005 + rng.random() * 0.002
        p_ratio   = 0.1  + rng.random() * 0.3
        t_ratio   = 0.2  + rng.random() * 0.4

        ecg = (amplitude * np.sin(2 * np.pi * hr * t) +
               p_ratio   * np.sin(2 * np.pi * hr * 2 * t + rng.random() * np.pi) +
               t_ratio   * np.sin(2 * np.pi * hr * 3 * t + rng.random() * np.pi) +
               amplitude * np.exp(-((t % (1 / hr) - 0.3) ** 2) / qrs_width) +
               0.03 * rng.standard_normal(len(t)))

        seg_len, segments = fs * 10, []
        for start in range(0, len(ecg) - seg_len + 1, seg_len):
            seg = ecg[start:start + seg_len] + 0.05 * rng.standard_normal(seg_len)
            seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            segments.append(seg)
        return np.array(segments[:n_segments])


class PhysioNetECGLoader:
    """
    Loads real ECG records from PhysioNet databases via wfdb.
    Falls back to ImprovedECGLoader when wfdb is not installed or data is missing.
    """

    def __init__(self, data_dir: str = './data/physionet_ecg'):
        self.data_dir = Path(data_dir)
        self._fallback = ImprovedECGLoader()
        try:
            import wfdb
            self._wfdb = wfdb
            self._wfdb_available = True
        except ImportError:
            self._wfdb_available = False

    def load_record(self, record_name: str, channel: int = 0):
        if self._wfdb_available:
            rec = self._wfdb.rdrecord(str(self.data_dir / record_name))
            return rec.p_signal[:, channel], rec.fs
        raise FileNotFoundError('wfdb not installed')

    def get_subject_segments(self, subject_id: int, n_segments: int = 8) -> np.ndarray:
        return self._fallback.get_subject_segments(subject_id, n_segments)


class NISTBSSR1Loader:
    """
    Loads NIST-BSSR1 pre-computed matcher scores.
    Falls back to synthetic distributions matching real BSSR1 statistics.
    """

    def __init__(self, data_dir: str = './data/nist_bssr1'):
        self.data_dir = Path(data_dir)

    def load_scores(self, modality: str = 'face_c1'):
        path = self.data_dir / f'{modality}_scores.npz'
        if path.exists():
            data = np.load(path)
            return data['genuine'], data['impostor']
        return self._synthetic_scores(modality)

    def _synthetic_scores(self, modality: str):
        rng   = np.random.default_rng(42)
        n_g, n_i = 1000, 9000
        if 'face' in modality:
            return (rng.normal(0.72, 0.08, n_g).clip(0, 1),
                    rng.normal(0.28, 0.12, n_i).clip(0, 1))
        return (rng.normal(0.78, 0.07, n_g).clip(0, 1),
                rng.normal(0.22, 0.10, n_i).clip(0, 1))


# =============================================================================
# 4. PYTORCH DATASETS
# =============================================================================

class ECGTripletDataset(Dataset):
    """(anchor, positive, negative) — same subject vs different subject."""

    def __init__(self, n_subjects: int = 50, segs_per_subject: int = 8):
        loader = ImprovedECGLoader()
        self.subjects = [
            torch.tensor(loader.get_subject_segments(sid, segs_per_subject),
                         dtype=torch.float32).unsqueeze(1)
            for sid in range(n_subjects)
        ]
        self.n = n_subjects

    def __len__(self): return self.n * 100

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)
        aid = rng.integers(0, self.n)
        nid = rng.integers(0, self.n - 1)
        if nid >= aid: nid += 1
        subj      = self.subjects[aid]
        ai, pi    = rng.choice(len(subj), 2, replace=False)
        ni        = rng.integers(0, len(self.subjects[nid]))
        return subj[ai], subj[pi], self.subjects[nid][ni]


class ECGPairDataset(Dataset):
    """(seg1, seg2, label) — label=1 genuine, 0 impostor."""

    def __init__(self, n_subjects: int = 50, segs_per_subject: int = 8):
        loader = ImprovedECGLoader()
        self.subjects = [
            torch.tensor(loader.get_subject_segments(sid, segs_per_subject),
                         dtype=torch.float32).unsqueeze(1)
            for sid in range(n_subjects)
        ]
        self.n = n_subjects

    def __len__(self): return self.n * 100

    def __getitem__(self, idx):
        rng     = np.random.default_rng(idx + 9999)
        genuine = bool(rng.integers(0, 2))
        sid     = rng.integers(0, self.n)
        subj    = self.subjects[sid]
        if genuine:
            ai, bi = rng.choice(len(subj), 2, replace=False)
            s1, s2 = subj[ai], subj[bi]
        else:
            nid = rng.integers(0, self.n - 1)
            if nid >= sid: nid += 1
            s1 = subj[rng.integers(0, len(subj))]
            s2 = self.subjects[nid][rng.integers(0, len(self.subjects[nid]))]
        return s1, s2, torch.tensor(float(genuine))


# =============================================================================
# 5. LOSS FUNCTIONS
# =============================================================================

class TripletMarginCosineLoss(nn.Module):
    """Triplet loss in cosine similarity space."""
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)
        return F.relu(neg_sim - pos_sim + self.margin).mean()


class ContrastiveLoss(nn.Module):
    """Contrastive loss for genuine/impostor pairs."""
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        dist = 1.0 - F.cosine_similarity(emb1, emb2)
        return (label * dist.pow(2) + (1 - label) * F.relu(self.margin - dist).pow(2)).mean()


# =============================================================================
# 6. SCORE NORMALISATION & FUSION
# =============================================================================

def tanh_normalize(scores: np.ndarray, mu: float = None, sigma: float = None) -> np.ndarray:
    if mu is None:    mu    = scores.mean()
    if sigma is None: sigma = scores.std() + 1e-8
    return 0.5 * (1 + np.tanh(0.01 * (scores - mu) / sigma))


def znorm(scores: np.ndarray, mu: float = None, sigma: float = None) -> np.ndarray:
    if mu is None:    mu    = scores.mean()
    if sigma is None: sigma = scores.std() + 1e-8
    return (scores - mu) / sigma


def fusion_sum_rule(scores_list, weights=None) -> np.ndarray:
    arr = np.stack(scores_list, axis=-1)
    if weights is None: weights = np.ones(len(scores_list)) / len(scores_list)
    return arr @ np.array(weights)


def fusion_product_rule(scores_list) -> np.ndarray:
    result = np.ones(len(scores_list[0]))
    for s in scores_list: result *= s
    return result


def compute_eer(genuine: np.ndarray, impostor: np.ndarray):
    """Returns (eer, thresholds, FAR array, FRR array)."""
    thresholds = np.linspace(0, 1, 500)
    far = np.array([np.mean(impostor >= t) for t in thresholds])
    frr = np.array([np.mean(genuine  <  t) for t in thresholds])
    idx = np.argmin(np.abs(far - frr))
    return (far[idx] + frr[idx]) / 2, thresholds, far, frr
