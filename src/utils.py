"""
utils.py
========
Utility functions: model loading, inference, and quick verification.
"""

import numpy as np
import torch
from pipeline import ECGEncoder, BiometricFusionNet, ImprovedECGLoader, preprocess_ecg


def load_models(encoder_path: str, fusion_path: str, embed_dim: int = 128, device=None):
    """Load trained encoder and fusion model from disk."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = ECGEncoder(embed_dim=embed_dim).to(device)
    fusion  = BiometricFusionNet(ecg_embed_dim=embed_dim).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    fusion.load_state_dict(torch.load(fusion_path,   map_location=device))
    encoder.eval(); fusion.eval()
    return encoder, fusion, device


def verify(encoder, fusion, ecg_segment: np.ndarray,
           face_score: float, fp_score: float,
           threshold: float = 0.5, device=None) -> dict:
    """
    Run a single verification trial.

    Args:
        ecg_segment : preprocessed ECG segment (1-D numpy array, length T)
        face_score  : normalised face matcher score in [0, 1]
        fp_score    : normalised fingerprint matcher score in [0, 1]
        threshold   : decision threshold (default 0.5)

    Returns:
        dict with keys: fused_score, decision, ecg_embedding
    """
    if device is None:
        device = next(encoder.parameters()).device

    ecg_t   = torch.tensor(ecg_segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    face_t  = torch.tensor([face_score], dtype=torch.float32).to(device)
    fp_t    = torch.tensor([fp_score],   dtype=torch.float32).to(device)

    with torch.no_grad():
        emb   = encoder(ecg_t)
        score = fusion(emb, face_t, fp_t).item()

    return {
        'fused_score':   round(score, 4),
        'decision':      'ACCEPT' if score >= threshold else 'REJECT',
        'ecg_embedding': emb.cpu().numpy()[0],
    }


def cosine_similarity_score(encoder, seg1: np.ndarray, seg2: np.ndarray, device=None) -> float:
    """Return cosine similarity between two ECG segments."""
    if device is None:
        device = next(encoder.parameters()).device
    t1 = torch.tensor(seg1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    t2 = torch.tensor(seg2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        e1, e2 = encoder(t1), encoder(t2)
    import torch.nn.functional as F
    return F.cosine_similarity(e1, e2).item()


def quick_demo(encoder_path='ecg_encoder.pt', fusion_path='fusion_model.pt'):
    """Run a quick genuine vs impostor demo."""
    encoder, fusion, device = load_models(encoder_path, fusion_path)
    loader = ImprovedECGLoader()

    print('\n── Quick Verification Demo ──')
    # Genuine: same subject, different segments
    segs0 = loader.get_subject_segments(0, n_segments=3)
    result = verify(encoder, fusion, segs0[0], face_score=0.78, fp_score=0.85, device=device)
    print(f'Genuine trial  →  score={result["fused_score"]}  {result["decision"]}')

    # Impostor: different subject
    segs1 = loader.get_subject_segments(1, n_segments=3)
    result = verify(encoder, fusion, segs1[0], face_score=0.25, fp_score=0.20, device=device)
    print(f'Impostor trial →  score={result["fused_score"]}  {result["decision"]}')
    print()


if __name__ == '__main__':
    quick_demo()
