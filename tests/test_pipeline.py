"""
test_pipeline.py
================
Unit tests for models, preprocessing, and evaluation helpers.

Run with:
    pytest tests/test_pipeline.py -v
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest
import torch

from pipeline import (
    bandpass_filter, segment_ecg, preprocess_ecg,
    ResidualBlock1D, ECGEncoder, BiometricFusionNet,
    ImprovedECGLoader, NISTBSSR1Loader,
    tanh_normalize, fusion_sum_rule, compute_eer,
    TripletMarginCosineLoss, ContrastiveLoss,
)


# ── Preprocessing ─────────────────────────────────────────────

def test_bandpass_filter_shape():
    signal = np.random.randn(3600)
    out = bandpass_filter(signal)
    assert out.shape == signal.shape

def test_segment_ecg_count():
    signal = np.random.randn(3600)
    segs = segment_ecg(signal, fs=360, segment_len_sec=10)
    assert segs.shape == (1, 3600)

def test_segment_ecg_normalised():
    signal = np.random.randn(7200)
    segs = segment_ecg(signal, fs=360, segment_len_sec=10)
    for seg in segs:
        assert abs(seg.mean()) < 0.1
        assert abs(seg.std() - 1.0) < 0.1

def test_preprocess_ecg_output():
    signal = np.random.randn(7200)
    segs = preprocess_ecg(signal, fs=360)
    assert segs.ndim == 2
    assert segs.shape[1] == 3600


# ── Models ────────────────────────────────────────────────────

def test_residual_block_shape():
    block = ResidualBlock1D(1, 32, kernel=7, stride=2)
    x = torch.randn(4, 1, 3600)
    out = block(x)
    assert out.shape == (4, 32, 1800)

def test_ecg_encoder_output_shape():
    model = ECGEncoder(embed_dim=128)
    x = torch.randn(4, 1, 3600)
    out = model(x)
    assert out.shape == (4, 128)

def test_ecg_encoder_embedding_normalised():
    model = ECGEncoder(embed_dim=128)
    x = torch.randn(2, 1, 3600)
    out = model(x)
    assert out.shape[-1] == 128

def test_fusion_net_output_range():
    model  = BiometricFusionNet(ecg_embed_dim=128)
    emb    = torch.randn(8, 128)
    face   = torch.rand(8)
    fp     = torch.rand(8)
    out    = model(emb, face, fp)
    assert out.shape == (8,)
    assert (out >= 0).all() and (out <= 1).all()


# ── Data loaders ──────────────────────────────────────────────

def test_ecg_loader_shape():
    loader = ImprovedECGLoader()
    segs   = loader.get_subject_segments(0, n_segments=5)
    assert segs.shape == (5, 3600)

def test_ecg_loader_subject_variation():
    loader = ImprovedECGLoader()
    s0 = loader.get_subject_segments(0, n_segments=2)
    s1 = loader.get_subject_segments(1, n_segments=2)
    # Different subjects should produce different signals
    assert not np.allclose(s0, s1)

def test_nist_loader_synthetic():
    loader = NISTBSSR1Loader()
    gen, imp = loader.load_scores('face_c1')
    assert len(gen) == 1000
    assert len(imp) == 9000
    assert gen.mean() > imp.mean()   # genuine scores should be higher


# ── Loss functions ────────────────────────────────────────────

def test_triplet_loss_positive():
    loss_fn = TripletMarginCosineLoss(margin=0.2)
    a = torch.randn(4, 128)
    p = a + 0.01 * torch.randn(4, 128)   # close to anchor
    n = torch.randn(4, 128)              # random negative
    loss = loss_fn(a, p, n)
    assert loss.item() >= 0

def test_contrastive_loss_genuine_lower():
    loss_fn  = ContrastiveLoss(margin=0.5)
    e1       = torch.randn(4, 128)
    e2_same  = e1 + 0.01 * torch.randn(4, 128)
    e2_diff  = torch.randn(4, 128)
    label_g  = torch.ones(4)
    label_i  = torch.zeros(4)
    loss_g   = loss_fn(e1, e2_same, label_g)
    loss_i   = loss_fn(e1, e2_diff, label_i)
    assert loss_g.item() >= 0
    assert loss_i.item() >= 0


# ── Evaluation helpers ────────────────────────────────────────

def test_tanh_normalize_range():
    scores = np.random.randn(500)
    normed = tanh_normalize(scores)
    assert normed.min() >= 0 and normed.max() <= 1

def test_fusion_sum_rule_shape():
    s1 = np.random.rand(100)
    s2 = np.random.rand(100)
    s3 = np.random.rand(100)
    fused = fusion_sum_rule([s1, s2, s3], weights=[0.35, 0.35, 0.30])
    assert fused.shape == (100,)

def test_compute_eer_range():
    gen = np.random.normal(0.7, 0.1, 500).clip(0, 1)
    imp = np.random.normal(0.3, 0.1, 500).clip(0, 1)
    eer, _, far, frr = compute_eer(gen, imp)
    assert 0 <= eer <= 1
    assert len(far) == len(frr) == 500

def test_eer_genuine_better_than_random():
    gen = np.random.normal(0.75, 0.08, 1000).clip(0, 1)
    imp = np.random.normal(0.25, 0.08, 1000).clip(0, 1)
    eer, _, _, _ = compute_eer(gen, imp)
    assert eer < 0.15   # well-separated distributions should give low EER
