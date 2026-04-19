"""
train.py
========
Training script for ECG encoder and fusion network.

Usage:
    python src/train.py                        # train everything
    python src/train.py --mode encoder         # encoder only
    python src/train.py --mode fusion          # fusion only
    python src/train.py --epochs 50 --subjects 100
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline import (
    ECGEncoder, BiometricFusionNet,
    ECGTripletDataset, ECGPairDataset,
    TripletMarginCosineLoss, ContrastiveLoss,
    ImprovedECGLoader
)


def train_encoder(args, device):
    print('\n=== Training ECG Encoder ===')
    model     = ECGEncoder(embed_dim=args.embed_dim).to(device)
    dataset   = ECGTripletDataset(n_subjects=args.subjects)
    criterion = TripletMarginCosineLoss(margin=0.2)
    loader    = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=True, num_workers=2, pin_memory=True)
    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs)

    print(f'  Params   : {sum(p.numel() for p in model.parameters()):,}')
    print(f'  Dataset  : {len(dataset)} triplets')
    print(f'  Device   : {device}\n')

    losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimiser.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += loss.item()
        scheduler.step()
        avg = total / len(loader)
        losses.append(avg)
        print(f'Epoch {epoch:03d}/{args.epochs}  loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}')

    path = args.save_dir + '/ecg_encoder.pt'
    torch.save(model.state_dict(), path)
    print(f'\nEncoder saved → {path}')
    return model, losses


def train_fusion(args, device, encoder=None):
    print('\n=== Training Fusion Network ===')

    if encoder is None:
        encoder = ECGEncoder(embed_dim=args.embed_dim).to(device)
        encoder.load_state_dict(torch.load(args.save_dir + '/ecg_encoder.pt', map_location=device))
        print('  Loaded encoder from checkpoint.')

    # Freeze encoder
    for p in encoder.parameters(): p.requires_grad = False

    fusion    = BiometricFusionNet(ecg_embed_dim=args.embed_dim).to(device)
    dataset   = ECGPairDataset(n_subjects=args.subjects)
    loader    = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=True, num_workers=2, pin_memory=True)
    opt       = AdamW(fusion.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    nist_rng  = np.random.default_rng(99)

    print(f'  Params   : {sum(p.numel() for p in fusion.parameters()):,}')
    print(f'  Dataset  : {len(dataset)} pairs\n')

    losses = []
    for epoch in range(1, args.fusion_epochs + 1):
        fusion.train()
        total = 0.0
        for s1, s2, label in loader:
            s1, s2, label = s1.to(device), s2.to(device), label.to(device)
            with torch.no_grad():
                emb = encoder(s1)
            bs, lbl_np = label.shape[0], label.cpu().numpy()
            face_s = torch.tensor(
                nist_rng.normal(0.74, 0.08, bs) * lbl_np +
                nist_rng.normal(0.27, 0.10, bs) * (1 - lbl_np),
                dtype=torch.float32).to(device)
            fp_s = torch.tensor(
                nist_rng.normal(0.80, 0.06, bs) * lbl_np +
                nist_rng.normal(0.22, 0.09, bs) * (1 - lbl_np),
                dtype=torch.float32).to(device)
            opt.zero_grad()
            loss = criterion(fusion(emb, face_s, fp_s), label.float())
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        print(f'Epoch {epoch:03d}/{args.fusion_epochs}  loss={avg:.4f}')

    path = args.save_dir + '/fusion_model.pt'
    torch.save(fusion.state_dict(), path)
    print(f'\nFusion model saved → {path}')
    return fusion, losses


def main():
    parser = argparse.ArgumentParser(description='Train biometric fusion models')
    parser.add_argument('--mode',          type=str,   default='all',
                        choices=['all', 'encoder', 'fusion'])
    parser.add_argument('--epochs',        type=int,   default=20)
    parser.add_argument('--fusion-epochs', type=int,   default=15)
    parser.add_argument('--subjects',      type=int,   default=50)
    parser.add_argument('--batch-size',    type=int,   default=32)
    parser.add_argument('--embed-dim',     type=int,   default=128)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--save-dir',      type=str,   default='.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    import os
    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode in ('all', 'encoder'):
        encoder, _ = train_encoder(args, device)
    else:
        encoder = None

    if args.mode in ('all', 'fusion'):
        train_fusion(args, device, encoder)


if __name__ == '__main__':
    main()
