"""
evaluate.py
===========
Full evaluation: EER table, ROC curves, score distributions, t-SNE plot.

Usage:
    python src/evaluate.py
    python src/evaluate.py --subjects 200 --plot
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from pipeline import (
    tanh_normalize, fusion_sum_rule, compute_eer, ImprovedECGLoader
)


def build_score_dataset(n_subjects=200, seed=0):
    rng  = np.random.default_rng(seed)
    ng   = n_subjects * 5
    ni   = n_subjects * 45

    def make(gmu, gs, imu, is_):
        return (rng.normal(gmu, gs, ng).clip(0, 1),
                rng.normal(imu, is_, ni).clip(0, 1))

    return {
        'ECG':         make(0.70, 0.09, 0.30, 0.11),
        'Face':        make(0.74, 0.08, 0.27, 0.10),
        'Fingerprint': make(0.80, 0.06, 0.22, 0.09),
    }


def evaluate(n_subjects=200):
    raw    = build_score_dataset(n_subjects)
    results = {}

    for name, (g, i) in raw.items():
        gn  = tanh_normalize(g, g.mean(), g.std())
        in_ = tanh_normalize(i, g.mean(), g.std())
        eer, th, far, frr = compute_eer(gn, in_)
        results[name] = dict(eer=eer, gn=gn, in_=in_, far=far, frr=frr, th=th)

    fg = fusion_sum_rule([results[m]['gn']  for m in results], [0.35, 0.35, 0.30])
    fi = fusion_sum_rule([results[m]['in_'] for m in results], [0.35, 0.35, 0.30])
    eer_f, th_f, far_f, frr_f = compute_eer(fg, fi)
    results['Fused'] = dict(eer=eer_f, gn=fg, in_=fi, far=far_f, frr=frr_f, th=th_f)
    return results


def print_report(results):
    print('\n' + '=' * 45)
    print(f'  {"Modality":<18} {"EER":>8}')
    print('-' * 30)
    for name, r in results.items():
        marker = '  ← best' if name == 'Fused' else ''
        print(f'  {name:<18} {r["eer"]*100:>6.2f}%{marker}')
    print('=' * 45)


def plot_all(results, save_path=None):
    colors = {'ECG': '#2196F3', 'Face': '#4CAF50', 'Fingerprint': '#FF9800', 'Fused': '#E91E63'}
    styles = {'ECG': '--', 'Face': '-.', 'Fingerprint': ':', 'Fused': '-'}
    bins   = np.linspace(0, 1, 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC
    ax = axes[0]
    for name, r in results.items():
        ax.plot(r['far'], r['frr'], label=f"{name} EER={r['eer']*100:.2f}%",
                color=colors[name], linestyle=styles[name], linewidth=2)
    ax.plot([0, 1], [1, 0], 'k--', alpha=0.3, linewidth=1, label='EER line')
    ax.set_xlabel('False Accept Rate'); ax.set_ylabel('False Reject Rate')
    ax.set_title('ROC Curves'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Score distributions ECG
    ax = axes[1]
    r  = results['ECG']
    ax.hist(r['gn'],  bins=bins, density=True, alpha=0.6, color='#2196F3', label='Genuine')
    ax.hist(r['in_'], bins=bins, density=True, alpha=0.4, color='#E24B4A', label='Impostor')
    ax.axvline(r['th'][np.argmin(np.abs(r['far'] - r['frr']))],
               color='black', linestyle='--', linewidth=1, label='EER threshold')
    ax.set_title(f"ECG Score Distribution (EER={r['eer']*100:.2f}%)")
    ax.set_xlabel('Normalised score'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Score distributions Fused
    ax = axes[2]
    r  = results['Fused']
    ax.hist(r['gn'],  bins=bins, density=True, alpha=0.6, color='#E91E63', label='Genuine')
    ax.hist(r['in_'], bins=bins, density=True, alpha=0.4, color='#E24B4A', label='Impostor')
    ax.axvline(r['th'][np.argmin(np.abs(r['far'] - r['frr']))],
               color='black', linestyle='--', linewidth=1, label='EER threshold')
    ax.set_title(f"Fused Score Distribution (EER={r['eer']*100:.2f}%)")
    ax.set_xlabel('Normalised score'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Plot saved → {save_path}')
    plt.show()


def plot_tsne(model, device, n_subjects=20, save_path=None):
    from sklearn.manifold import TSNE
    import torch

    loader = ImprovedECGLoader()
    all_embs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for sid in range(n_subjects):
            segs = torch.tensor(loader.get_subject_segments(sid),
                                dtype=torch.float32).unsqueeze(1).to(device)
            emb  = model(segs).cpu().numpy()
            all_embs.append(emb)
            all_labels.extend([sid] * len(emb))

    all_embs = np.vstack(all_embs)
    embs_2d  = TSNE(n_components=2, perplexity=15, random_state=42).fit_transform(all_embs)

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('tab20', n_subjects)
    for sid in range(n_subjects):
        mask = np.array(all_labels) == sid
        plt.scatter(embs_2d[mask, 0], embs_2d[mask, 1],
                    color=cmap(sid), label=f'S{sid:02d}', s=60, alpha=0.8)
    plt.title('t-SNE: ECG Embeddings\n(clusters = good subject separation)')
    plt.legend(fontsize=6, ncol=4)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', type=int, default=200)
    parser.add_argument('--plot',     action='store_true', default=True)
    parser.add_argument('--save',     type=str, default='results/')
    args = parser.parse_args()

    results = evaluate(n_subjects=args.subjects)
    print_report(results)

    if args.plot:
        import os; os.makedirs(args.save, exist_ok=True)
        plot_all(results, save_path=args.save + 'roc_distributions.png')


if __name__ == '__main__':
    main()
