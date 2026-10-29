"""Trois figures seulement : robustesse, convergence, prédictions."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_results(df, output):
    summary = df.groupby(['model', 'contamination'], as_index=False).agg(
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        mae_mean=('mae', 'mean'), r2_mean=('r2', 'mean'),
        collapse_rate=('collapsed', 'mean'))
    summary.to_csv(Path(output) / 'summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, g in summary.groupby('model', sort=False):
        g = g.sort_values('contamination')
        xx = g.contamination.to_numpy() * 100
        yy = g.rmse_mean.to_numpy()
        ax.plot(xx, yy, marker='o', label=name)
        if df.seed.nunique() > 1:
            std = g.rmse_std.fillna(0).to_numpy()
            ax.fill_between(xx, yy - std, yy + std, alpha=0.10)
    ax.set(xlabel='Contamination du train (%)', ylabel='RMSE sur test propre',
           title='Robustesse des MLP — moyenne ± écart-type')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / '01_robustesse.png', dpi=160)
    plt.close(fig)


def plot_learning_curves(histories, output, seed, contamination):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, rows in histories.items():
        ax.plot([r['epoch'] for r in rows], [r['val_rmse'] for r in rows], label=name)
    ax.set(xlabel='Époque', ylabel='RMSE de validation (standardisée)',
           title=f'Convergence — seed {seed}, contamination {contamination:.0%}')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / '02_convergence.png', dpi=160)
    plt.close(fig)


def plot_predictions(predictions, x, y, output, seed, contamination):
    order = np.argsort(x[:, 0])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x[:, 0], y[:, 0], s=8, alpha=0.18, label='Test propre (bruité)')
    for name, pred in predictions.items():
        ax.plot(x[order, 0], pred[order, 0], label=name, linewidth=1.5)
    ax.set(xlabel='x standardisé', ylabel='y',
           title=f'Prédictions — seed {seed}, contamination {contamination:.0%}')
    ax.grid(alpha=0.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / '03_predictions.png', dpi=160)
    plt.close(fig)
