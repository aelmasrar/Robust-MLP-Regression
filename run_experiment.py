import argparse
import copy
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data import make_data, contaminate, normalize
from src.models import MLP
from src.train import fit, predict
from src.plots import plot_results, plot_learning_curves, plot_predictions


def main():
    parser = argparse.ArgumentParser(description='Étude des pertes robustes pour MLP')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--quick', action='store_true', help='Test rapide du pipeline, pas une expérience scientifique')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if args.quick:
        cfg.update(n_seeds=1, n_train=250, n_val=100, n_test=120,
                   contamination_rates=[0.0, 0.2], hidden_layers=2,
                   hidden_dim=32, epochs=10, min_epochs=10, patience=10, warmup_epochs=3)
    if not cfg['models'] or 'mse' not in cfg['models']:
        raise ValueError('models doit contenir mse')
    if not all(0 <= p <= 1 for p in cfg['contamination_rates']):
        raise ValueError('Taux de contamination invalide')
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    for filename in ('metrics.csv', 'summary.csv', '01_robustesse.png',
                     '02_convergence.png', '03_predictions.png', 'config_used.yaml'):
        (out / filename).unlink(missing_ok=True)
    (out / 'config_used.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    if cfg['device'] == 'auto':
        device = ('cuda' if torch.cuda.is_available() else
                  'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
    else:
        device = cfg['device']
    print(f'Appareil : {device} | seeds : {cfg["n_seeds"]} | mode rapide : {args.quick}')
    rows = []
    demo_histories, demo_predictions = {}, {}
    demo_x = demo_y = None
    for s in range(cfg['n_seeds']):
        seed = cfg['seed'] + s
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        (xt, yt), (xv, yv), (xe, ye) = make_data(
            seed, cfg['n_train'], cfg['n_val'], cfg['n_test'], cfg['noise_std'])
        xt, xv, xe = [v.astype(np.float32) for v in normalize(xt, xv, xe)]
        # Target scaling uses clean labels available only in this synthetic experiment.
        y_mean, y_std = float(yt.mean()), max(float(yt.std()), 1e-8)
        yv_scaled = ((yv - y_mean) / y_std).astype(np.float32)
        torch.manual_seed(seed)
        initial = copy.deepcopy(MLP(1, cfg['hidden_dim'], cfg['hidden_layers']).state_dict())
        for p in cfg['contamination_rates']:
            yc, mask = contaminate(yt, p, cfg['outlier_magnitude'],
                                   seed + int(round(p * 10000)) + 10000)
            yc_scaled = ((yc - y_mean) / y_std).astype(np.float32)
            for name in cfg['models']:
                model, history, best_epoch = fit(name, xt, yc_scaled, xv, yv_scaled,
                                                 cfg, seed, device, initial)
                pred = predict(model, xe, device) * y_std + y_mean
                collapsed = bool(float(np.std(pred)) < 0.10 * float(np.std(ye)))
                row = dict(seed=seed, contamination=float(p), model=name,
                           rmse=float(np.sqrt(mean_squared_error(ye, pred))),
                           mae=float(mean_absolute_error(ye, pred)),
                           r2=float(r2_score(ye, pred)), best_epoch=best_epoch,
                           n_outliers=int(mask.sum()), pred_std=float(np.std(pred)),
                           collapsed=collapsed)
                rows.append(row)
                print(f'seed={seed} contamination={p:.0%} model={name:13s} '
                      f'RMSE={row["rmse"]:.4f} epoch={best_epoch}'
                      f'{" [quasi-constant]" if collapsed else ""}', flush=True)
                if s == 0 and p == cfg['contamination_rates'][-1]:
                    demo_histories[name] = history
                    demo_predictions[name] = pred
                    demo_x, demo_y = xe, ye
    df = pd.DataFrame(rows)
    df.to_csv(out / 'metrics.csv', index=False)
    plot_results(df, out)
    plot_learning_curves(demo_histories, out, cfg['seed'], cfg['contamination_rates'][-1])
    plot_predictions(demo_predictions, demo_x, demo_y, out, cfg['seed'], cfg['contamination_rates'][-1])
    print(f'Résultats : {out.resolve()} (2 CSV, 3 figures, 1 configuration)')


if __name__ == '__main__':
    main()
