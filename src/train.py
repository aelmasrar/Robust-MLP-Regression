"""Entraînement contrôlé : initialisation commune, validation propre et patience."""
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .models import MLP
from .losses import loss_for


def fit(name, x_train, y_train, x_val, y_val, cfg, seed, device, initial_state):
    torch.manual_seed(seed)
    model = MLP(x_train.shape[1], cfg['hidden_dim'], cfg['hidden_layers']).to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    opt = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    vx, vy = torch.from_numpy(x_val).to(device), torch.from_numpy(y_val).to(device)
    history, best, best_state, best_epoch, stale = [], float('inf'), None, 0, 0
    for epoch in range(cfg['epochs']):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_for(name, model(xb), yb, cfg['huber_delta'], cfg['tukey_c'],
                            epoch, cfg['warmup_epochs'])
            loss.backward()
            if cfg.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val_rmse = torch.sqrt(torch.mean((model(vx) - vy)**2)).item()
        history.append({'epoch': epoch + 1, 'train_objective': total / len(ds), 'val_rmse': val_rmse})
        if val_rmse < best - cfg.get('min_delta', 1e-5):
            best, best_epoch, best_state, stale = val_rmse, epoch + 1, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        # Pas d'arrêt avant min_epochs, ni pendant le warmup.
        if epoch + 1 >= max(cfg['min_epochs'], cfg['warmup_epochs'] + 1) and stale >= cfg['patience']:
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch


def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x).to(device)).cpu().numpy()
