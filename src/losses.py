import torch
import torch.nn.functional as F


def tukey_loss(pred, target, c=4.685):
    if c <= 0:
        raise ValueError('c doit être > 0')
    r = pred - target
    z2 = (r / c).square()
    # Residuals beyond c have zero gradient.
    return ((c * c / 6) * (1 - (1 - z2).clamp(min=0).pow(3))).mean()


def loss_for(name, pred, target, huber_delta, tukey_c, epoch=0, warmup_epochs=0):
    if name == 'mse' or (name in ('huber_warmup', 'tukey_warmup') and epoch < warmup_epochs):
        return F.mse_loss(pred, target)
    if name in ('huber', 'huber_warmup'):
        return F.huber_loss(pred, target, delta=huber_delta)
    if name in ('tukey', 'tukey_warmup'):
        return tukey_loss(pred, target, c=tukey_c)
    raise ValueError(f'Modèle inconnu : {name}')
