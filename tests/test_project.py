import numpy as np
import torch
from src.data import make_data, contaminate
from src.losses import tukey_loss, loss_for
from src.models import MLP
from src.train import fit, predict


def test_contamination():
    y = np.zeros((100, 1), dtype=np.float32)
    yc, mask = contaminate(y, 0.2, 5, 42)
    assert mask.sum() == 20 and np.count_nonzero(yc) == 20 and np.count_nonzero(y) == 0


def test_tukey_gradient_saturation():
    pred = torch.tensor([[100.0]], requires_grad=True)
    loss = tukey_loss(pred, torch.zeros_like(pred))
    loss.backward()
    assert torch.isclose(loss, torch.tensor(4.685**2 / 6), atol=1e-5)
    assert pred.grad.item() == 0.0


def test_tukey_zero():
    x = torch.zeros((3, 1))
    assert tukey_loss(x, x).item() == 0


def test_model_shape():
    assert MLP(3, 8, 2)(torch.zeros(5, 3)).shape == (5, 1)


def test_data_shapes():
    assert [x.shape for x, _ in make_data(42, 20, 10, 5, 0.1)] == [(20, 1), (10, 1), (5, 1)]


def test_all_losses_and_warmup():
    x, y = torch.ones((2, 1)), torch.zeros((2, 1))
    for name in ('mse', 'huber', 'huber_warmup', 'tukey', 'tukey_warmup'):
        assert torch.isfinite(loss_for(name, x, y, 1, 4.685, 0, 2))
    assert torch.isclose(loss_for('huber_warmup', x, y, 1, 4.685, 0, 2), torch.tensor(1.0))
    assert torch.isclose(loss_for('huber_warmup', x, y, 1, 4.685, 2, 2), torch.tensor(0.5))


def test_training_smoke():
    (x, y), (v, z), _ = make_data(7, 40, 20, 10, 0.1)
    cfg = dict(hidden_dim=8, hidden_layers=2, learning_rate=0.001,
               weight_decay=0, batch_size=16, epochs=4, min_epochs=4,
               patience=3, min_delta=1e-5, grad_clip=5,
               huber_delta=1, tukey_c=4.685, warmup_epochs=2)
    initial = MLP(1, 8, 2).state_dict()
    model, history, epoch = fit('huber_warmup', x, y, v, z, cfg, 7, 'cpu', initial)
    assert len(history) == 4 and 1 <= epoch <= 4
    assert predict(model, v, 'cpu').shape == z.shape
