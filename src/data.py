"""Jeu de données synthétique et contamination uniquement du train."""
import numpy as np

def signal(x):
    return np.sin(2 * np.pi * x) + 0.5 * x**2

def make_data(seed, n_train, n_val, n_test, noise_std):
    rng = np.random.default_rng(seed)
    parts = []
    for n in (n_train, n_val, n_test):
        x = rng.uniform(-2, 2, size=(n, 1)).astype(np.float32)
        y = (signal(x) + rng.normal(0, noise_std, size=(n, 1))).astype(np.float32)
        parts.append((x, y))
    return parts

def contaminate(y, rate, magnitude, seed):
    if not 0 <= rate <= 1:
        raise ValueError('rate doit appartenir à [0, 1]')
    rng = np.random.default_rng(seed)
    result = y.copy()
    count = int(round(len(y) * rate))
    mask = np.zeros(len(y), dtype=bool)
    if count:
        ids = rng.choice(len(y), count, replace=False)
        mask[ids] = True
        result[ids, 0] += rng.choice([-1.0, 1.0], size=count) * magnitude * rng.uniform(0.7, 1.3, size=count)
    return result.astype(np.float32), mask

def normalize(x_train, x_val, x_test):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1, std)
    return [(x - mean) / std for x in (x_train, x_val, x_test)]
