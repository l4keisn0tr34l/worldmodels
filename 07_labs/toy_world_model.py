"""A complete tiny learned, action-conditioned world model (Python stdlib only).

The environment has true state (position, velocity), and known action a. This lab
provides full state to the learner deliberately; it is NOT a POMDP/RSSM demo.
Run: python3 07_labs/toy_world_model.py
"""
from dataclasses import dataclass
import math
import random

@dataclass(frozen=True)
class State:
    x: float
    v: float


def true_step(s: State, action: float, rng: random.Random, noise: float = 0.015) -> State:
    """Semi-implicit integration with damping and acceleration."""
    v_next = 0.92 * s.v + 0.35 * max(-1.0, min(1.0, action)) + rng.gauss(0.0, noise)
    return State(s.x + v_next, v_next)


def features(s: State, action: float):
    return [s.x, s.v, action, 1.0]


def solve(a, b):
    """Small dense Gaussian elimination with partial pivoting; no packages."""
    n = len(b)
    aug = [list(a[i]) + [b[i]] for i in range(n)]
    for j in range(n):
        pivot = max(range(j, n), key=lambda i: abs(aug[i][j]))
        aug[j], aug[pivot] = aug[pivot], aug[j]
        if abs(aug[j][j]) < 1e-12:
            raise ValueError('Singular normal matrix')
        inv = 1.0 / aug[j][j]
        aug[j] = [z * inv for z in aug[j]]
        for i in range(n):
            if i != j:
                k = aug[i][j]
                aug[i] = [u - k * v for u, v in zip(aug[i], aug[j])]
    return [aug[i][-1] for i in range(n)]


class LinearWorldModel:
    def __init__(self):
        self.coef_x = [0.0] * 4
        self.coef_v = [0.0] * 4

    def fit(self, transitions, ridge=1e-5):
        gram = [[0.0 for _ in range(4)] for _ in range(4)]
        tx = [0.0] * 4
        tv = [0.0] * 4
        for s, a, sn in transitions:
            f = features(s, a)
            for i in range(4):
                tx[i] += f[i] * sn.x
                tv[i] += f[i] * sn.v
                for j in range(4):
                    gram[i][j] += f[i] * f[j]
        for i in range(4):
            gram[i][i] += ridge
        self.coef_x = solve(gram, tx)
        self.coef_v = solve(gram, tv)
        return self

    def predict(self, s: State, a: float) -> State:
        f = features(s, a)
        return State(sum(b*x for b, x in zip(self.coef_x, f)),
                     sum(b*x for b, x in zip(self.coef_v, f)))


def collect_data(seed=10, episodes=180, length=30):
    rng = random.Random(seed)
    data = []
    for _ in range(episodes):
        s = State(rng.uniform(-2.0, 2.0), rng.uniform(-0.5, 0.5))
        for _ in range(length):
            a = rng.uniform(-1, 1)
            sn = true_step(s, a, rng)
            data.append((s, a, sn))
            s = sn
    return data


def rollout_errors(model, horizon=30, seed=61, episodes=80):
    rng = random.Random(seed)
    errs = {1: [], 5: [], 15: [], 30: []}
    for _ in range(episodes):
        s_true = State(rng.uniform(-2, 2), rng.uniform(-0.5, 0.5))
        s_pred = s_true
        for k in range(1, horizon+1):
            a = rng.uniform(-1, 1)
            s_true = true_step(s_true, a, rng)
            s_pred = model.predict(s_pred, a)
            if k in errs:
                errs[k].append(abs(s_true.x-s_pred.x) + abs(s_true.v-s_pred.v))
    return {k: sum(v)/len(v) for k,v in errs.items()}


def main():
    model = LinearWorldModel().fit(collect_data())
    print('True approximate x_next coefficients: [1, 0.92, 0.35, 0]')
    print('Learned x_next coefficients:         ', [round(x,4) for x in model.coef_x])
    print('True approximate v_next coefficients: [0, 0.92, 0.35, 0]')
    print('Learned v_next coefficients:         ', [round(x,4) for x in model.coef_v])
    print('Open-loop mean absolute combined error (x,v) by rollout step:')
    for k, err in rollout_errors(model).items():
        print(f'  horizon={k:2d}: {err:.4f}')
    print('\nTry changing the noise level, training action range, or horizon.')
    print('If test actions go outside training range, coefficients alone do not guarantee validity.')


if __name__ == '__main__':
    main()
