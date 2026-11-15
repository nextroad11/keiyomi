"""3連複・3連単の確率。

- pl_tri：Plackett-Luce。2着・3着では強さを lam2・lam3 乗して弱める（Henery 型の割引）。
  lam=1 で普通の PL。lam<1 だと人気馬が2・3着に来る確率が下がり、穴が上がる。
- 3連複は3連単の6通りの合計。
- blend：実際のオッズから出した人気の確率と、AI の確率を、対数で重み付けして組み合わせる（レース内で合計1）。
"""
from __future__ import annotations

import itertools

import numpy as np

_P, _C, _MAP = {}, {}, {}


def perms(n):
    if n not in _P:
        P = np.array(list(itertools.permutations(range(n), 3)))
        C = np.array(list(itertools.combinations(range(n), 3)))
        k = lambda a: a[:, 0] * 10000 + a[:, 1] * 100 + a[:, 2]
        _P[n], _C[n], _MAP[n] = P, C, np.searchsorted(k(C), k(np.sort(P, 1)))
    return _P[n], _C[n], _MAP[n]


def pl_tri(s: np.ndarray, lam2: float = 1.0, lam3: float = 1.0) -> np.ndarray:
    n = len(s); P, _, _ = perms(n)
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    s = np.maximum(s, 1e-12)
    t2, t3 = s ** lam2, s ** lam3
    return s[a] * t2[b] / np.maximum(1e-12, t2.sum() - t2[a]) * t3[c] / np.maximum(1e-12, t3.sum() - t3[a] - t3[b])


def trio_from_tri(pt: np.ndarray, n: int) -> np.ndarray:
    _, C, M = perms(n)
    return np.bincount(M, weights=pt, minlength=len(C))


def market_prob(odds: np.ndarray):
    ok = odds > 0
    q = np.where(ok, 1 / np.where(ok, odds, 1), 0)
    return q / q.sum(), ok


def blend(logs: list[np.ndarray], w: np.ndarray, ok: np.ndarray) -> np.ndarray:
    s = sum(wi * li for wi, li in zip(w, logs))
    s = np.where(ok, s, -1e9)
    e = np.exp(s - s.max()) * ok
    return e / e.sum()
