"""Plackett-Luce（上位3頭の着順）で線形スコアを学習する。

スコア s = X·w。1着は softmax(s)、2着は残りの馬で softmax、3着も同様。
レースごとに頭数が違うので、最大頭数に詰めて mask で無効にする。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def pack(df, cols, order_col="pos"):
    """df（1行1頭、rid と pos を持つ）→ X[R,N,K], mask[R,N], order[R,3]（各着の行位置、無ければ -1）"""
    groups = list(df.groupby("rid", sort=False))
    R, N, K = len(groups), max(len(g) for _, g in groups), len(cols)
    X = np.zeros((R, N, K)); M = np.zeros((R, N), bool); O = -np.ones((R, 3), int)
    rids, idx = [], -np.ones((R, N), int)
    for r, (rid, g) in enumerate(groups):
        n = len(g)
        X[r, :n] = g[cols].to_numpy(float); M[r, :n] = True; idx[r, :n] = g.index.to_numpy(); rids.append(rid)
        pos = g[order_col].to_numpy()
        for k in range(3):
            w = np.where(pos == k + 1)[0]
            if len(w):
                O[r, k] = w[0]
    return {"X": X, "M": M, "O": O, "rids": rids, "idx": idx}


def _loglik(w, P, l2, stages):
    X, M, O = P["X"], P["M"], P["O"]
    s = X @ w
    s = np.where(M, s, -1e9)
    alive = M.copy()
    ll = 0.0; grad = np.zeros_like(w)
    R = np.arange(len(X))
    for k in range(stages):
        o = O[:, k]; ok = o >= 0
        if not ok.any():
            break
        sk = np.where(alive, s, -1e9)
        mx = sk.max(1, keepdims=True)
        e = np.exp(sk - mx) * alive
        Z = e.sum(1, keepdims=True)
        p = e / Z
        ll += (s[R[ok], o[ok]] - (np.log(Z[ok, 0]) + mx[ok, 0])).sum()
        grad += (X[R[ok], o[ok]] - (p[ok][:, :, None] * X[ok]).sum(1)).sum(0)
        alive[R[ok], o[ok]] = False
    ll -= l2 * (w @ w); grad -= 2 * l2 * w
    return -ll, -grad


def fit(P, l2=1.0, stages=3):
    K = P["X"].shape[2]
    r = minimize(_loglik, np.zeros(K), args=(P, l2, stages), jac=True, method="L-BFGS-B")
    return r.x


def win_prob(w, P):
    s = np.where(P["M"], P["X"] @ w, -1e9)
    s -= s.max(1, keepdims=True)
    e = np.exp(s) * P["M"]
    return e / e.sum(1, keepdims=True)


def winner_logloss(p, P):
    o = P["O"][:, 0]; ok = o >= 0
    return float(-np.log(np.clip(p[np.arange(len(p))[ok], o[ok]], 1e-9, 1)).mean())
