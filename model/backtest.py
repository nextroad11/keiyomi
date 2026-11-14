"""買い方の検証（外れも含め、払戻は実際の結果から）。

- 確率：モデルのスコアから Plackett-Luce で全組の確率。
- 推定オッズ：アプリと同じく、単勝オッズから作った人気モデルの確率と控除率から (1−控除率)/人気確率。
  単勝だけは確定の単勝オッズをそのまま使う。
- 注意：確定オッズは買う時点では分からない。ここでの単勝の数字は「締切直前のオッズで買えた」と仮定した上限寄りの値。
"""
from __future__ import annotations

import itertools

import numpy as np

TAKE = {"単勝": 0.20, "複勝": 0.20, "馬連": 0.225, "ワイド": 0.225, "馬単": 0.25, "3連複": 0.25, "3連単": 0.275}


def pl_tables(s: np.ndarray):
    """s: 強さ（合計1）。戻り値: 3連単 dict (a,b,c)->p, 単勝 win, 3着内 top3, 馬連・ワイド・馬単・3連複"""
    n = len(s); tri = {}; top3 = np.zeros(n); ex = {}
    for a in range(n):
        for b in range(n):
            if b == a:
                continue
            pab = s[a] * s[b] / max(1e-12, 1 - s[a]); ex[(a, b)] = pab
            rest = max(1e-12, 1 - s[a] - s[b])
            for c in range(n):
                if c == a or c == b:
                    continue
                p = pab * s[c] / rest; tri[(a, b, c)] = p
                top3[a] += p; top3[b] += p; top3[c] += p
    quin = {}; wide = {}; trio = {}
    for (a, b), p in ex.items():
        k = tuple(sorted((a, b))); quin[k] = quin.get(k, 0) + p
    for (a, b, c), p in tri.items():
        trio_k = tuple(sorted((a, b, c))); trio[trio_k] = trio.get(trio_k, 0) + p
        for k in ((a, b), (a, c), (b, c)):
            k = tuple(sorted(k)); wide[k] = wide.get(k, 0) + p
    return {"単勝": {(i,): s[i] for i in range(n)}, "複勝": {(i,): top3[i] for i in range(n)},
            "馬連": quin, "ワイド": wide, "馬単": ex, "3連複": trio, "3連単": tri}


def key_str(typ, k, nos):
    v = [str(nos[i]) for i in k]
    return "-".join(v)  # JRA の払戻表記に合わせる（馬単・3連単は着順どおり、他は昇順）


def payout(race, typ, k, nos):
    """100円あたりの払戻（外れは0）"""
    rows = race["pay"].get(typ, [])
    ks = key_str(typ, k if typ in ("馬単", "3連単") else tuple(sorted(k, key=lambda i: nos[i])), nos)
    for r in rows:
        if r["combo"].replace(" ", "") == ks:
            return r["yen"]
    return 0
