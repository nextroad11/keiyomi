"""3連複・3連単オッズの詰め方。

馬番を昇順に並べ、3連複は組み合わせ順、3連単は順列順に、オッズ×10 を uint32 で並べて base64。
票数なし・取消は 0。18頭なら3連単4,896組で約20KB。
"""
from __future__ import annotations

import base64
import itertools
from array import array


def pack(nos: list[int], odds: dict, ordered: bool) -> str:
    nos = sorted(nos)
    it = itertools.permutations(nos, 3) if ordered else itertools.combinations(nos, 3)
    a = array("I", (int(round(odds.get(k, 0) * 10)) for k in it))
    return base64.b64encode(a.tobytes()).decode()


def unpack(nos: list[int], s: str, ordered: bool) -> dict:
    nos = sorted(nos)
    a = array("I"); a.frombytes(base64.b64decode(s))
    it = itertools.permutations(nos, 3) if ordered else itertools.combinations(nos, 3)
    return {k: v / 10 for k, v in zip(it, a) if v}


# ---------- 枠連・馬連・ワイド・馬単 ----------
def pair_keys(nos: list[int], kind: str):
    nos = sorted(nos)
    if kind == "umatan":
        return list(itertools.permutations(nos, 2))
    if kind == "waku":
        return [(a, b) for a in range(1, 9) for b in range(a, 9)]
    return list(itertools.combinations(nos, 2))


def pack_pairs(nos: list[int], odds: dict, kind: str) -> str:
    """馬連・馬単・枠連はオッズ×10、ワイドは下限×10と上限×10を交互に。無い組は0。"""
    vals = []
    for k in pair_keys(nos, kind):
        v = odds.get(k)
        if kind == "wide":
            vals += [int(round(v[0] * 10)), int(round(v[1] * 10))] if v else [0, 0]
        else:
            vals.append(int(round(v * 10)) if v else 0)
    return base64.b64encode(array("I", vals).tobytes()).decode()


def unpack_pairs(nos: list[int], s: str, kind: str) -> dict:
    a = array("I"); a.frombytes(base64.b64decode(s))
    keys = pair_keys(nos, kind)
    if kind == "wide":
        return {k: (a[2 * i] / 10, a[2 * i + 1] / 10) for i, k in enumerate(keys) if a[2 * i]}
    return {k: v / 10 for k, v in zip(keys, a) if v}
