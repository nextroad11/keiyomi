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
