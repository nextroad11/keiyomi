"""本番の予想で使う共通部品（学習 train.py と開催日の予想 predict.py の両方から使う）。

確率の作り方（検証レポート第4回と同じ）
1. 勝ち馬モデル：単勝人気の確率（対数）＋指数19項目（レース内で標準化）を Plackett-Luce で学習 → 各馬の強さ
2. 3連複・3連単：実際のオッズから出した人気の確率、AI の強さからの PL、単勝人気だけの PL を対数で重み付けして組み合わせる
買い方（記録用。まだ買わない）
- 3連複：期待値≧1.0・確率≧1%・オッズ≧10倍の組を全部（上限なし）
- 3連単（検証中）：3連複で選んだ3頭の並びのうち、3連単でも期待値≧1.0の組を期待値の高い順に48点まで
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import exotic as X  # noqa: E402
import features as F  # noqa: E402

Z = ["z_" + c for c in F.FEATURES]
RULES = {
    "trio": {"th": 1.0, "pmin": 0.01, "omin": 10},
    # 第8回：3連複で選んだ3頭の並びから、当たる確率の高い順に10点まで（確率≧0.5%・期待値≧1）
    "tri": {"th": 1.0, "cap": 10, "pmin": 0.005, "from_trio": True},
}
MARKS = ["◎", "○", "▲", "△", "△"]


def load_results(store, dates) -> list[dict]:
    out = []
    for d in dates:
        out += list(store.read(f"results/{d}.json", {"races": {}})["races"].values())
    return out


def hist_frame(races: list[dict], fig_end: str) -> pd.DataFrame:
    """過去の結果から、指数を付けた1行1頭の表（履歴づくり用）"""
    df = F.runners_frame(races)
    return df


def today_frame(live: dict) -> pd.DataFrame:
    """出馬表から、今日の出走馬の行（結果の列は空）"""
    rows = []
    for rid, r in live["races"].items():
        c = r.get("card")
        if not c:
            continue
        surf = F.surface_of(c)
        for x in c["runners"]:
            if x.get("no") is None:  # 取消・除外で馬番が無い
                continue
            sa = x.get("sexage") or ""
            rows.append({
                "rid": rid, "date": live["date"], "jcd": r["jcd"], "surface": surf, "distance": c.get("distance"),
                "going": c.get("going"), "cls": F.class_level(c.get("cls"), c.get("name")), "n": len(c["runners"]),
                "no": x["no"], "waku": x.get("waku"), "hid": x.get("hid"), "jockey": (x.get("jockey") or "").lstrip("▲△☆★◇ "),
                "pos": np.nan, "time": np.nan, "f3": np.nan, "corner1": np.nan, "body": np.nan, "body_diff": np.nan,
                "weight": x.get("weight"), "sex": sa[:1] or None, "age": F._int(sa[1:]), "pop": x.get("pop"), "odds": x.get("odds"),
                "pmin": np.nan,
            })
    df = pd.DataFrame(rows)
    if len(df):
        df["dt"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def build_features(hist: pd.DataFrame, today: pd.DataFrame | None, fig_end: str) -> pd.DataFrame:
    df = pd.concat([hist, today], ignore_index=True) if today is not None and len(today) else hist
    df = F.add_figures(df, fig_end)
    df = F.add_history(df)
    return df


def zscore_race(g: pd.DataFrame) -> pd.DataFrame:
    """1レース分（取消を除いた行）の市場確率と標準化"""
    g = g.copy()
    inv = 1 / g["odds"]
    g["p_mkt"] = inv / inv.sum(); g["logp_mkt"] = np.log(g["p_mkt"])
    g["runs_log"] = np.log1p(g["runs"]); g["female"] = (g["sex"] == "牝").astype(float)
    for c in F.FEATURES:
        mu, sd = g[c].mean(), g[c].std()
        g["z_" + c] = ((g[c] - mu) / (sd if sd and sd == sd else np.nan)).fillna(0).clip(-4, 4)
    return g


def strengths(g: pd.DataFrame, W3, WM):
    sc = g[["logp_mkt"] + Z].to_numpy(float) @ np.asarray(W3)
    e = np.exp(sc - sc.max()); s_ai = e / e.sum()
    sm = g["logp_mkt"].to_numpy(float) * WM[0]
    e = np.exp(sm - sm.max()); s_mk = e / e.sum()
    return s_ai, s_mk


def exotic_probs(s_ai, s_mk, odds_tri, odds_trio, BW):
    """3連単・3連複の各組の確率（実オッズ＋AI＋単勝人気PLの組み合わせ）。s は馬番昇順の並び"""
    n = len(s_ai)
    pt, pmt = X.pl_tri(s_ai), X.pl_tri(s_mk)
    pf, pmf = X.trio_from_tri(pt, n), X.trio_from_tri(pmt, n)
    out = {}
    for k, o, a, m in (("t", odds_tri, pt, pmt), ("f", odds_trio, pf, pmf)):
        q, ok = X.market_prob(o)
        if ok.sum() == 0:
            out[k] = (None, ok)
            continue
        out[k] = (X.blend([np.log(q + 1e-9), np.log(a + 1e-12), np.log(m + 1e-12)], BW[k], ok), ok)
    return out, pt


def picks(n, probs, odds_tri, odds_trio):
    P, C, M = X.perms(n)
    res = {"trio": [], "tri": []}
    pf, okf = probs["f"]
    if pf is not None:
        r = RULES["trio"]; ev = pf * odds_trio
        idx = np.where(okf & (ev >= r["th"]) & (pf >= r["pmin"]) & (odds_trio >= r["omin"]))[0]
        res["trio"] = [int(i) for i in idx[np.argsort(-ev[idx])]]
    pt, okt = probs["t"]
    if pt is not None and res["trio"]:
        r = RULES["tri"]; ev = pt * odds_tri
        want = np.isin(M, res["trio"])
        idx = np.where(okt & want & (ev >= r["th"]) & (pt >= r.get("pmin", 0)))[0]
        res["tri"] = [int(i) for i in idx[np.argsort(-pt[idx])][: r["cap"]]]
    return res


# ---------- ほかの券種（単勝・複勝・枠連・馬連・ワイド・馬単）：検証中。記録だけ付ける ----------
# 重みと条件は model/experiments/17・18（学習〜2025-05、条件は基準決め期間）で決めたもの。REPORT.md 第5回。
PAIR_W = {"単勝": [0.7218, 0.5118, -0.1386], "複勝": [-0.2985, 1.5858, 0.0328, 0.5712], "枠連": [0.2734, 0.7338, 0.0821],
          "馬連": [0.8137, 0.4801, -0.2247], "ワイド": [0.6714, 0.8215, -0.3585, 1.2368], "馬単": [0.862, 0.5233, -0.3238]}
PAIR_RULES = {"単勝": {"th": 1.0, "pmin": 0, "omin": 0}, "複勝": {"th": 1.0, "pmin": 0.2, "omin": 2},
              "枠連": {"th": 1.1, "pmin": 0.05, "omin": 10}, "馬連": {"th": 1.0, "pmin": 0, "omin": 0},
              "ワイド": {"th": 1.0, "pmin": 0.05, "omin": 5}, "馬単": {"th": 1.0, "pmin": 0, "omin": 0}}
PAIR_KIND = {"枠連": "waku", "馬連": "umaren", "ワイド": "wide", "馬単": "umatan"}


def pair_tables(pt, n, waku):
    P, C, _ = X.perms(n)
    win = np.bincount(P[:, 0], pt, n)
    top3 = win + np.bincount(P[:, 1], pt, n) + np.bincount(P[:, 2], pt, n)
    ex = np.zeros((n, n)); np.add.at(ex, (P[:, 0], P[:, 1]), pt)
    wd = np.zeros((n, n))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = np.minimum(P[:, i], P[:, j]), np.maximum(P[:, i], P[:, j]); np.add.at(wd, (a, b), pt)
    out = {"単勝": {(i,): win[i] for i in range(n)}, "複勝": {(i,): top3[i] for i in range(n)},
           "馬単": {(i, j): ex[i, j] for i in range(n) for j in range(n) if i != j},
           "馬連": {(i, j): ex[i, j] + ex[j, i] for i in range(n) for j in range(i + 1, n)},
           "ワイド": {(i, j): wd[i, j] for i in range(n) for j in range(i + 1, n)}}
    wk = {}
    if waku and None not in waku:
        for (i, j), p in out["馬連"].items():
            k = tuple(sorted((waku[i], waku[j]))); wk[k] = wk.get(k, 0) + p
    out["枠連"] = wk
    return out


def pair_picks(sa, sm, nos, waku, win_odds: dict, place_odds: dict, pairs: dict) -> dict:
    """券種ごとの候補 [{c, p, o, ev}]。オッズが取得不可の券種は入れない。"""
    n = len(nos)
    A, M = pair_tables(X.pl_tri(sa), n, waku), pair_tables(X.pl_tri(sm), n, waku)
    res = {}
    for t, w in PAIR_W.items():
        if t == "単勝":
            od = {(i,): win_odds.get(no) for i, no in enumerate(nos)}
        elif t == "複勝":
            od = {(i,): place_odds.get(no) for i, no in enumerate(nos)}
        elif t == "枠連":
            od = {k: (pairs.get("waku") or {}).get(k) for k in A["枠連"]}
        else:
            src = pairs.get(PAIR_KIND[t]) or {}
            od = {k: src.get((nos[k[0]], nos[k[1]])) for k in A[t]}
            if t == "ワイド":
                od = {k: (v[0] if v else None) for k, v in od.items()}
        keys = [k for k in A[t] if od.get(k)]
        if len(keys) < 2:
            continue
        o = np.array([od[k] for k in keys], float); q = 1 / o
        Xm = np.stack([np.log(q / q.sum()), np.log(np.array([A[t][k] for k in keys]) + 1e-12), np.log(np.array([M[t][k] for k in keys]) + 1e-12)], 1)
        s = Xm @ np.array(w[:3])
        p = 1 / (1 + np.exp(-(s + w[3]))) if t in ("複勝", "ワイド") else np.exp(s - s.max()) / np.exp(s - s.max()).sum()
        ev = p * o; r = PAIR_RULES[t]
        idx = [i for i in np.argsort(-ev) if ev[i] >= r["th"] and p[i] >= r["pmin"] and o[i] >= r["omin"]]
        lab = (lambda k: str(nos[k[0]])) if t in ("単勝", "複勝") else (lambda k: f"{k[0]}-{k[1]}") if t == "枠連" else \
              (lambda k: f"{nos[k[0]]}→{nos[k[1]]}") if t == "馬単" else (lambda k: f"{nos[k[0]]}-{nos[k[1]]}")
        res[t] = [{"c": lab(keys[i]), "p": round(float(p[i]), 4), "o": float(o[i]), "ev": round(float(ev[i]), 3)} for i in idx]
    return res


# ---------- 金額の決め方（本番の賭け方。REPORT.md 第6回） ----------
# 1点の金額＝ケリー基準×0.25（基準の元手100万円）、100円単位・最低100円。1レース30,000円まで。
# 1日50,000円まで。その日すでに決めた買い目の見込み利益が6,000円を超えたら「勝負日」として1日100,000円まで。
KELLY, BANK, RACE_CAP, DAY_CAP, BIG_CAP, BIG_EV = 0.25, 1_000_000, 30000, 50000, 100000, 6000
# 金額の倍率（config/bet.json の scale。1/10 なら 0.1）。上限も一緒に縮める。1点の最低は100円のまま
try:
    import json as _json
    _cfg = _json.loads((Path(__file__).resolve().parent.parent / "config" / "bet.json").read_text())
    SCALE = float(_cfg["scale"]); BET_TYPES = list(_cfg.get("types", ["trio", "tri"]))
except Exception:  # noqa: BLE001
    SCALE, BET_TYPES = 1.0, ["trio", "tri"]
if SCALE != 1.0:
    _r = lambda v: max(100, int(round(v * SCALE / 100)) * 100)
    BANK, RACE_CAP, DAY_CAP, BIG_CAP, BIG_EV = BANK * SCALE, _r(RACE_CAP), _r(DAY_CAP), _r(BIG_CAP), BIG_EV * SCALE
# 記録だけのパターン（お金は賭けない）。条件は experiments/29_patterns.py で基準決め期間だけから決めた
PATTERNS = {
    "honsen": {"kind": "trio", "pmin": 0.06, "th": 1.0, "cap": 3, "unit": 1000},          # 本線：固い組に厚く
    "ana": {"kind": "tri", "win_odds": 20, "th": 1.0, "pmin": 0.001, "cap": 20, "unit": 100},  # 大穴：人気薄を1着に広く
}


def pattern_picks(n, probs, o_t, o_f, win_odds_first):
    """{"honsen": [3連複の組の番号], "ana": [3連単の並びの番号]}（確率の高い順）"""
    out = {}
    pf, okf = probs["f"]; pt, okt = probs["t"]
    a = PATTERNS["honsen"]
    if pf is not None:
        c = np.where(okf & (pf >= a["pmin"]) & (pf * o_f >= a["th"]))[0]
        out["honsen"] = [int(i) for i in c[np.argsort(-pf[c])][: a["cap"]]]
    a = PATTERNS["ana"]
    if pt is not None:
        c = np.where(okt & (win_odds_first >= a["win_odds"]) & (pt * o_t >= a["th"]) & (pt >= a["pmin"]))[0]
        out["ana"] = [int(i) for i in c[np.argsort(-pt[c])][: a["cap"]]]
    return out


# 3連単は3連複と別の枠。1点100円の均等買い（検証と同じ）。1レース・1日の上限だけ決める
TRI_RACE_CAP, TRI_DAY_CAP = 1000, 5000
ANA_RACE_CAP = 2000   # 大穴：1点100円・20点まで
try:
    TRI_RACE_CAP = int(_cfg.get("tri_race_cap", TRI_RACE_CAP)); TRI_DAY_CAP = int(_cfg.get("tri_day_cap", TRI_DAY_CAP))
    ANA_RACE_CAP = int(_cfg.get("ana_race_cap", ANA_RACE_CAP))
except Exception:  # noqa: BLE001
    pass
# 上限を直接決めたとき（config/bet.json の race_cap・day_cap・big_cap）はそちらを使う
try:
    RACE_CAP = int(_cfg.get("race_cap", RACE_CAP)); DAY_CAP = int(_cfg.get("day_cap", DAY_CAP)); BIG_CAP = int(_cfg.get("big_cap", BIG_CAP))
except Exception:  # noqa: BLE001
    pass


def stakes(entries: list[dict], room: int) -> list[int]:
    """entries: [{p, o}]（期待値の高い順）。room: この日に残っている予算。戻り値: 各組の金額（0は買わない）"""
    st = []
    for e in entries:
        p, o = e["p"], e["o"]
        f = max(0.0, (p * o - 1) / (o - 1)) * KELLY if o > 1 else 0
        st.append(max(100, int(round(BANK * f / 100)) * 100))
    cap = min(RACE_CAP, max(0, room))
    tot = sum(st)
    if tot > cap:
        s = cap / tot if tot else 0
        st = [int(v * s / 100) * 100 for v in st]
        while sum(st) > cap:
            i = min((i for i, v in enumerate(st) if v), key=lambda i: entries[i]["p"] * entries[i]["o"]); st[i] = 0
    return st
