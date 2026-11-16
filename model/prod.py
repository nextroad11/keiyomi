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
    "tri": {"th": 1.0, "cap": 48, "from_trio": True},
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
        idx = np.where(okt & want & (ev >= r["th"]))[0]
        res["tri"] = [int(i) for i in idx[np.argsort(-ev[idx])][: r["cap"]]]
    return res
