"""追加の特徴量（第2弾）。すべて「そのレースより前」の情報だけ。

- 着差（勝ち馬からの秒差）、上がりの順位、道中の位置取りの上げ下げ
- 前走の人気と着順の差（人気以上に走ったか）
- 同じ場・芝ダでの最高指数、近い距離（±200m）での平均指数
- 斤量の増減、乗り替わり（騎手の勝率の変化）、調教師の直近1年勝率
- 馬番の内外（芝・ダート別）
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd


def add_race_relative(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("rid")
    df["tdiff"] = (df["time"] - g["time"].transform("min")).clip(upper=5)
    df["f3rank"] = (g["f3"].rank(method="average") - 1) / (df["n"] - 1).clip(lower=1)
    df["gain"] = ((df["corner1"] - df["pos"]) / df["n"].clip(lower=1))
    df["pop_rel"] = (df["pop"] - 1) / (df["n"] - 1).clip(lower=1)
    df["pos_rel"] = (df["pos"] - 1) / (df["n"] - 1).clip(lower=1)
    df["draw"] = (df["no"] - 1) / (df["n"] - 1).clip(lower=1)
    return df


def add_history2(df: pd.DataFrame, trainer_col: str = "trainer") -> pd.DataFrame:
    df = df.sort_values(["dt", "rid", "no"]).reset_index(drop=True)
    hist = defaultdict(list)
    tq, tsum = defaultdict(deque), defaultdict(int)
    jrate_last = {}
    feats = []
    for dt, g in df.groupby("dt", sort=True):
        for i, x in g.iterrows():
            past = [p for p in hist.get(x["hid"], []) if p["dt"] < dt][-10:] if x["hid"] else []
            last = past[-1] if past else None
            l3 = past[-3:]
            sv = [p["fig"] for p in past if p["jcd"] == x["jcd"] and p["surface"] == x["surface"] and p["fig"] == p["fig"]]
            dv = [p["fig"] for p in past if p["surface"] == x["surface"] and p["distance"] and x["distance"]
                  and abs(p["distance"] - x["distance"]) <= 200 and p["fig"] == p["fig"]]
            q = tq[x.get(trainer_col)]
            while q and (dt - q[0][0]).days > 365:
                tsum[x.get(trainer_col)] -= q.popleft()[1]
            feats.append({
                "idx": i,
                "tdiff_last": last["tdiff"] if last and last["tdiff"] == last["tdiff"] else np.nan,
                "tdiff_mean": np.nanmean([p["tdiff"] for p in l3]) if l3 else np.nan,
                "f3rank_last": last["f3rank"] if last else np.nan,
                "f3rank_mean": np.nanmean([p["f3rank"] for p in l3]) if l3 else np.nan,
                "gain_mean": np.nanmean([p["gain"] for p in l3]) if l3 else np.nan,
                "over_pop": np.nanmean([p["pop_rel"] - p["pos_rel"] for p in l3]) if l3 else np.nan,
                "venue_best": max(sv) if sv else np.nan,
                "dist_fig": float(np.mean(dv)) if dv else np.nan,
                "wt_chg": x["weight"] - last["weight"] if last and x["weight"] == x["weight"] and last["weight"] == last["weight"] else np.nan,
                "jk_up": x["jk_rate"] - jrate_last.get(x["hid"], np.nan) if x["hid"] in jrate_last else np.nan,
                "tr_rate": (tsum[x.get(trainer_col)] + 5 * 0.07) / (len(q) + 5),
                "draw_turf": x["draw"] if x["surface"] == "芝" else 0.0,
                "draw_dirt": x["draw"] if x["surface"] == "ダート" else 0.0,
            })
        for i, x in g.iterrows():
            if x["hid"]:
                hist[x["hid"]].append({k: x[k] for k in ("dt", "fig", "tdiff", "f3rank", "gain", "pop_rel", "pos_rel", "jcd", "surface", "distance", "weight")})
                jrate_last[x["hid"]] = x["jk_rate"]
            t = x.get(trainer_col)
            if t:
                w = 1 if x["pos"] == 1 else 0
                tq[t].append((dt, w)); tsum[t] += w
    f = pd.DataFrame(feats).set_index("idx")
    return df.join(f)


FEATURES2 = ["tdiff_last", "tdiff_mean", "f3rank_last", "f3rank_mean", "gain_mean", "over_pop", "venue_best", "dist_fig",
             "wt_chg", "jk_up", "tr_rate", "draw_turf", "draw_dirt"]


def zscore(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        g = df.groupby("rid")[c]
        mu, sd = g.transform("mean"), g.transform("std")
        df["z_" + c] = ((df[c] - mu) / sd.replace(0, np.nan)).fillna(0).clip(-4, 4)
    return df
