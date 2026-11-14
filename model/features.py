"""特徴量づくり。すべて「そのレースより前」に分かる情報だけを使う。

指数（走破タイム・上がり3F）の作り方
1. 基準タイム：場・芝ダ・距離ごとの平均と散らばり。学習期間のレースだけで決める（採点期間の情報は混ぜない）。
2. 馬場差：同じ日・同じ場・同じ芝ダの全出走馬が、基準からどれだけ速かったかの中央値。
   その日のレースが終わったあとに分かる値なので、使うのは「その日より後のレース」の特徴量としてだけ。
3. 指数＝ −（基準からのずれ − 馬場差）。大きいほど速い。
4. レースの中で標準化して、出走馬どうしを同じものさしで比べる。
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd

TAKE = {"単勝": 0.20, "複勝": 0.20, "枠連": 0.225, "馬連": 0.225, "ワイド": 0.225, "馬単": 0.25, "3連複": 0.25, "3連単": 0.275}


def surface_of(r: dict) -> str:
    name = r.get("name") or ""
    if "障害" in name or "ジャンプ" in name or "障害" in (r.get("course_detail") or ""):
        return "障害"
    return r.get("surface") or "不明"


def secs(t):
    if not t:
        return None
    m = re.match(r"(?:(\d+):)?(\d+)\.(\d)", t)
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + int(m.group(2)) + int(m.group(3)) / 10


def class_level(cls: str | None, name: str | None) -> float:
    c = (cls or "") + (name or "")
    for k, v in (("新馬", 0), ("未勝利", 0), ("1勝", 1), ("500万", 1), ("2勝", 2), ("1000万", 2), ("3勝", 3), ("1600万", 3)):
        if k in c:
            return v
    if re.search(r"G[1Ⅰ]|ＧⅠ|GⅠ", c):
        return 6
    if re.search(r"G[23ⅡⅢ]|ＧⅡ|ＧⅢ|GⅡ|GⅢ", c):
        return 5
    if "オープン" in c or "リステッド" in c or "L" in (cls or ""):
        return 4
    return 2


def runners_frame(races: list[dict]) -> pd.DataFrame:
    rows = []
    for r in races:
        surf = surface_of(r)
        odds = r.get("odds_final") or {}
        n = len(r["runners"])
        for x in r["runners"]:
            o = odds.get(str(x["no"]), {}) if odds else {}
            sa = x.get("sexage") or ""
            rows.append({
                "rid": r["id"], "date": r["date"], "jcd": r["jcd"], "surface": surf, "distance": r.get("distance"),
                "going": r.get("going"), "cls": class_level(r.get("cls"), r.get("name")), "n": n,
                "no": x["no"], "waku": x.get("waku"), "hid": x.get("hid"), "jockey": (x.get("jockey") or "").lstrip("▲△☆★◇ "),
                "pos": x.get("pos"), "time": secs(x.get("time")), "f3": x.get("f3"),
                "corner1": _first_corner(x.get("corner")), "body": x.get("body"), "body_diff": x.get("body_diff"),
                "weight": x.get("weight"), "sex": sa[:1] or None, "age": _int(sa[1:]),
                "pop": x.get("pop"), "odds": o.get("win"), "pmin": o.get("pmin"),
            })
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def _first_corner(c):
    if not c:
        return None
    m = re.match(r"(\d+)", c)
    return int(m.group(1)) if m else None


def _int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def add_figures(df: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """走破タイム指数・上がり指数（レース後に分かる値。次のレースからの特徴量に使う）。"""
    df = df.copy()
    ok = df["pos"].notna() & df["time"].notna() & (df["surface"] != "障害")
    tr = df[ok & (df["date"] <= train_end)]
    key = ["jcd", "surface", "distance"]
    std_t = tr.groupby(key)["time"].agg(["median", "std", "count"]).rename(columns={"median": "t_mu", "std": "t_sd", "count": "t_n"})
    std_f = tr[tr["f3"].notna()].groupby(key)["f3"].agg(["median", "std"]).rename(columns={"median": "f_mu", "std": "f_sd"})
    df = df.join(std_t, on=key).join(std_f, on=key)
    df.loc[df["t_n"].fillna(0) < 30, ["t_mu", "t_sd", "f_mu", "f_sd"]] = np.nan  # 少なすぎるコースは使わない
    df["t_z"] = (df["time"] - df["t_mu"]) / df["t_sd"]
    df["f_z"] = (df["f3"] - df["f_mu"]) / df["f_sd"]
    # 馬場差：同じ日・場・芝ダの中央値（極端な遅れは除く）
    day = ["date", "jcd", "surface"]
    m = ok & df["t_z"].between(-4, 4)
    df["t_var"] = df[m].groupby(day)["t_z"].transform("median")
    df["t_var"] = df.groupby(day)["t_var"].transform("max")
    df["f_var"] = df[m].groupby(day)["f_z"].transform("median")
    df["f_var"] = df.groupby(day)["f_var"].transform("max")
    df["fig"] = -(df["t_z"] - df["t_var"])
    df["f3fig"] = -(df["f_z"] - df["f_var"])
    df.loc[~ok, ["fig", "f3fig"]] = np.nan
    df["fig"] = df["fig"].clip(-5, 5)
    df["f3fig"] = df["f3fig"].clip(-5, 5)
    return df


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """各出走について、その馬・騎手の「それより前のレース」から特徴量を作る。"""
    df = df.sort_values(["dt", "rid", "no"]).reset_index(drop=True)
    hist = defaultdict(list)  # hid -> [(dt, fig, f3fig, posrel, c1rel, distance, surface, cls)]
    feats = []
    # 騎手：直近365日の騎乗と勝利
    from collections import deque
    jk = defaultdict(deque)  # jockey -> deque[(dt, win)]（365日より古いものは捨てる）
    jsum = defaultdict(int)
    for dt, g in df.groupby("dt", sort=True):
        day_rows = []
        for i, x in g.iterrows():
            h = hist.get(x["hid"], []) if x["hid"] else []
            past = [p for p in h if p[0] < dt][-10:]
            same = [p for p in past if p[6] == x["surface"]]
            pool = same if same else past
            figs = [p[1] for p in pool[-3:] if p[1] == p[1] and p[1] is not None]
            f3s = [p[2] for p in pool[-3:] if p[2] == p[2] and p[2] is not None]
            posr = [p[3] for p in past[-3:] if p[3] is not None]
            c1 = [p[4] for p in past[-5:] if p[4] is not None]
            q = jk[x["jockey"]]
            while q and (dt - q[0][0]).days > 365:
                jsum[x["jockey"]] -= q.popleft()[1]
            jn, jw = len(q), jsum[x["jockey"]]
            feats.append({
                "idx": i,
                "runs": len(past),
                "debut": 1.0 if not past else 0.0,
                "days": math.log1p((dt - past[-1][0]).days) if past else np.nan,
                "fig_last": figs[-1] if figs else np.nan,
                "fig_best": max(figs) if figs else np.nan,
                "fig_mean": float(np.mean(figs)) if figs else np.nan,
                "f3_last": f3s[-1] if f3s else np.nan,
                "f3_best": max(f3s) if f3s else np.nan,
                "pos_last": posr[-1] if posr else np.nan,
                "pos_mean": float(np.mean(posr)) if posr else np.nan,
                "front": float(np.mean(c1)) if c1 else np.nan,  # 1コーナーの位置（小さいほど前）
                "dist_chg": (x["distance"] - past[-1][5]) / 400 if past and x["distance"] and past[-1][5] else np.nan,
                "cls_chg": x["cls"] - past[-1][7] if past else np.nan,
                "surf_new": 1.0 if past and not same else 0.0,
                "jk_rate": (jw + 5 * 0.07) / (jn + 5),
                "jk_n": math.log1p(jn),
            })
        # その日のレースが終わってから履歴に足す
        for i, x in g.iterrows():
            if x["hid"]:
                posrel = (x["pos"] - 1) / max(1, x["n"] - 1) if x["pos"] == x["pos"] and x["pos"] is not None else None
                c1rel = (x["corner1"] - 1) / max(1, x["n"] - 1) if x["corner1"] == x["corner1"] and x["corner1"] is not None else None
                hist[x["hid"]].append((dt, x["fig"], x["f3fig"], posrel, c1rel, x["distance"], x["surface"], x["cls"]))
            if x["jockey"]:
                w = 1 if x["pos"] == 1 else 0
                jk[x["jockey"]].append((dt, w)); jsum[x["jockey"]] += w
    f = pd.DataFrame(feats).set_index("idx")
    return df.join(f)


FEATURES = ["fig_last", "fig_best", "fig_mean", "f3_last", "f3_best", "pos_last", "pos_mean", "front",
            "days", "dist_chg", "cls_chg", "surf_new", "debut", "runs_log", "jk_rate", "weight", "body_diff", "age", "female"]


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["runs_log"] = np.log1p(df["runs"])
    df["female"] = (df["sex"] == "牝").astype(float)
    # 市場（確定単勝オッズ）の確率
    inv = 1 / df["odds"]
    df["p_mkt"] = inv / inv.groupby(df["rid"]).transform("sum")
    df["logp_mkt"] = np.log(df["p_mkt"])
    # レース内で標準化（欠けは0＝レース平均、別に欠け印は debut で持つ）
    for c in FEATURES:
        g = df.groupby("rid")[c]
        mu, sd = g.transform("mean"), g.transform("std")
        df["z_" + c] = ((df[c] - mu) / sd.replace(0, np.nan)).fillna(0).clip(-4, 4)
    return df
