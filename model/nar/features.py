"""地方競馬の指数と前走までの成績（その日より前の情報だけ）。"""
import numpy as np, pandas as pd

GOING = {"良": 0, "稍重": 1, "重": 2, "不良": 3}


def build(R, H):
    d = H.merge(R[["rid", "dist", "going", "n", "surface", "race3f"]], on="rid", how="left")
    d = d[d.surface == "ダート"].copy()
    d["g"] = d.going.map(GOING).fillna(1)
    d["ok"] = d.sec.between(30, 400) & d.pos.notna()
    # 基準タイム：場・距離・馬場ごと（2025年までの勝ち馬以外も含む全頭の中央値と散らばり）
    base = d[d.ok & (d.date <= "20251231")]
    st = base.groupby(["venue", "dist", "g"]).sec.agg(["median", "std", "size"]).reset_index()
    st = st[st["size"] >= 50]
    d = d.merge(st[["venue", "dist", "g", "median", "std"]], on=["venue", "dist", "g"], how="left")
    d["z"] = -(d.sec - d["median"]) / d["std"]
    lb = base.groupby(["venue", "dist", "g"]).l3f.agg(["median", "std"]).reset_index().rename(columns={"median": "lm", "std": "ls"})
    d = d.merge(lb, on=["venue", "dist", "g"], how="left")
    d["zl"] = -(d.l3f - d.lm) / d.ls
    # その日・その場の馬場の速さ（全頭の中央値）を引く
    day = d[d.ok].groupby(["date", "venue"]).z.median().rename("var")
    d = d.merge(day, on=["date", "venue"], how="left")
    d["fig"] = (d.z - d["var"]).clip(-5, 4)
    dayl = d[d.ok].groupby(["date", "venue"]).zl.median().rename("varl")
    d = d.merge(dayl, on=["date", "venue"], how="left")
    d["figl"] = (d.zl - d.varl).clip(-5, 4)
    d.loc[~d.ok, ["fig", "figl"]] = np.nan
    d["win"] = (d.pos == 1).astype(float); d["top3"] = (d.pos <= 3).astype(float)
    d["rpos"] = (d.pos - 1) / (d.n - 1).clip(lower=1)
    d["dt"] = pd.to_datetime(d.date)
    d = d.sort_values(["hid", "dt", "rno"]).reset_index(drop=True)
    g = d.groupby("hid", sort=False)
    sh = lambda c: g[c].shift(1)
    d["f1"] = sh("fig"); d["f2"] = g["fig"].shift(2); d["f3"] = g["fig"].shift(3)
    d["fl1"] = sh("figl")
    d["fbest"] = d[["f1", "f2", "f3"]].max(1); d["fmean"] = d[["f1", "f2", "f3"]].mean(1)
    d["flbest"] = pd.concat([d.fl1, g["figl"].shift(2), g["figl"].shift(3)], axis=1).max(1)
    d["rpos1"] = sh("rpos"); d["rpos2"] = g["rpos"].shift(2)
    d["runs"] = g.cumcount()
    d["wins"] = g["win"].cumsum() - d.win; d["t3s"] = g["top3"].cumsum() - d.top3
    d["wr"] = (d.wins + 0.1) / (d.runs + 1); d["t3r"] = (d.t3s + 0.3) / (d.runs + 1)
    d["days"] = (d.dt - g["dt"].shift(1)).dt.days
    d["dist1"] = sh("dist"); d["ddist"] = (d.dist - d.dist1) / 400
    d["venue1"] = sh("venue"); d["moved"] = (d.venue1.notna() & (d.venue1 != d.venue)).astype(float)
    # 騎手・調教師：その日より前の勝率（日単位でまとめて、同じ日の結果は使わない）
    for who in ("jockey", "trainer"):
        x = d.groupby([who, "date"]).agg(w=("win", "sum"), r=("win", "size"), t=("top3", "sum")).reset_index().sort_values([who, "date"])
        gg = x.groupby(who)
        x[who[0] + "_wr"] = (gg.w.cumsum() - x.w + 1) / (gg.r.cumsum() - x.r + 10)
        x[who[0] + "_t3"] = (gg.t.cumsum() - x.t + 3) / (gg.r.cumsum() - x.r + 10)
        d = d.merge(x[[who, "date", who[0] + "_wr", who[0] + "_t3"]], on=[who, "date"], how="left")
    return d


FEATS = ["f1", "fbest", "fmean", "fl1", "flbest", "rpos1", "rpos2", "lruns", "wr", "t3r", "ldays", "ddist", "moved",
         "j_wr", "j_t3", "t_wr", "wt_r", "bd", "age", "waku_r", "no_f1", "no_fl"]


def finalize(d):
    d = d.copy()
    d["lruns"] = np.log1p(d.runs); d["ldays"] = np.log1p(d.days.fillna(60).clip(0, 400))
    d["no_f1"] = d.f1.isna().astype(float); d["no_fl"] = d.fl1.isna().astype(float)
    d["wt_r"] = d.wt - d.groupby("rid").wt.transform("mean")
    d["bd"] = d.body_diff.fillna(0).clip(-20, 20) / 10
    d["waku_r"] = (d.no - 1) / (d.n - 1).clip(lower=1)
    d["j_wr"] = np.log(d.j_wr); d["j_t3"] = np.log(d.j_t3); d["t_wr"] = np.log(d.t_wr)
    d["wr"] = np.log(d.wr); d["t3r"] = np.log(d.t3r)
    for c in FEATS:
        d[c] = d[c].astype(float)
        m = d.groupby("rid")[c].transform("mean")
        d[c] = d[c].fillna(m).fillna(0)
        # レース内で標準化
        sd = d.groupby("rid")[c].transform("std").replace(0, 1).fillna(1)
        d[c] = ((d[c] - d.groupby("rid")[c].transform("mean")) / sd).clip(-4, 4)
    d["lpop"] = -np.log(d["pop"].fillna(d.n))
    return d
