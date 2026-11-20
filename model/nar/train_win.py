"""地方：勝ち馬モデル。① 2018〜2025年で、人気順位（全期間にある）を入れたうえで指数の重みを学習（人気に上乗せできる分）。
② オッズのある期間の前半（2026-02〜05）で、単勝オッズの人気と①のAIの点数の組み合わせを学習。"""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/keiyomi/model"); sys.path.insert(0, "/home/claude/keiyomi/model/nar")
import pl, features as F

d = pd.read_pickle("/tmp/nar_feat.pkl")
d = d[d.pos.notna() & d.no.notna()].copy()
d["pop"] = d["pop"].where(d["pop"] > 0); d["lpop"] = -np.log(d["pop"].fillna(d.n))
O = pickle.load(open("/tmp/nar_odds.pkl", "rb"))
tr = d[(d.date >= "20180101") & (d.date <= "20251231")]
ho = d[(d.date >= "20260101") & (d.date <= "20260131")]
res = {}
for nm, cols in (("人気だけ", ["lpop"]), ("人気＋指数", ["lpop"] + F.FEATS)):
    P = pl.pack(tr.reset_index(drop=True), cols); w = pl.fit(P, l2=1.0)
    Ph = pl.pack(ho.reset_index(drop=True), cols); v = pl._loglik(w, Ph, 0, 1)[0] / len(Ph["X"])
    res[nm] = w; print(nm, "1月の当てはまり（1着, NLL）", round(v, 4), flush=True)
w = res["人気＋指数"]
print("重み", dict(zip(["lpop"] + F.FEATS, np.round(w, 3))))
WF = w[1:]
# ② オッズのある期間：単勝オッズの人気（対数）＋ AIの点数
od = d[d.rid.isin(O.keys())].copy()
def q_of(row):
    o = O[row.rid]; i = o["nos"].index(int(row.no)) if int(row.no) in o["nos"] else None
    return o["win"][i] if i is not None else 0
od["wo"] = [q_of(r) for r in od.itertuples()]
od = od[od.wo > 0]
od["lq"] = np.log(1 / od.wo); od["lq"] -= np.log(od.groupby("rid").lq.transform(lambda s: np.exp(s).sum()))
od["ai"] = od[F.FEATS].to_numpy() @ WF
od.to_pickle("/tmp/nar_od.pkl")
a = od[od.date <= "20260531"]; b = od[(od.date >= "20260601") & (od.date <= "20260731")]
for nm, cols in (("単勝人気だけ", ["lq"]), ("単勝人気＋AI", ["lq", "ai"])):
    P = pl.pack(a.reset_index(drop=True), cols); w2 = pl.fit(P, l2=0.01)
    Pb = pl.pack(b.reset_index(drop=True), cols)
    print(nm, np.round(w2, 3), "6-7月 NLL 1着", round(pl._loglik(w2, Pb, 0, 1)[0] / len(Pb["X"]), 4), "3着まで", round(pl._loglik(w2, Pb, 0, 3)[0] / len(Pb["X"]), 4), flush=True)
    res[nm] = w2
pickle.dump({"WF": WF, "W2": res["単勝人気＋AI"], "WM": res["単勝人気だけ"]}, open("/tmp/nar_w.pkl", "wb"))
