"""5年分での3連複・3連単。
学習 2022-03〜2025-05 / 基準決め 2025-06〜2026-05 / 採点 2026-06〜2026-11（第2回で一部の買い方の採点に一度使っている）
1) AIの勝ち馬モデル（人気＋指数、Plackett-Luce）を学習期間で学習
2) 実オッズ＋AI＋単勝人気PL の組み合わせの重みを学習期間で学習
3) 各レース・各組の確率とオッズを保存（買い方の検証は 13_rules.py）
"""
import pickle, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/keiyomi/model")
import features as F, pl, exotic as X
from scipy.optimize import minimize

TR, VA, TE = ("20220301", "20250531"), ("20250601", "20260531"), ("20260601", "20261131")
df = pd.read_pickle("/tmp/df5.pkl")
df = df[(df.surface != "障害") & df.pos.notna() & df.odds.notna()].copy().reset_index(drop=True)
inv = 1 / df.odds; df["p_mkt"] = inv / inv.groupby(df.rid).transform("sum"); df["logp_mkt"] = np.log(df.p_mkt)
Z = ["z_" + c for c in F.FEATURES]
tr = df[(df.date >= TR[0]) & (df.date <= TR[1])]
Pt = pl.pack(tr, ["logp_mkt"] + Z)
W3 = pl.fit(Pt, l2=3, stages=3)
WM = pl.fit(pl.pack(tr, ["logp_mkt"]), l2=3, stages=3)
print("勝ち馬モデル 学習レース", len(Pt["rids"]))
for nm, (a, b) in [("基準決め", VA), ("採点", TE)]:
    d = df[(df.date >= a) & (df.date <= b)]
    for cols, w, lab in [(["logp_mkt"], WM, "人気のみ"), (["logp_mkt"] + Z, W3, "AI")]:
        P = pl.pack(d, cols); print(f"  {nm} 上位3着NLL {lab}", round(pl._loglik(w, P, 0, 3)[0] / len(P["rids"]), 4))

R = {r["id"]: r for r in pickle.load(open("/tmp/races5.pkl", "rb"))}
EX = pickle.load(open("/tmp/ex5.pkl", "rb"))
rows = []
for rid, g in df.groupby("rid", sort=False):
    if rid not in EX or g.date.iloc[0] < TR[0]:
        continue
    nos, trio, tri = EX[rid]; n = len(nos)
    if len(tri) != n * (n - 1) * (n - 2):
        continue
    race = R[rid]
    if "3連単" not in race["pay"] or len(race["pay"]["3連単"]) > 1 or len(race["pay"].get("3連複", [])) > 1:
        continue  # 同着は除く
    pos = {no: i for i, no in enumerate(nos)}
    def st(w, cols):
        sc = g[cols].to_numpy() @ w; e = np.exp(sc - sc.max()); e /= e.sum(); s = np.zeros(n)
        for i, no in enumerate(g["no"]): s[pos[no]] = e[i]
        return s
    s_ai, s_mk = st(W3, ["logp_mkt"] + Z), st(WM, ["logp_mkt"])
    pt, pmt = X.pl_tri(s_ai), X.pl_tri(s_mk)
    P, C, _ = X.perms(n)
    w3 = [pos[int(x)] for x in race["pay"]["3連単"][0]["combo"].split("-")]
    wt = int(np.where((P[:, 0] == w3[0]) & (P[:, 1] == w3[1]) & (P[:, 2] == w3[2]))[0][0]); sw = sorted(w3)
    wf = int(np.where((C[:, 0] == sw[0]) & (C[:, 1] == sw[1]) & (C[:, 2] == sw[2]))[0][0])
    rows.append(dict(rid=rid, date=race["date"], n=n,
                     lp_t=np.log(pt + 1e-12).astype(np.float32), lm_t=np.log(pmt + 1e-12).astype(np.float32), o_t=tri,
                     lp_f=np.log(X.trio_from_tri(pt, n) + 1e-12).astype(np.float32), lm_f=np.log(X.trio_from_tri(pmt, n) + 1e-12).astype(np.float32), o_f=trio,
                     wt=wt, wf=wf, pay_t=race["pay"]["3連単"][0]["yen"], pay_f=race["pay"]["3連複"][0]["yen"]))
print("3連のレース", len(rows))


def nll(w, D, k):
    L = 0; G = np.zeros(3)
    for r in D:
        o = r["o_" + k]; ok = o > 0
        q = np.where(ok, 1 / np.where(ok, o, 1), 0); q /= q.sum()
        Xm = np.stack([np.log(q + 1e-9), r["lp_" + k], r["lm_" + k]], 1).astype(np.float64)
        s = np.where(ok, Xm @ w, -1e9); m = s.max(); e = np.exp(s - m) * ok; Zs = e.sum(); p = e / Zs; wi = r["w" + k]
        if not ok[wi]:
            continue
        L += s[wi] - m - np.log(Zs); G += Xm[wi] - (p[:, None] * Xm).sum(0)
    return -L, -G


TRr = [r for r in rows if r["date"] <= TR[1]]
BW = {}
for k, name in [("t", "3連単"), ("f", "3連複")]:
    w = minimize(nll, np.array([0.85, 0.7, -0.5]), args=(TRr, k), jac=True, method="L-BFGS-B").x
    BW[k] = w
    for nm, (a, b) in [("基準決め", VA), ("採点", TE)]:
        D = [r for r in rows if a <= r["date"] <= b]
        base = minimize(lambda x: nll(np.array([x[0], 0, 0]), TRr, k), [1.0]).x[0] if False else None
        print(name, "重み", np.round(w, 3), nm, "NLL", round(nll(w, D, k)[0] / len(D), 4), "実オッズのみ", round(nll(np.array([1.0, 0, 0]), D, k)[0] / len(D), 4))
pickle.dump({"rows": rows, "BW": BW, "W3": W3, "WM": WM}, open("/tmp/five.pkl", "wb"), protocol=4)
