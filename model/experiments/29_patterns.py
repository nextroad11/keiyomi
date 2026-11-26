"""本線パターン・大穴パターンの条件を、基準決め期間（2025-06〜2026-05）だけで決める。採点期間は見ない。
本線：3連複でAIの確率が高い組（確率≧a、期待値≧b）を上位k点、1点同額。
大穴：3連単で「1着に置く馬の単勝オッズ≧c」の並びのうち期待値≧d・確率≧0.1% を確率の高い順にN点、1点100円。"""
import sys
src = open("/home/claude/keiyomi/model/experiments/26_ten_years.py").read()
exec(src.split("rng = np.random")[0])
W3, WM, BW = pickle.load(open(C / "models.pkl", "rb"))["A 2022-03〜"]
rids = df[(df.date >= VA[0]) & (df.date <= VA[1])].rid.unique()
D = []
for rid in rids:
    a = arrays(rid, W3, WM)
    if not a: continue
    g = G[rid]; v = EX[rid]; nos = v["nos"]; wo = dict(zip(g["no"], g["odds"]))
    pr, _ = prod.exotic_probs(a["s_ai"], a["s_mk"], a["tri"], a["trio"], BW)
    P_, C_, M_ = X.perms(a["n"])
    a["pf"], a["okf"] = pr["f"]; a["pt"], a["okt"] = pr["t"]
    a["w1"] = np.array([wo.get(nos[i], 0) or 0 for i in P_[:, 0]])
    D.append(a)
print("レース", len(D), flush=True)
def f(r, unit):
    r = np.array(r, float); s = np.sort(r)[::-1]; c = len(r) * unit
    return (len(r), int((r > 0).sum()), r.sum() / c if c else 0, (r.sum() - s[:3].sum()) / c if c else 0)
print("== 本線（3連複）")
best = []
for pa in (0.04, 0.06, 0.08, 0.12, 0.16):
    for evb in (0.9, 1.0, 1.1):
        for k in (1, 2, 3):
            r = []
            for a in D:
                ev = a["pf"] * a["trio"]; c = np.where(a["okf"] & (a["pf"] >= pa) & (ev >= evb))[0]
                c = c[np.argsort(-a["pf"][c])][:k]
                r += [a["pay_f"] if i == a["wf"] else 0 for i in c]
            if r:
                x = f(r, 100); best.append(((pa, evb, k), x)); print(pa, evb, k, f"{x[0]}点 的中{x[1]} 回収率{x[2]:.1%} 上位3つ除く{x[3]:.1%}", flush=True)
print("== 大穴（3連単）")
for c1 in (10, 20, 30):
    for evd in (1.0, 1.5, 2.0):
        for N in (10, 20, 30, 50):
            r = []; races = 0
            for a in D:
                ev = a["pt"] * a["tri"]; c = np.where(a["okt"] & (a["w1"] >= c1) & (ev >= evd) & (a["pt"] >= 0.001))[0]
                c = c[np.argsort(-a["pt"][c])][:N]
                races += bool(len(c)); r += [a["pay_t"] if i == a["wt"] else 0 for i in c]
            if r:
                x = f(r, 100); print(c1, evd, N, f"{races}R {x[0]}点 的中{x[1]} 回収率{x[2]:.1%} 上位3つ除く{x[3]:.1%}", flush=True)
