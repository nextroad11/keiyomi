"""地方：買い方の設定。06〜07月だけで選び（的中が少ない設定は選ばない）、08〜11月に1回だけ当てる。上位3つの払戻を除いた回収率も出す。"""
import pickle, sys, numpy as np
src = open("/home/claude/keiyomi/model/experiments/18_all_types_rules.py").read()
exec(src.split("data = pickle.load")[0])
data = pickle.load(open("/tmp/nar_alltypes.pkl", "rb")); W = pickle.load(open("/tmp/nar_alltypes_w.pkl", "rb"))
VA = [d for d in data if "20260601" <= d["date"] <= "20260731"]; TE = [d for d in data if d["date"] >= "20260801"]
print("基準決め", len(VA), "レース / 採点", len(TE), "レース")
rng = np.random.default_rng(0)
for d in VA + TE:
    for t, x in d["types"].items():
        p = probs(x, W[t], t); x["p"] = p; x["ev"] = p * x["o"]
        ks = x["keys"].tolist() if hasattr(x["keys"], "tolist") else x["keys"]
        x["ret"] = np.array([x["pay"].get(k, 0) for k in ks], dtype=np.float64)
exec("def pick" + src.split("def pick")[1].split("GRID = {")[0])


def ex3(R, t, a):
    r = []
    for d in R:
        x = d["types"].get(t)
        if x:
            c = pick(x, a, t); r += list(x["ret"][c])
    r = np.array(r); s = np.sort(r)[::-1]
    return (r.sum() - s[:3].sum()) / (len(r) * 100) if len(r) else np.nan


def allbuy(R, t):
    c = s = 0
    for d in R:
        x = d["types"].get(t)
        if x:
            c += len(x["ret"]) * 100; s += x["ret"].sum()
    return s / c


GRID = {t: [dict(th=th, pmin=pm, omin=om) for th in (1.0, 1.1, 1.2, 1.3, 1.5) for pm in pms for om in oms]
        for t, pms, oms in [("単勝", (0.02, 0.05, 0.1), (0, 5, 10)), ("複勝", (0.1, 0.2, 0.3), (0, 2, 3)), ("枠連", (0.01, 0.02, 0.05), (0, 10, 20)),
                            ("馬連", (0.005, 0.01, 0.02), (0, 10, 30)), ("ワイド", (0.02, 0.05, 0.1), (0, 5, 10)), ("馬単", (0.003, 0.005, 0.01), (0, 20, 50)),
                            ("3連複", (0.003, 0.005, 0.01), (0, 10, 30)), ("3連単", (0.001, 0.002, 0.005), (0, 30, 100))]}
MINHIT = {"単勝": 40, "複勝": 60, "枠連": 30, "馬連": 30, "ワイド": 40, "馬単": 25, "3連複": 30, "3連単": 15}
chosen = {}
print("券種 | 全部買った場合（採点期間） | 選んだ設定 | 基準決め | 採点（1回だけ） | 採点・上位3つ除く")
for t in TYPES:
    res = [(a, run(VA, t, a)) for a in GRID[t]]
    ok = sorted([(a, r) for a, r in res if r and r["hit"] >= MINHIT[t]], key=lambda z: -z[1]["roi"])
    if not ok:
        print(t, "条件を満たす設定なし"); continue
    a, v = ok[0]; chosen[t] = a
    te = run(TE, t, a, boot=True)
    print(f"{t} | {allbuy(TE, t):.1%} | {a} | {f(v)} | {f(te)} | {ex3(TE, t, a):.1%}", flush=True)
pickle.dump(chosen, open("/tmp/nar_chosen.pkl", "wb"))
