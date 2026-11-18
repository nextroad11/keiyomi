"""全券種の買い方：期待値・確率・オッズの下限で選ぶ。基準決め期間だけで設定を選び（的中が少ない設定は選ばない）、採点期間に1回だけ当てる。
3連単は1レース48点まで。ほかは上限なし。最後に「レースごとに割に合う券種を選ぶ」組み合わせも見る。"""
import pickle, sys, numpy as np
sys.argv = ["x"]
exec(open("/home/claude/keiyomi/model/experiments/17_all_types.py").read().split('if __name__ == "__main__":')[0].split("D = pickle.load")[0])
exec("def probs" + open("/home/claude/keiyomi/model/experiments/17_all_types.py").read().split("def probs")[1].split('if __name__')[0])
TYPES = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]
data = pickle.load(open("/tmp/alltypes.pkl", "rb")); W = pickle.load(open("/tmp/alltypes_w.pkl", "rb"))
VA = [d for d in data if "20250601" <= d["date"] <= "20260531"]; TE = [d for d in data if d["date"] >= "20260601"]
rng = np.random.default_rng(0)
for d in VA + TE:
    for t, x in d["types"].items():
        p = probs(x, W[t], t); x["p"] = p; x["ev"] = p * x["o"]
        ks = x["keys"].tolist() if hasattr(x["keys"], "tolist") else x["keys"]
        x["ret"] = np.array([x["pay"].get(k, 0) for k in ks], dtype=np.float64)


def pick(x, a, t):
    c = np.where((x["ev"] >= a["th"]) & (x["p"] >= a["pmin"]) & (x["o"] >= a["omin"]))[0]
    c = c[np.argsort(-x["ev"][c])]
    return c[:48] if t == "3連単" else c


def run(R, t, a, boot=False):
    out = []
    for d in R:
        x = d["types"].get(t)
        if not x:
            continue
        c = pick(x, a, t); out.append((len(c), x["ret"][c].sum(), int((x["ret"][c] > 0).sum())))
    A = np.array(out, float); cost = A[:, 0].sum() * 100
    if cost == 0:
        return None
    res = dict(pts=int(A[:, 0].sum()), races=int((A[:, 0] > 0).sum()), hit=int(A[:, 2].sum()), roi=A[:, 1].sum() / cost)
    if boot:
        bs = [A[i, 1].sum() / (A[i, 0].sum() * 100) for i in (rng.integers(0, len(A), len(A)) for _ in range(2000)) if A[i, 0].sum()]
        res["lo"], res["hi"] = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    return res


def f(d):
    if not d:
        return "買い目なし"
    s = f"{d['races']}R・{d['pts']}点 的中{d['hit']} 回収率{d['roi']:.1%}"
    return s + (f"（95%: {d['lo']:.0%}〜{d['hi']:.0%}）" if "lo" in d else "")


GRID = {t: [dict(th=th, pmin=pm, omin=om) for th in (1.0, 1.1, 1.2, 1.3, 1.5) for pm in pms for om in oms]
        for t, pms, oms in [("単勝", (0.02, 0.05, 0.1), (0, 5, 10)), ("複勝", (0.1, 0.2, 0.3), (0, 2, 3)), ("枠連", (0.01, 0.02, 0.05), (0, 10, 20)),
                            ("馬連", (0.005, 0.01, 0.02), (0, 10, 30)), ("ワイド", (0.02, 0.05, 0.1), (0, 5, 10)), ("馬単", (0.003, 0.005, 0.01), (0, 20, 50)),
                            ("3連複", (0.003, 0.005, 0.01), (0, 10, 30)), ("3連単", (0.001, 0.002, 0.005), (0, 30, 100))]}
MINHIT = {"単勝": 40, "複勝": 60, "枠連": 30, "馬連": 30, "ワイド": 40, "馬単": 25, "3連複": 30, "3連単": 20}
BASE = {t: 1 - v for t, v in {"単勝": .2, "複勝": .2, "枠連": .225, "馬連": .225, "ワイド": .225, "馬単": .25, "3連複": .25, "3連単": .275}.items()}
chosen = {}
print("券種 | 払戻率 | 基準決めで選んだ設定 | 基準決め | 採点（1回だけ）")
for t in TYPES:
    res = [(a, run(VA, t, a)) for a in GRID[t]]
    ok = sorted([(a, r) for a, r in res if r and r["hit"] >= MINHIT[t]], key=lambda z: -z[1]["roi"])
    if not ok:
        print(t, "条件を満たす設定なし"); continue
    a, v = ok[0]; chosen[t] = a
    te = run(TE, t, a, boot=True)
    top5 = [run(TE, t, b)["roi"] for b, _ in ok[:5] if run(TE, t, b)]
    print(f"{t} | {BASE[t]:.1%} | {a} | {f(v)} | {f(te)} | 上位5設定の採点 {[f'{x:.0%}' for x in top5]}", flush=True)
pickle.dump(chosen, open("/tmp/alltypes_chosen.pkl", "wb"))
