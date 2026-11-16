"""買い方の比較。基準決め期間だけで設定を選び（的中が少ない設定は選ばない）、最後に採点期間に1回だけ当てる。
3連単は1レース48点まで、3連複は点数の上限なし（ユーザーの指定）。"""
import pickle, sys, itertools, numpy as np
D = pickle.load(open("/tmp/five.pkl", "rb")); rows, BW = D["rows"], D["BW"]
VA = [r for r in rows if "20250601" <= r["date"] <= "20260531"]
TE = [r for r in rows if "20260601" <= r["date"] <= "20261131"]
rng = np.random.default_rng(0)
for r in rows:
    for k in "tf":
        o = r["o_" + k]; ok = o > 0
        q = np.where(ok, 1 / np.where(ok, o, 1), 0); q /= q.sum()
        s = BW[k][0] * np.log(q + 1e-9) + BW[k][1] * r["lp_" + k] + BW[k][2] * r["lm_" + k]
        s = np.where(ok, s, -1e9); e = np.exp(s - s.max()) * ok; r["p_" + k] = e / e.sum(); r["ok_" + k] = ok


def pick(r, k, rule, a):
    p, o, ok = r["p_" + k], r["o_" + k], r["ok_" + k]; ev = p * o
    if rule == "ev":
        c = np.where(ok & (ev >= a["th"]) & (p >= a["pmin"]) & (o >= a.get("omin", 0)))[0]
        c = c[np.argsort(-ev[c])]
        return c[: a["cap"]] if a.get("cap") else c
    if rule == "are":
        hi = ok & (o >= 100)
        if p[hi].sum() < a["man"]:
            return np.array([], int)
        idx = np.where(hi)[0]; idx = idx[np.argsort(-p[idx])]
        return idx[p[idx] >= a["step"]][:48]
    if rule == "f2t":  # 3連複で選んだ3頭の並びのうち、3連単でも期待値が高いもの
        pf, of, okf = r["p_f"], r["o_f"], r["ok_f"]
        cf = np.where(okf & (pf * of >= 1.0) & (pf >= 0.01))[0]
        if not len(cf):
            return np.array([], int)
        n = r["n"]; perm = np.array(list(itertools.permutations(range(n), 3)))
        comb = np.array(list(itertools.combinations(range(n), 3)))
        want = {tuple(comb[i]) for i in cf}
        mask = np.array([tuple(sorted(x)) in want for x in perm])
        c = np.where(mask & ok & (ev >= a["th"]))[0]
        return c[np.argsort(-ev[c])][:48]


def run(R, k, rule, a, boot=False):
    out = []
    for r in R:
        c = pick(r, k, rule, a); hit = int(r["w" + k] in set(c.tolist()))
        out.append((len(c), r["pay_" + k] * hit, hit))
    A = np.array(out, float); cost = A[:, 0].sum() * 100
    if cost == 0:
        return None
    d = dict(pts=int(A[:, 0].sum()), races=int((A[:, 0] > 0).sum()), hit=int(A[:, 2].sum()), roi=A[:, 1].sum() / cost,
             avg=A[:, 0].sum() / max(1, (A[:, 0] > 0).sum()), mx=int(A[:, 0].max()))
    if boot:
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(A), len(A)); cc = A[i, 0].sum()
            bs.append(A[i, 1].sum() / (cc * 100) if cc else np.nan)
        d["lo"], d["hi"] = np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)
    return d


def fmt(d):
    if not d:
        return "買い目なし"
    s = f"{d['races']}R・{d['pts']}点（平均{d['avg']:.1f}点・最大{d['mx']}点） 的中{d['hit']} 回収率{d['roi']:.1%}"
    return s + (f"（95%: {d['lo']:.0%}〜{d['hi']:.0%}）" if "lo" in d else "")


GRID = {
    ("f", "ev"): [dict(th=t, pmin=p, omin=om) for t in (0.9, 1.0, 1.1, 1.2) for p in (0.003, 0.005, 0.01, 0.02) for om in (0, 10, 30)],
    ("t", "ev"): [dict(th=t, pmin=p, cap=c, omin=om) for t in (1.0, 1.1, 1.2, 1.3, 1.5) for p in (0.001, 0.002, 0.005) for c in (12, 24, 48) for om in (0, 30, 100)],
    ("t", "are"): [dict(man=m, step=s) for m in (0.6, 0.7, 0.8, 0.85) for s in (0.005, 0.01)],
    ("t", "f2t"): [dict(th=t) for t in (0.8, 0.9, 1.0, 1.1, 1.2)],
}
MINHIT = {"f": 40, "t": 20}
if __name__ == "__main__":
    final = {}
    for (k, rule), grid in GRID.items():
        name = {"f": "3連複", "t": "3連単"}[k]
        res = [(a, run(VA, k, rule, a)) for a in grid]
        ok = [(a, d) for a, d in res if d and d["hit"] >= MINHIT[k]]
        ok.sort(key=lambda t: -t[1]["roi"])
        print(f"===== {name} {rule}：基準決め期間の上位（的中{MINHIT[k]}回以上） / 条件を満たす設定 {len(ok)}/{len(grid)}")
        for a, d in ok[:6]:
            print("  ", a, fmt(d))
        if ok:
            final[(k, rule)] = [a for a, _ in ok[:5]]
    pickle.dump(final, open("/tmp/chosen.pkl", "wb"))
