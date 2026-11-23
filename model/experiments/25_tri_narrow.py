"""3連単を「3連複で選んだ3頭の並び」から、期待値の高い順に少ない点数（3〜10点）へ絞る。
基準決め期間（2025-06〜2026-05）で設定を選び（的中10回以上、上位3つを除いた回収率で比べる）、採点期間（2026-06〜）は1回だけ。"""
import sys, pickle, numpy as np
src = open("/home/claude/keiyomi/model/experiments/24_tri_position.py").read()
exec(src.split("res = {}")[0])
w = BW["t"]; cols = [0, 1, 2]


def run(R, th, cap, pmin=0.0, key="ev"):
    ret = []; races = 0
    for r in R:
        p, ok = probs(r, w, cols); o = r["o_t"]; ev = p * o
        tp, tok, of = r["Xp"][2], r["Xp"][3], r["Xp"][4]
        c = ok & (ev >= th) & (p >= pmin) & tok & (tp >= 0.01) & (tp * of >= 1.0)
        i = np.where(c)[0]; i = i[np.argsort(-(ev if key == "ev" else p)[i])][:cap]
        if len(i): races += 1
        ret += [r["pay_t"] if j == r["wt"] else 0 for j in i]
    r_ = np.array(ret)
    if not len(r_): return None
    s = np.sort(r_)[::-1]; cost = len(r_) * 100
    return dict(pts=len(r_), races=races, hit=int((r_ > 0).sum()), roi=r_.sum() / cost, ex3=(r_.sum() - s[:3].sum()) / cost)


f = lambda d: "なし" if not d else f"{d['pts']}点({d['races']}R) 的中{d['hit']} 回収率{d['roi']:.1%} 上位3つ除く{d['ex3']:.1%}"
print("今の形（48点まで）基準決め", f(run(VA, 1.0, 48)))
grid = []
for key in ("ev", "p"):
  for th in (1.0, 1.2):
    for cap in (3, 5, 8, 10):
        for pm in (0.0, 0.003, 0.005):
            r = run(VA, th, cap, pm, key)
            if r: grid.append(((th, cap, pm, key), r)); print(key, th, cap, pm, f(r), flush=True)
ok = [g for g in grid if g[1]["hit"] >= 10]
best = max(ok, key=lambda g: g[1]["ex3"])
print("→ 選んだ設定（期待値, 点数, 確率の下限）", best[0], f(best[1]))
print("採点（1回だけ） 選んだ設定:", f(run(TE, *best[0])))
print("採点（1回だけ） 今の形:", f(run(TE, 1.0, 48)))
pickle.dump(best[0], open("/tmp/tri_narrow.pkl", "wb"))
