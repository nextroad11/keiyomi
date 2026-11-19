"""① 3連複の高配当頼みを減らす。基準決め期間で「上位3つの払戻を除いた回収率」が高い設定を選び、採点期間に1回だけ当てる。
条件：期待値の下限・確率の下限・オッズの上限（大穴を切る）・1レースの点数の上限。"""
import sys, numpy as np
sys.argv = ["x"]
exec(open("/home/claude/keiyomi/model/experiments/18_all_types_rules.py").read().split("GRID = {")[0])


def run2(R, a):
    out = []
    for d in R:
        x = d["types"].get("3連複")
        if not x:
            continue
        c = np.where((x["ev"] >= a["th"]) & (x["p"] >= a["pmin"]) & (x["o"] >= a["omin"]) & (x["o"] <= a["omax"]))[0]
        c = c[np.argsort(-x["ev"][c])][: a["cap"]]
        for i in c:
            out.append(x["ret"][i])
    if not out:
        return None
    r = np.array(out); cost = len(r) * 100; s = np.sort(r)[::-1]
    return dict(pts=len(r), hit=int((r > 0).sum()), roi=r.sum() / cost, ex3=(r.sum() - s[:3].sum()) / cost, ex1=(r.sum() - s[:1].sum()) / cost)


def f2(d):
    return "なし" if not d else f"{d['pts']}点 的中{d['hit']} 回収率{d['roi']:.1%} 最大1つ除く{d['ex1']:.1%} 上位3つ除く{d['ex3']:.1%}"


base = dict(th=1.1, pmin=0.005, omin=10, omax=1e9, cap=999)  # いまの本番（第5回で選んだもの）
base.update({k: v for k, v in CH["3連複"].items()}) if "CH" in dir() else None
import pickle
CH = pickle.load(open("/tmp/alltypes_chosen.pkl", "rb")); base = dict(CH["3連複"], omax=1e9, cap=999)
print("いまの本番", base); print("  基準決め", f2(run2(VA, base)));
grid = []
for th in (1.0, 1.1, 1.2):
    for pm in (0.005, 0.01, 0.02, 0.03):
        for om in (5, 10):
            for ox in (100, 200, 400, 1e9):
                for cap in (3, 6, 999):
                    a = dict(th=th, pmin=pm, omin=om, omax=ox, cap=cap); r = run2(VA, a)
                    if r and r["hit"] >= 40:
                        grid.append((a, r))
grid.sort(key=lambda z: -z[1]["ex3"])
print("基準決め期間で『上位3つ除く回収率』が高い設定（的中40回以上）")
for a, r in grid[:6]:
    print(" ", a, f2(r))
best = grid[0][0]
print("→ 選んだ設定", best)
print("採点（1回だけ） 選んだ設定:", f2(run2(TE, best)))
print("採点（1回だけ） いまの本番:", f2(run2(TE, base)))
pickle.dump(best, open("/tmp/trio_robust.pkl", "wb"))

# 本当の本番ルール（prod.RULES：期待値≧1.0・確率≧1%・オッズ≧10・上限なし）でも並べる
real = dict(th=1.0, pmin=0.01, omin=10, omax=1e9, cap=999)
print("本当の本番 基準決め", f2(run2(VA, real)))
print("本当の本番 採点    ", f2(run2(TE, real)))
for cap in (3, 6, 10):
    a = dict(real, cap=cap); print(f"本番＋1レース{cap}点まで 基準決め", f2(run2(VA, a)), "| 採点", f2(run2(TE, a)))
