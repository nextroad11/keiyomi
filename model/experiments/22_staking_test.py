"""採点期間に1回だけ当てる。基準決め期間で選んだ：ケリー×0.25、勝負日＝その日の見込み利益6,000円以上で上限100,000円。"""
exec(open("/home/claude/keiyomi/model/experiments/21_staking.py").read().split('if __name__ == "__main__":')[0])
for name, kw in [("A 均等100円", {}), ("B ケリー×0.25", dict(kfrac=0.25)), ("C B＋勝負日", dict(kfrac=0.25, big=6000)), ("参考 ケリー×0.1", dict(kfrac=0.1))]:
    print(name, "採点:", f(simulate(TE, **kw)))
# 高配当頼みの度合い：Cで払戻の大きい順上位3レースを除いた回収率
import numpy as np
def by_race(R, **kw):
    out = []
    days = {}
    for d in R: days.setdefault(d["date"], []).append(d)
    for date in sorted(days):
        r = simulate(days[date], **kw); out.append((date, r["cost"], r["ret"]))
    return out
for per, R in (("基準決め", VA), ("採点", TE)):
    rows = by_race(R, kfrac=0.25, big=6000); c = sum(x[1] for x in rows); s = sorted((x[2] for x in rows), reverse=True)
    print(per, f"C: 払戻の多い日 上位3日を除く回収率 {(sum(s)-sum(s[:3]))/c:.1%}（上位3日の払戻 {[f'{v:,.0f}' for v in s[:3]]}）")
