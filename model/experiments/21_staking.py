"""賭け方（金額配分）の検証。ルール（ユーザー指定）：1レース30,000円まで、1日50,000円まで、勝負日は1日100,000円まで。
3連複は点数をAIに任せる。3連単は普段48点まで、大荒れと見たレースは48点を超えてよい。1点は100円単位。
比べるもの：
  A 均等：選んだ組を各100円（これまでの検証と同じ）
  B 金額配分：各組の金額を「ケリー基準の一部」で決める（当たる確率とオッズから）。レース・1日の上限で縮める
  C B＋勝負日：その日の見込み利益が一定を超えたら1日の上限を100,000円に上げる
  D B＋3連単の大荒れ拡張：AIの万馬券確率が高く、人気より荒れると見たレースは3連単を96点まで
レースは発走順に処理し、その時点までの情報だけで決める（あとのレースは見ない）。
設定は基準決め期間（2025-06〜2026-05）だけで選び、採点期間（2026-06〜11）に1回だけ当てる。"""
import pickle, sys, numpy as np
sys.argv = ["x"]
exec(open("/home/claude/keiyomi/model/experiments/18_all_types_rules.py").read().split("GRID = {")[0])
CH = pickle.load(open("/tmp/alltypes_chosen.pkl", "rb"))
POST = {r["id"]: r.get("post") or "99:99" for r in pickle.load(open("/tmp/races5.pkl", "rb"))}
RACE_CAP, DAY_CAP, BIG_CAP = 30000, 50000, 100000


def race_bets(d, kfrac, bank, expand):
    """1レースの買い目 [(券種, 当たる確率, オッズ, 払戻/100円, 金額)]（レース上限まで縮めた後）"""
    out = []
    for t in ("3連複", "3連単"):
        x = d["types"].get(t)
        if not x or t not in CH:
            continue
        a = dict(CH[t])
        c = np.where((x["ev"] >= a["th"]) & (x["p"] >= a["pmin"]) & (x["o"] >= a["omin"]))[0]
        c = c[np.argsort(-x["ev"][c])]
        cap = 48
        if t == "3連単" and expand:
            hi = x["o"] >= 100
            man_ai, man_mk = float(x["p"][hi].sum()), float(np.exp(x["lq"][hi].astype(np.float64)).sum())
            if man_ai >= expand["man"] and man_ai - man_mk >= expand["gap"]:
                cap = 96
        if t == "3連単":
            c = c[:cap]
        for i in c:
            p, o = float(x["p"][i]), float(x["o"][i])
            if kfrac is None:
                st = 100
            else:
                f = max(0.0, (p * o - 1) / (o - 1)) * kfrac
                st = max(100, int(round(bank * f / 100)) * 100)
            out.append([t, p, o, float(x["ret"][i]), st])
    tot = sum(b[4] for b in out)
    if tot > RACE_CAP:
        s = RACE_CAP / tot
        for b in out:
            b[4] = max(100, int(b[4] * s / 100) * 100)
        while sum(b[4] for b in out) > RACE_CAP:  # 端数で超えたら小さい期待値から削る
            out.sort(key=lambda b: b[1] * b[2]); out.pop(0)
    return out


def simulate(R, kfrac=None, bank=1_000_000, big=None, expand=None):
    days = {}
    for d in R:
        days.setdefault(d["date"], []).append(d)
    cost = ret = 0; curve = []; hits = 0; nbet = 0; bigdays = 0; daily = []
    for date in sorted(days):
        cap, used, ev_sum = DAY_CAP, 0, 0.0
        dc = dr = 0
        for d in sorted(days[date], key=lambda d: POST.get(d["rid"], "99")):
            bets = race_bets(d, kfrac, bank, expand)
            if not bets:
                continue
            if big and ev_sum >= big and cap == DAY_CAP:
                cap = BIG_CAP; bigdays += 1
            room = cap - used
            tot = sum(b[4] for b in bets)
            if room < 100:
                break
            if tot > room:
                s = room / tot
                for b in bets:
                    b[4] = int(b[4] * s / 100) * 100
                bets = [b for b in bets if b[4] >= 100]
            for t, p, o, pay, st in bets:
                dc += st; dr += pay * st / 100; nbet += 1; hits += pay > 0
                ev_sum += st * (p * o - 1)
            used += sum(b[4] for b in bets)
        cost += dc; ret += dr; daily.append((date, dc, dr)); curve.append(ret - cost)
    curve = np.array(curve) if curve else np.zeros(1)
    dd = float((np.maximum.accumulate(np.r_[0, curve]) - np.r_[0, curve]).max())
    worst_day = min((r - c for _, c, r in daily), default=0)
    return dict(cost=cost, ret=ret, roi=ret / cost if cost else None, profit=ret - cost, dd=dd, days=len(daily),
                bets=nbet, hits=int(hits), bigdays=bigdays, worst=worst_day, avg_day=cost / max(1, len(daily)))


def f(r):
    return (f"投資{r['cost']:,.0f}円 払戻{r['ret']:,.0f}円 回収率{r['roi']:.1%} 収支{r['profit']:+,.0f}円 "
            f"最大の落ち込み{r['dd']:,.0f}円 最悪の1日{r['worst']:+,.0f}円 1日平均{r['avg_day']:,.0f}円 的中{r['hits']}/{r['bets']}点"
            + (f" 勝負日{r['bigdays']}日" if r["bigdays"] else ""))


if __name__ == "__main__":
    print("A 均等100円   基準決め:", f(simulate(VA)))
    grid = []
    for k in (0.05, 0.1, 0.25):
        r = simulate(VA, kfrac=k); grid.append((k, r)); print(f"B ケリー×{k}  基準決め:", f(r), flush=True)
    kbest = max(grid, key=lambda z: z[1]["profit"] - 0.5 * z[1]["dd"])[0]
    print("→ 選んだ割合", kbest)
    bg = []
    for big in (3000, 6000, 10000):
        r = simulate(VA, kfrac=kbest, big=big); bg.append((big, r)); print(f"C 勝負日（見込み利益{big}円以上） 基準決め:", f(r), flush=True)
    eg = []
    for man, gap in ((0.8, 0.05), (0.85, 0.08), (0.9, 0.1)):
        r = simulate(VA, kfrac=kbest, expand=dict(man=man, gap=gap)); eg.append(((man, gap), r)); print(f"D 大荒れ拡張 {man},{gap} 基準決め:", f(r), flush=True)
    pickle.dump(dict(kbest=kbest, bg=bg, eg=eg), open("/tmp/staking_valid.pkl", "wb"))
