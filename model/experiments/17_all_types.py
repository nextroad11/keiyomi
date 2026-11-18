"""全券種（単勝・複勝・枠連・馬連・ワイド・馬単・3連複・3連単）で、AIの確率×実際のオッズの検証。
学習 〜2025-05（組み合わせの重み）／基準決め 2025-06〜2026-05（買い方の設定を選ぶ）／採点 2026-06〜11（1回だけ）。
AIの確率：勝ち馬モデルの強さ（12_five_years.py で保存した3連単の PL）から各券種の確率を足し上げる。
人気の確率：各券種の実際のオッズから。ワイドは下限オッズで期待値を出す（控えめ）、払戻は実際の結果。
"""
import pickle, itertools, numpy as np
from scipy.optimize import minimize

D = pickle.load(open("/tmp/five.pkl", "rb")); rows = D["rows"]; del D["rows"]
PR = pickle.load(open("/tmp/pairs5.pkl", "rb"))
RACES = {r["id"]: r for r in pickle.load(open("/tmp/races5.pkl", "rb"))}
EX = pickle.load(open("/tmp/ex5.pkl", "rb"))
TYPES = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]
PERM, COMB = {}, {}


def perms(n):
    if n not in PERM:
        PERM[n] = np.array(list(itertools.permutations(range(n), 3))); COMB[n] = np.array(list(itertools.combinations(range(n), 3)))
    return PERM[n], COMB[n]


def tables(pt, n, waku):
    """3連単の確率 pt から全券種の確率（dict: key -> p）"""
    P, C = perms(n)
    win = np.bincount(P[:, 0], pt, n)
    top3 = np.bincount(P[:, 0], pt, n) + np.bincount(P[:, 1], pt, n) + np.bincount(P[:, 2], pt, n)
    ex = np.zeros((n, n)); np.add.at(ex, (P[:, 0], P[:, 1]), pt)
    wd = np.zeros((n, n))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = np.minimum(P[:, i], P[:, j]), np.maximum(P[:, i], P[:, j]); np.add.at(wd, (a, b), pt)
    out = {"単勝": {(i,): win[i] for i in range(n)}, "複勝": {(i,): top3[i] for i in range(n)},
           "馬単": {(i, j): ex[i, j] for i in range(n) for j in range(n) if i != j},
           "馬連": {(i, j): ex[i, j] + ex[j, i] for i in range(n) for j in range(i + 1, n)},
           "ワイド": {(i, j): wd[i, j] for i in range(n) for j in range(i + 1, n)}}
    wk = {}
    for (i, j), p in out["馬連"].items():
        k = tuple(sorted((waku[i], waku[j]))); wk[k] = wk.get(k, 0) + p
    out["枠連"] = wk
    return out


def build():
    data = []
    for r in rows:
        rid = r["rid"]; race = RACES[rid]
        if rid not in PR or not race.get("odds_final"):
            continue
        nos = EX[rid][0]; n = len(nos)
        runners = {x["no"]: x for x in race["runners"]}
        if any(no not in runners for no in nos):
            continue
        waku = [runners[no].get("waku") for no in nos]
        if None in waku:
            continue
        pt = np.exp(r["lp_t"].astype(np.float64)); pmt = np.exp(r["lm_t"].astype(np.float64))
        A, M = tables(pt, n, waku), tables(pmt, n, waku)
        pr = PR[rid]; of = race["odds_final"]
        odds = {"単勝": {(i,): (of.get(str(no)) or {}).get("win") for i, no in enumerate(nos)},
                "複勝": {(i,): (of.get(str(no)) or {}).get("pmin") for i, no in enumerate(nos)},
                "馬連": {(i, j): pr.get("umaren", {}).get((nos[i], nos[j])) for (i, j) in A["馬連"]},
                "ワイド": {(i, j): (pr.get("wide", {}).get((nos[i], nos[j])) or (None,))[0] for (i, j) in A["ワイド"]},
                "馬単": {(i, j): pr.get("umatan", {}).get((nos[i], nos[j])) for (i, j) in A["馬単"]},
                "枠連": {k: pr.get("waku", {}).get(k) for k in A["枠連"]}}
        pay = race["pay"]
        def win_keys(t):
            ks = set()
            for x in pay.get(t, []):
                v = [int(s) for s in x["combo"].split("-")]
                if t == "枠連":
                    ks.add((tuple(sorted(v)), x["yen"]))
                else:
                    idx = [nos.index(a) for a in v if a in nos]
                    if len(idx) != len(v):
                        continue
                    k = tuple(idx) if t in ("馬単", "単勝", "複勝") else tuple(sorted(idx))
                    ks.add((k, x["yen"]))
            return {k: y for k, y in ks}
        rec = {"rid": rid, "date": r["date"], "types": {}}
        for t in ("単勝", "複勝", "枠連", "馬連", "ワイド", "馬単"):
            keys = [k for k in A[t] if odds[t].get(k)]
            if len(keys) < 2:
                continue
            o = np.array([odds[t][k] for k in keys]); q = 1 / o
            # 複勝・ワイドは1レースに当たりが複数あるので、人気の確率は合計で割らず「1/オッズ×払戻率」の形のまま使う
            rec["types"][t] = {"keys": keys, "o": o, "lq": np.log(q / q.sum()), "la": np.log(np.array([A[t][k] for k in keys]) + 1e-12),
                               "lm": np.log(np.array([M[t][k] for k in keys]) + 1e-12), "pay": win_keys(t)}
        for t, ok_key, pk, lpk, lmk in (("3連複", "o_f", "f", "lp_f", "lm_f"), ("3連単", "o_t", "t", "lp_t", "lm_t")):
            o = r[ok_key].astype(np.float64); okm = o > 0
            q = np.where(okm, 1 / np.where(okm, o, 1), 0)
            sel = np.where(okm)[0].astype(np.int32)
            # 3連は組が多いので、組は通し番号のまま持つ（keys＝番号の配列）
            rec["types"][t] = {"keys": sel, "o": o[sel].astype(np.float32), "lq": np.log(q[sel] / q.sum()).astype(np.float32),
                               "la": r[lpk][sel].astype(np.float32), "lm": r[lmk][sel].astype(np.float32), "pay": {int(r["w" + pk]): r["pay_" + pk]}}
        for key in ("lp_t", "lm_t", "o_t", "lp_f", "lm_f", "o_f"):
            r[key] = None  # 使い終わった配列を手放す
        data.append(rec)
    return data


def fit(train, t):
    multi = t in ("複勝", "ワイド")
    def nll(w):
        L = 0; G = np.zeros(3)
        for rec in train:
            x = rec["types"].get(t)
            if not x:
                continue
            X = np.stack([x["lq"], x["la"], x["lm"]], 1).astype(np.float64); s = X @ w
            if multi:  # 当たりが3つ：各組が当たるかどうか（ロジスティック）
                y = np.array([1.0 if k in x["pay"] else 0.0 for k in x["keys"]])
                p = 1 / (1 + np.exp(-(s + w_b[0])))
                L += (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).sum(); G += ((y - p)[:, None] * X).sum(0)
            else:
                win = [i for i, k in enumerate(x["keys"].tolist() if hasattr(x["keys"], "tolist") else x["keys"]) if k in x["pay"]]
                if not win:
                    continue
                m = s.max(); e = np.exp(s - m); Z = e.sum(); p = e / Z
                L += s[win[0]] - m - np.log(Z); G += X[win[0]] - (p[:, None] * X).sum(0)
        return -L, -G
    w_b = [np.log(3) if t == "複勝" else np.log(3)]  # 当たり3つ分のずらし（下で一緒に推定）
    if multi:
        def nll2(wb):
            w_b[0] = wb[3]; v, g = nll(wb[:3])
            # 切片の勾配
            gb = 0
            for rec in train:
                x = rec["types"].get(t)
                if not x:
                    continue
                X = np.stack([x["lq"], x["la"], x["lm"]], 1); y = np.array([1.0 if k in x["pay"] else 0.0 for k in x["keys"]])
                p = 1 / (1 + np.exp(-(X @ wb[:3] + wb[3]))); gb += (y - p).sum()
            return v, np.r_[g, -gb]
        r = minimize(nll2, np.array([0.9, 0.3, -0.2, 1.0]), jac=True, method="L-BFGS-B").x
        return r
    return minimize(nll, np.array([0.9, 0.3, -0.2]), jac=True, method="L-BFGS-B").x


def probs(x, w, t):
    s = np.stack([x["lq"], x["la"], x["lm"]], 1).astype(np.float64) @ w[:3]
    if t in ("複勝", "ワイド"):
        return 1 / (1 + np.exp(-(s + w[3])))
    e = np.exp(s - s.max()); return e / e.sum()


if __name__ == "__main__":
    data = build()
    rows.clear()
    pickle.dump(data, open("/tmp/alltypes.pkl", "wb"), protocol=4)
    TR = [d for d in data if d["date"] <= "20250531"]
    print("レース", len(data), "学習", len(TR))
    W = {}
    for t in TYPES:
        W[t] = fit(TR, t); print(t, np.round(W[t], 3), flush=True)
    pickle.dump(W, open("/tmp/alltypes_w.pkl", "wb"))
