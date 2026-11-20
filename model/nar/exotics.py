"""地方：全券種で実オッズ＋AIの組み合わせ（中央の第5回と同じ形）。
前半 2026-02〜05：組み合わせの重み／前半 06〜07：買い方の設定を選ぶ／後半 08〜11：採点（1回だけ）。"""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/keiyomi/model")
import exotic as X
from scipy.optimize import minimize
src = open("/home/claude/keiyomi/model/experiments/17_all_types.py").read()
exec(src.split("D = pickle.load")[0].split('"""', 2)[2])           # import 等
perms = lambda n: X.perms(n)[:2]
exec("def tables" + src.split("def tables")[1].split("def build")[0])
exec("def fit" + src.split("def fit")[1].split("def probs")[0])
exec("def probs" + src.split("def probs")[1].split('if __name__')[0])

od = pd.read_pickle("/tmp/nar_od.pkl"); W = pickle.load(open("/tmp/nar_w.pkl", "rb"))
O = pickle.load(open("/tmp/nar_odds.pkl", "rb")); PAY = pickle.load(open("/tmp/nar_pay.pkl", "rb"))
KEYS = {"単勝": "win", "複勝": "place", "枠連": "wakuren", "馬連": "umaren", "ワイド": "wide", "馬単": "umatan"}
data = []
for rid, g in od.groupby("rid", sort=False):
    o = O[rid]; nos = o["nos"]; n = len(nos)
    if n < 5 or rid not in PAY or "tri" not in o or "trio" not in o:
        continue
    g = g.set_index(g.no.astype(int))
    if any(no not in g.index for no in nos):
        continue   # 取消などでずれるレースは除く
    lq = g.loc[nos, "lq"].to_numpy(); ai = g.loc[nos, "ai"].to_numpy(); waku = g.loc[nos, "waku"].astype(int).tolist()
    s_ai = np.exp(W["W2"][0] * lq + W["W2"][1] * ai); s_mk = np.exp(W["WM"][0] * lq)
    pt, pmt = X.pl_tri(s_ai), X.pl_tri(s_mk)
    A, M = tables(pt, n, waku), tables(pmt, n, waku)
    pay = PAY[rid]; idx = {v: i for i, v in enumerate(nos)}
    rec = {"rid": rid, "date": rid[:8], "types": {}}
    for t, k in KEYS.items():
        if t == "単勝":
            odds = {(i,): o["win"][i] for i in range(n)}
        else:
            src_ = o.get(k, {})
            if t == "枠連":
                odds = {kk: src_.get(kk) for kk in A[t]}
            else:
                odds = {kk: src_.get(tuple(nos[i] for i in kk)) for kk in A[t]}
        keys = [kk for kk in A[t] if odds.get(kk)]
        if len(keys) < 2:
            continue
        oo = np.array([odds[kk] for kk in keys], float); q = 1 / oo
        win = {}
        for combo, yen in pay.get(k, []):
            if t == "枠連":
                win[tuple(sorted(combo))] = yen
            elif all(c in idx for c in combo):
                ii = [idx[c] for c in combo]
                win[tuple(ii) if t in ("単勝", "複勝", "馬単") else tuple(sorted(ii))] = yen
        rec["types"][t] = {"keys": keys, "o": oo, "lq": np.log(q / q.sum()), "la": np.log(np.array([A[t][kk] for kk in keys]) + 1e-12),
                           "lm": np.log(np.array([M[t][kk] for kk in keys]) + 1e-12), "pay": win}
    P, C, MAP = X.perms(n)
    pf, pmf = X.trio_from_tri(pt, n), X.trio_from_tri(pmt, n)
    for t, oarr, pa, pm, kind in (("3連複", o["trio"], pf, pmf, "trio"), ("3連単", o["tri"], pt, pmt, "tri")):
        okm = oarr > 0; sel = np.where(okm)[0].astype(np.int32); q = np.where(okm, 1 / np.where(okm, oarr, 1), 0)
        win = {}
        for combo, yen in pay.get(kind, []):
            if all(c in idx for c in combo):
                ii = [idx[c] for c in combo]
                if kind == "trio":
                    j = np.where((C == sorted(ii)).all(1))[0]
                else:
                    j = np.where((P == ii).all(1))[0]
                if len(j):
                    win[int(j[0])] = yen
        rec["types"][t] = {"keys": sel, "o": oarr[sel].astype(np.float32), "lq": np.log(q[sel] / q.sum()).astype(np.float32),
                           "la": np.log(pa[sel] + 1e-12).astype(np.float32), "lm": np.log(pm[sel] + 1e-12).astype(np.float32), "pay": win}
    data.append(rec)
TYPES = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]
print("レース", len(data), flush=True)
pickle.dump(data, open("/tmp/nar_alltypes.pkl", "wb"), protocol=4)
TR = [d for d in data if d["date"] <= "20260531"]
Wt = {}
for t in TYPES:
    Wt[t] = fit(TR, t); print(t, np.round(Wt[t], 3), flush=True)
pickle.dump(Wt, open("/tmp/nar_alltypes_w.pkl", "wb"))
