"""③ 3連単：人気馬が1着の並びの買われすぎを、着順ごとの補正で直せるか。
今の形（実オッズ人気・AIのPL・単勝人気PL）に、
  a) 1着に置いた馬の単勝人気の確率（対数）
  b) 2着に置いた馬の単勝人気の確率（対数）
  c) 1着が1番人気かどうか
を足して学習期間で重みを決め、基準決め期間の当てはまり(NLL)と回収率で判断。採点期間は1回だけ。"""
import pickle, sys, numpy as np
sys.path.insert(0, "/home/claude/keiyomi/model")
import exotic as X
from scipy.optimize import minimize

D = pickle.load(open("/tmp/five.pkl", "rb")); rows = D["rows"]; BW = D["BW"]
TRr = [r for r in rows if r["date"] <= "20250531"]
VA = [r for r in rows if "20250601" <= r["date"] <= "20260531"]
TE = [r for r in rows if r["date"] >= "20260601"]


def feats(r):
    n = r["n"]; P, _, M = X.perms(n)
    o = r["o_t"]; ok = o > 0; q = np.where(ok, 1 / np.where(ok, o, 1), 0); q /= q.sum()
    pm = np.exp(r["lm_t"].astype(np.float64))
    win = np.bincount(P[:, 0], weights=pm, minlength=n)      # 単勝人気から出した各馬の勝つ確率
    fav = np.argmax(win)
    of = r["o_f"]; okf = of > 0; qf = np.where(okf, 1 / np.where(okf, of, 1), 0); qf /= max(qf.sum(), 1e-12)
    pf = np.exp(r["lp_f"].astype(np.float64))
    trio_p = X.blend([np.log(qf + 1e-9), np.log(pf + 1e-12), np.log(np.exp(r["lm_f"].astype(np.float64)) + 1e-12)], BW["f"], okf)
    Xm = np.stack([np.log(q + 1e-9), r["lp_t"], r["lm_t"], np.log(win[P[:, 0]] + 1e-9), np.log(win[P[:, 1]] + 1e-9),
                   (P[:, 0] == fav).astype(float)], 1)
    return Xm, ok, trio_p[M], okf[M] & (of[M] >= 10), of[M]


for r in TRr + VA + TE:
    r["Xp"] = feats(r)


def nll(w, Dd, cols):
    L = 0; G = np.zeros(len(cols))
    for r in Dd:
        Xm, ok = r["Xp"][0][:, cols], r["Xp"][1]
        if not ok[r["wt"]]:
            continue
        s = np.where(ok, Xm @ w, -1e9); m = s.max(); e = np.exp(s - m) * ok; Z = e.sum(); p = e / Z
        L += s[r["wt"]] - m - np.log(Z); G += Xm[r["wt"]] - (p[:, None] * Xm).sum(0)
    return -L, -G


def probs(r, w, cols):
    Xm, ok = r["Xp"][0][:, cols], r["Xp"][1]
    s = np.where(ok, Xm @ w, -1e9); e = np.exp(s - s.max()) * ok
    return e / e.sum(), ok


def roi(R, w, cols, th=1.0, cap=48, from_trio=True):
    pts = 0; ret = []; races = 0
    for r in R:
        p, ok = probs(r, w, cols); o = r["o_t"]; ev = p * o
        c = ok & (ev >= th)
        if from_trio:   # 本番と同じ：3連複で選んだ3頭（期待値≧1・確率≧1%・オッズ≧10）の並びだけ
            tp, tok, of = r["Xp"][2], r["Xp"][3], r["Xp"][4]
            c &= tok & (tp >= 0.01) & (tp * of >= 1.0)
        i = np.where(c)[0]; i = i[np.argsort(-ev[i])][:cap]
        if len(i):
            races += 1
        for j in i:
            ret.append(r["pay_t"] if j == r["wt"] else 0)
    r_ = np.array(ret); cost = len(r_) * 100
    if not len(r_):
        return "なし"
    s = np.sort(r_)[::-1]
    return f"{len(r_)}点({races}R) 的中{(r_>0).sum()} 回収率{r_.sum()/cost:.1%} 上位3つ除く{(r_.sum()-s[:3].sum())/cost:.1%}"


res = {}
for nm, cols in [("今の形", [0, 1, 2]), ("+1着の単勝人気", [0, 1, 2, 3]), ("+1着・2着の単勝人気", [0, 1, 2, 3, 4]),
                 ("+1着・2着+1番人気1着", [0, 1, 2, 3, 4, 5])]:
    w = minimize(nll, np.r_[BW["t"], np.zeros(len(cols) - 3)], args=(TRr, cols), jac=True, method="L-BFGS-B").x
    v = nll(w, VA, cols)[0] / len(VA); res[nm] = (w, cols, v)
    print(nm, np.round(w, 3), "基準決めNLL", round(v, 4), "| 基準決め", roi(VA, w, cols), "| 単独48点", roi(VA, w, cols, from_trio=False), flush=True)

best = min(res, key=lambda k: res[k][2])
print("→ 基準決めNLLが最も良い:", best)
for nm in ("今の形", best):
    w, cols, _ = res[nm]
    print("採点（1回だけ）", nm, roi(TE, w, cols), "| 単独48点", roi(TE, w, cols, from_trio=False))
pickle.dump(res, open("/tmp/tri_position.pkl", "wb"))
