"""第9回：10年分（2016-12〜）で学び直す。
A: 今の学習期間（2022-03〜2025-05） / B: 2017-06〜2025-05 に延ばす。
基準決め 2025-06〜2026-05、採点 2026-06〜（1回だけ）。今の本番の買い方（prod.RULES）で、100円均等の成績を比べる。"""
import sys, json, gzip, pathlib, pickle, base64, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/keiyomi/model"); sys.path.insert(0, "/home/claude/keiyomi/scraper")
import features as F, pl, exotic as X, prod
from crypt import unseal
from scipy.optimize import minimize
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
priv = X25519PrivateKey.from_private_bytes(bytes.fromhex(pathlib.Path("~/.keiyomi/probe_priv.hex").expanduser().read_text().strip()))


def op(p):
    raw = unseal(priv, json.loads(pathlib.Path(p).read_text()))
    if raw[:2] == b"\x1f\x8b": raw = gzip.decompress(raw)
    return json.loads(raw)


C = pathlib.Path("/tmp/ten")
if not (C / "races.pkl").exists():
    races = []
    for f in sorted(pathlib.Path("/tmp/kd4/dev/results").glob("*.json")):
        if f.stem <= "20261123": races += list(op(f)["races"].values())
    pickle.dump(races, open(C / "races.pkl", "wb"))
    ex = {}
    for f in sorted(pathlib.Path("/tmp/ko4/dev/exotic").glob("*.json")):
        ex.update(op(f)["races"])
    pickle.dump(ex, open(C / "ex.pkl", "wb"))
races = pickle.load(open(C / "races.pkl", "rb")); EX = pickle.load(open(C / "ex.pkl", "rb"))
print("結果", len(races), "3連オッズ", len(EX), flush=True)
if not (C / "df.pkl").exists():
    df = F.runners_frame(races); df = F.add_figures(df, "20250531"); df = F.add_history(df); df = F.finalize(df)
    df = df[(df.surface != "障害") & df.pos.notna() & df.odds.notna()].copy().reset_index(drop=True)
    inv = 1 / df.odds; df["p_mkt"] = inv / inv.groupby(df.rid).transform("sum"); df["logp_mkt"] = np.log(df.p_mkt)
    df.to_pickle(C / "df.pkl")
df = pd.read_pickle(C / "df.pkl"); print("出走", len(df), df.date.min(), df.date.max(), flush=True)
byid = {r["id"]: r for r in races}
VA, TE = ("20250601", "20260531"), ("20260601", "20261131")
G = {rid: g for rid, g in df.groupby("rid", sort=False)}


def arrays(rid, W3, WM):
    g = G.get(rid); race = byid.get(rid); v = EX.get(rid)
    if g is None or race is None or v is None or "3連単" not in race["pay"] or len(race["pay"]["3連単"]) > 1 or len(race["pay"].get("3連複", [])) > 1:
        return None
    nos = v["nos"]; n = len(nos)
    tri = np.frombuffer(base64.b64decode(v["tri"]), dtype="<u4") / 10; trio = np.frombuffer(base64.b64decode(v["trio"]), dtype="<u4") / 10
    if len(tri) != n * (n - 1) * (n - 2) or set(g["no"]) - set(nos):
        return None
    pos = {no: i for i, no in enumerate(nos)}
    sa, sm = prod.strengths(g, W3, WM); s_ai, s_mk = np.full(n, 1e-6), np.full(n, 1e-6)
    for i, no in enumerate(g["no"]):
        s_ai[pos[no]], s_mk[pos[no]] = sa[i], sm[i]
    P, Cc, M = X.perms(n)
    w3 = [pos[int(x)] for x in race["pay"]["3連単"][0]["combo"].split("-")]
    wt = int(np.where((P == w3).all(1))[0][0]); wf = int(np.where((Cc == sorted(w3)).all(1))[0][0])
    return dict(n=n, tri=tri, trio=trio, s_ai=s_ai, s_mk=s_mk, wt=wt, wf=wf,
                pay_t=race["pay"]["3連単"][0]["yen"], pay_f=race["pay"]["3連複"][0]["yen"])


def fit_bw(rids, W3, WM):
    data = {"t": [], "f": []}
    for rid in rids:
        a = arrays(rid, W3, WM)
        if not a: continue
        n = a["n"]; pt, pmt = X.pl_tri(a["s_ai"]), X.pl_tri(a["s_mk"])
        for k, o, p, m, wi in (("t", a["tri"], pt, pmt, a["wt"]), ("f", a["trio"], X.trio_from_tri(pt, n), X.trio_from_tri(pmt, n), a["wf"])):
            q, ok = X.market_prob(o)
            if ok[wi]: data[k].append((np.stack([np.log(q + 1e-9), np.log(p + 1e-12), np.log(m + 1e-12)], 1).astype(np.float32), ok, wi))
    BW = {}
    for k in "tf":
        dd = data[k]
        def nll(w):
            L = 0; Gr = np.zeros(3)
            for Xm, ok, wi in dd:
                s = np.where(ok, Xm @ w, -1e9); mx = s.max(); e = np.exp(s - mx) * ok; Zs = e.sum(); p = e / Zs
                L += s[wi] - mx - np.log(Zs); Gr += Xm[wi] - (p[:, None] * Xm).sum(0)
            return -L, -Gr
        BW[k] = minimize(nll, np.array([0.85, 0.7, -0.5]), jac=True, method="L-BFGS-B").x.tolist()
    return BW, len(data["f"])


def evaluate(rids, W3, WM, BW):
    out = {"trio": [], "tri": []}; nll = []
    for rid in rids:
        a = arrays(rid, W3, WM)
        if not a: continue
        pr, _ = prod.exotic_probs(a["s_ai"], a["s_mk"], a["tri"], a["trio"], BW)
        if pr["f"][0] is not None and pr["f"][1][a["wf"]]: nll.append(-np.log(pr["f"][0][a["wf"]] + 1e-12))
        pk = prod.picks(a["n"], pr, a["tri"], a["trio"])
        out["trio"] += [a["pay_f"] if i == a["wf"] else 0 for i in pk["trio"]]
        out["tri"] += [a["pay_t"] if i == a["wt"] else 0 for i in pk["tri"]]
    def f(r):
        r = np.array(r); s = np.sort(r)[::-1]; c = len(r) * 100
        return f"{len(r)}点 的中{(r>0).sum()} 回収率{r.sum()/c:.1%} 上位3つ除く{(r.sum()-s[:3].sum())/c:.1%}" if len(r) else "なし"
    return f"3連複NLL {np.mean(nll):.4f} | 3連複 {f(out['trio'])} | 3連単 {f(out['tri'])}"


rng = np.random.default_rng(0)
exrids = set(EX)
res = {}
for nm, start in (("A 2022-03〜", "20220301"), ("B 2017-06〜", "20170601")):
    tr = df[(df.date >= start) & (df.date <= "20250531")]
    W3 = pl.fit(pl.pack(tr, ["logp_mkt"] + prod.Z), l2=3, stages=3); WM = pl.fit(pl.pack(tr, ["logp_mkt"]), l2=3, stages=3)
    va = df[(df.date >= VA[0]) & (df.date <= VA[1])]
    nl = pl._loglik(W3, pl.pack(va, ["logp_mkt"] + prod.Z), 0, 3)[0] / va.rid.nunique()
    trr = [r for r in tr.rid.unique() if r in exrids]; samp = rng.choice(trr, min(6000, len(trr)), replace=False)
    BW, nb = fit_bw(samp, W3, WM)
    print(nm, "勝ち馬 学習", tr.rid.nunique(), "レース | 基準決め 3着までNLL", round(nl, 4), "| 組み合わせ", {k: np.round(v, 3).tolist() for k, v in BW.items()}, flush=True)
    res[nm] = (W3, WM, BW)
    for pn, (a, b) in (("基準決め", VA),):
        rids = [r for r in df[(df.date >= a) & (df.date <= b)].rid.unique()]
        print("  ", pn, evaluate(rids, W3, WM, BW), flush=True)
pickle.dump(res, open(C / "models.pkl", "wb"))
print("--- 採点（1回だけ）")
for nm, (W3, WM, BW) in res.items():
    rids = [r for r in df[(df.date >= TE[0]) & (df.date <= TE[1])].rid.unique()]
    print(nm, evaluate(rids, W3, WM, BW), flush=True)
