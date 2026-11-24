import sys, json, gzip, pathlib, pickle, numpy as np, collections
sys.path.insert(0, "/home/claude/keiyomi/model"); sys.path.insert(0, "/home/claude/keiyomi/scraper")
import prod, exotic as X, odds_codec
from crypt import unseal
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
priv = X25519PrivateKey.from_private_bytes(bytes.fromhex(pathlib.Path("~/.keiyomi/probe_priv.hex").expanduser().read_text().strip()))
def op(p):
    raw = unseal(priv, json.loads(pathlib.Path(p).read_text()))
    if raw[:2] == b"\x1f\x8b": raw = gzip.decompress(raw)
    return json.loads(raw)
W3, WM, BW = pickle.load(open("/tmp/ten/models.pkl", "rb"))["A 2022-03〜"]
R = prod.RULES["trio"]
def reason(pf, okf, of):
    if pf is None or okf.sum() == 0: return "3連複オッズなし"
    ev = pf * of
    a = okf & (ev >= R["th"])
    if not a.any(): return "期待値1以上の組なし"
    b = a & (pf >= R["pmin"])
    if not b.any(): return "期待値≧1はあるが確率1%未満のみ"
    c = b & (of >= R["omin"])
    if not c.any(): return "期待値≧1・確率≧1%はあるがオッズ10倍未満のみ"
    return "買い"
stats = collections.Counter(); rows = []; agree = [0, 0]
for D in ("20261121", "20261122", "20261123"):
    P = op(f"/tmp/kd4/dev/pred/{D}.json"); OL = op(f"/tmp/ko4/dev/live/{D}.json"); FX = op(f"/tmp/ko4/dev/exotic/{D}.json")
    RS = op(f"/tmp/kd4/dev/results/{D}.json")["races"]
    for rid, p in P["races"].items():
        hs = [h for h in p.get("horses", []) if h.get("ai") is not None]
        res = RS.get(rid)
        if len(hs) < 3 or not res: stats["予想なし・結果なし"] += 1; continue
        nos = sorted(h["no"] for h in hs); n = len(nos)
        sa = np.array([next(h["ai"] for h in hs if h["no"] == k) for k in nos]); sa /= sa.sum()
        mk = np.array([next(h["mkt"] for h in hs if h["no"] == k) for k in nos]); sm = mk ** WM[0]; sm /= sm.sum()
        Pp, C, M = X.perms(n)
        top = [x["no"] for x in sorted([x for x in res["runners"] if x.get("pos")], key=lambda x: x["pos"])[:3]]
        if len(top) < 3 or any(t not in nos for t in top): stats["着順照合できず"] += 1; continue
        wf = int(np.where((C == sorted(nos.index(t) for t in top)).all(1))[0][0])
        payf = (res["pay"].get("3連複") or [{}])[0].get("yen", 0)
        out = {"rid": rid, "paper": p.get("paper") is not None}
        for lab, src in (("5分前", (OL.get("races", {}).get(rid) or {}).get("5")), ("確定", FX["races"].get(rid))):
            if not src: out[lab] = None; continue
            enos = sorted(src["nos"])
            tri = odds_codec.unpack(enos, src["tri"], True); trio = odds_codec.unpack(enos, src["trio"], False)
            o_t = np.array([tri.get(tuple(nos[i] for i in q), 0.0) for q in Pp]); o_f = np.array([trio.get(tuple(nos[i] for i in c), 0.0) for c in C])
            pr, _ = prod.exotic_probs(sa, sm, o_t, o_f, BW); pf, okf = pr["f"]
            pk = prod.picks(n, pr, o_t, o_f)
            q, okq = X.market_prob(o_f)
            out[lab] = dict(reason=reason(pf, okf, o_f), picks=pk["trio"], pw=float(pf[wf]) if pf is not None else None, qw=float(q[wf]),
                            ret=sum(payf for i in pk["trio"] if i == wf), sump=float(sum(pf[i] for i in pk["trio"])), maxev=float((pf*o_f)[okf].max()) if pf is not None else 0)
        if out["5分前"] and p.get("paper"):
            mine = {tuple(sorted(int(t) for t in x["c"].split("-"))) for x in p["paper"]["trio"]}
            recon = {tuple(nos[i] for i in C[j]) for j in out["5分前"]["picks"]}
            agree[0] += mine == recon; agree[1] += 1
        rows.append(out)
pickle.dump(rows, open("/tmp/audit_live.pkl", "wb"))
print("レース", len(rows), stats, "本番の記録と再現の一致", agree)
for lab in ("5分前", "確定"):
    rr = [r[lab] for r in rows if r[lab]]
    c = collections.Counter(r["reason"] for r in rr)
    bet = [r for r in rr if r["picks"]]
    pts = sum(len(r["picks"]) for r in bet); ret = sum(r["ret"] for r in bet); ep = sum(r["sump"] for r in bet)
    print(lab, dict(c), f"| 買いレース{len(bet)} 点{pts} 払戻{ret} 期待的中数{ep:.2f}")
    # 予測確率の精度：勝った組の確率の対数（AI vs 市場）
    lw = np.mean([np.log(r["pw"] + 1e-9) for r in rr]); lq = np.mean([np.log(r["qw"] + 1e-9) for r in rr])
    print("   当たった組の平均対数確率 AI", round(lw, 3), "市場", round(lq, 3))
    for grp, sel in (("買い", [r for r in rr if r["picks"]]), ("見送り", [r for r in rr if not r["picks"]])):
        if sel: print("   ", grp, len(sel), "AI", round(np.mean([np.log(r["pw"]+1e-9) for r in sel]),3), "市場", round(np.mean([np.log(r["qw"]+1e-9) for r in sel]),3))
