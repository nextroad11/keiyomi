"""地方競馬：確定オッズ（2026-02〜）と払戻をレースごとの配列にまとめる。"""
import glob, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/keiyomi/model")
import exotic as X

RAW = "/tmp/narraw"
TYPES = {"単勝": "win", "複勝": "place", "枠複": "wakuren", "枠単": "wakutan", "馬複": "umaren", "馬単": "umatan", "ワイド": "wide", "３連複": "trio", "３連単": "tri"}


def load_odds():
    R = {}
    for f in sorted(glob.glob(f"{RAW}/odds/*/*_odds.csv")):
        d = pd.read_csv(f, dtype={"競馬場": str, "競走年月日": str, "賭式": str})
        d = d[d["競馬場"] != "帯広ば"]
        d["rid"] = d["競走年月日"] + "-" + d["競馬場"] + "-" + d["レース番号"].astype(int).astype(str).str.zfill(2)
        for rid, g in d.groupby("rid", sort=False):
            r = {}
            for t, gg in g.groupby("賭式"):
                k = TYPES.get(t)
                if not k:
                    continue
                a = gg[["番号1", "番号2", "番号3"]].fillna(0).astype(int).to_numpy()
                o = pd.to_numeric(gg["オッズ"], errors="coerce").fillna(0).to_numpy()
                r[k] = (a, o)
            win = r.get("win")
            if not win:
                continue
            nos = sorted(int(x) for x in win[0][:, 0])
            n = len(nos); idx = {v: i for i, v in enumerate(nos)}
            out = {"nos": nos, "n": n, "win": np.array([dict(zip(win[0][:, 0], win[1])).get(v, 0) for v in nos], float)}
            if "tri" in r and n >= 3:
                P, C, M = X.perms(n); key = {tuple(p): i for i, p in enumerate(P.tolist())}
                ot = np.zeros(len(P))
                for (a, b, c), v in zip(r["tri"][0], r["tri"][1]):
                    if a in idx and b in idx and c in idx:
                        ot[key[(idx[a], idx[b], idx[c])]] = v
                out["tri"] = ot
            if "trio" in r and n >= 3:
                P, C, M = X.perms(n); key = {tuple(p): i for i, p in enumerate(C.tolist())}
                of = np.zeros(len(C))
                for (a, b, c), v in zip(r["trio"][0], r["trio"][1]):
                    if a in idx and b in idx and c in idx:
                        of[key[tuple(sorted((idx[a], idx[b], idx[c])))]] = v
                out["trio"] = of
            for k in ("place", "wakuren", "wakutan", "umaren", "umatan", "wide"):
                if k in r:
                    a, o = r[k]
                    out[k] = {tuple(int(x) for x in row if x) : v for row, v in zip(a.tolist(), o)}
            R[rid] = out
        print(f, len(R), flush=True)
    return R


def load_pay():
    P = {}
    cols = {"win": [("単勝組番", "単勝払戻金（円）")],
            "place": [(f"複勝組番{i}", f"複勝払戻金{i}（円）") for i in (1, 2, 3)],
            "wakuren": [(("枠複組番1", "枠複組番2"), "枠複払戻金（円）")],
            "wakutan": [(("枠単組番1", "枠単組番2"), "枠単払戻金（円）")],
            "umaren": [(("馬複組番1", "馬複組番2"), "馬複払戻金（円）")],
            "umatan": [(("馬単組番1", "馬単組番2"), "馬単払戻金（円）")],
            "wide": [((f"ワイド組番{i}馬番1", f"ワイド組番{i}馬番2"), f"ワイド払戻金{i}（円）") for i in (1, 2, 3)],
            "trio": [(("３連複組番馬番1", "３連複組番馬番2", "３連複組番馬番3"), "３連複払戻金（円）")],
            "tri": [(("３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3"), "３連単払戻金（円）")]}
    for f in sorted(glob.glob(f"{RAW}/race/2026*/*_payback.csv")):
        d = pd.read_csv(f, dtype=str)
        d = d[d["競馬場"] != "帯広ば"]
        for _, row in d.iterrows():
            rid = row["競走年月日"] + "-" + row["競馬場"] + "-" + str(int(row["レース番号"])).zfill(2)
            p = {}
            for k, lst in cols.items():
                p[k] = []
                for c, v in lst:
                    c = (c,) if isinstance(c, str) else c
                    if c[0] not in row or pd.isna(row[c[0]]) or pd.isna(row.get(v)):
                        continue
                    try:
                        p[k].append((tuple(int(float(row[x])) for x in c), int(float(row[v]))))
                    except (ValueError, TypeError):
                        pass
            P[rid] = p
    return P


if __name__ == "__main__":
    O = load_odds(); pickle.dump(O, open("/tmp/nar_odds.pkl", "wb"), protocol=4)
    P = load_pay(); pickle.dump(P, open("/tmp/nar_pay.pkl", "wb"), protocol=4)
    print(len(O), len(P)); k = next(iter(O)); print(k, {a: (type(b).__name__) for a, b in O[k].items()}, P.get(k))
