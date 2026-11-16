"""本番用の学習（GitHub Actions で週1回と手動）。全期間のデータで学習して enc/model.json（暗号化）に保存する。

検証（どの作り方・買い方が良いか）はこのファイルではしない。それは model/experiments/ と REPORT.md。
ここでは、検証で決めた作り方を、使えるデータ全部でもう一度学習するだけ。

使い方（GitHub Actions 内）: APP_PASS=... python model/train.py --data data --odds odds
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scraper"))
import exotic as X  # noqa: E402
import features as F  # noqa: E402
import pl  # noqa: E402
import prod  # noqa: E402
from store import Store  # noqa: E402

START = "20220301"  # これより前は過去走の履歴づくりにだけ使う


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--odds", default="odds")
    a = ap.parse_args()
    D, O = Store(Path(a.data)), Store(Path(a.odds))
    dates = sorted(p.stem for p in (Path(a.data) / "enc" / "results").glob("*.json"))
    races = prod.load_results(D, dates)
    last = dates[-1]
    print(f"結果 {len(races)}レース（{dates[0]}〜{last}）", flush=True)
    df = F.runners_frame(races)
    df = F.add_figures(df, last)
    df = F.add_history(df)
    df = F.finalize(df)
    df = df[(df.surface != "障害") & df.pos.notna() & df.odds.notna()].copy().reset_index(drop=True)
    inv = 1 / df.odds; df["p_mkt"] = inv / inv.groupby(df.rid).transform("sum"); df["logp_mkt"] = np.log(df.p_mkt)
    tr = df[df.date >= START]
    W3 = pl.fit(pl.pack(tr, ["logp_mkt"] + prod.Z), l2=3, stages=3)
    WM = pl.fit(pl.pack(tr, ["logp_mkt"]), l2=3, stages=3)
    print("勝ち馬モデル 学習", tr.rid.nunique(), "レース", flush=True)

    # 3連複・3連単の組み合わせの重み
    byid = {r["id"]: r for r in races}
    rows = []
    for d in dates:
        if d >= START:
            rows += list(O.read(f"exotic/{d}.json", {"races": {}})["races"].items())
    gidx = {rid: g for rid, g in df[df.date >= START].groupby("rid", sort=False)}
    data = []
    for rid, v in rows:
        g = gidx.get(rid); race = byid.get(rid)
        if g is None or race is None or "3連単" not in race["pay"] or len(race["pay"]["3連単"]) > 1 or len(race["pay"].get("3連複", [])) > 1:
            continue
        nos = v["nos"]; n = len(nos)
        tri = np.frombuffer(base64.b64decode(v["tri"]), dtype="<u4") / 10
        trio = np.frombuffer(base64.b64decode(v["trio"]), dtype="<u4") / 10
        if len(tri) != n * (n - 1) * (n - 2):
            continue
        pos = {no: i for i, no in enumerate(nos)}
        sa, sm = prod.strengths(g, W3, WM)
        s_ai, s_mk = np.zeros(n), np.zeros(n)
        for i, no in enumerate(g["no"]):
            s_ai[pos[no]], s_mk[pos[no]] = sa[i], sm[i]
        pt, pmt = X.pl_tri(s_ai), X.pl_tri(s_mk)
        P, C, _ = X.perms(n)
        w3 = [pos[int(x)] for x in race["pay"]["3連単"][0]["combo"].split("-")]
        wt = int(np.where((P[:, 0] == w3[0]) & (P[:, 1] == w3[1]) & (P[:, 2] == w3[2]))[0][0]); sw = sorted(w3)
        wf = int(np.where((C[:, 0] == sw[0]) & (C[:, 1] == sw[1]) & (C[:, 2] == sw[2]))[0][0])
        for k, o, p, m, wi in (("t", tri, pt, pmt, wt), ("f", trio, X.trio_from_tri(pt, n), X.trio_from_tri(pmt, n), wf)):
            q, ok = X.market_prob(o)
            data.append((k, np.stack([np.log(q + 1e-9), np.log(p + 1e-12), np.log(m + 1e-12)], 1), ok, wi))
    BW = {}
    for k in "tf":
        dd = [(Xm, ok, wi) for kk, Xm, ok, wi in data if kk == k]

        def nll(w):
            L = 0; G = np.zeros(3)
            for Xm, ok, wi in dd:
                s = np.where(ok, Xm @ w, -1e9); mx = s.max(); e = np.exp(s - mx) * ok; Zs = e.sum(); p = e / Zs
                if not ok[wi]:
                    continue
                L += s[wi] - mx - np.log(Zs); G += Xm[wi] - (p[:, None] * Xm).sum(0)
            return -L, -G
        BW[k] = minimize(nll, np.array([0.85, 0.7, -0.5]), jac=True, method="L-BFGS-B").x.tolist()
        print("組み合わせ", k, np.round(BW[k], 3), len(dd), "レース", flush=True)
    model = {"version": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M"),
             "fig_end": last, "train": [START, last], "races_win": int(tr.rid.nunique()), "races_exotic": len(data) // 2,
             "W3": np.asarray(W3).tolist(), "WM": np.asarray(WM).tolist(), "BW": BW, "features": F.FEATURES, "rules": prod.RULES}
    D.write("model.json", model)
    print("model.json を保存しました", model["version"])


if __name__ == "__main__":
    main()
