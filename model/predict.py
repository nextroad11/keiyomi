"""開催日の予想と記録（1分ごとに daily.py のあとで呼ばれる）。

- 最新の締切前オッズで、各馬の AI 確率・1番人気が飛ぶ確率・3連単1万円以上の確率・3連複/3連単の買い目を計算。
- 記録用の買い目：3連複・3連単のオッズが「5分前」以降に取れた最初の時点で固定する（あとから変えない）。
  5分前が取れなかったレースは10分前のものを使い、その旨を残す。実際には買わない。
- 結果が出たら、固定した買い目を実際の払戻で採点して paper/records.json に足す。

保存先（data ブランチ、暗号化）: pred/YYYYMMDD.json、pred/index.json、paper/records.json
使い方（GitHub Actions 内）: APP_PASS=... python model/predict.py --data data --odds odds
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scraper"))
import exotic as X  # noqa: E402
import features as F  # noqa: E402
import odds_codec  # noqa: E402
import prod  # noqa: E402
from store import Store  # noqa: E402

JST = ZoneInfo("Asia/Tokyo")
FREEZE_AT = 5.5     # この分数以内のオッズで記録用の買い目を固定
FALLBACK_AT = 10.5  # 5分前が取れず発走した場合に使う時点


def feature_rows(D, live, model, cache_dir: Path) -> pd.DataFrame:
    date = live["date"]
    have = {rid for rid, r in live["races"].items() if r.get("card")}
    f = cache_dir / f"keiyomi_feat_{date}_{model['version'].replace(' ', '_').replace(':', '')}.pkl"
    if f.exists():
        cached = pickle.loads(f.read_bytes())
        if have <= set(cached["rid"].unique()):
            return cached
    dates = sorted(p.stem for p in (D.data / "enc" / "results").glob("*.json") if p.stem < date)
    hist = F.runners_frame(prod.load_results(D, dates))
    today = prod.today_frame(live)
    df = prod.build_features(hist, today, model["fig_end"])
    out = df[df["date"] == date].copy()
    f.write_bytes(pickle.dumps(out))
    print(f"特徴量を作りました（今日 {out.rid.nunique()}レース・{len(out)}頭）", flush=True)
    return out


def latest(snaps: dict):
    best = None
    for m, v in (snaps or {}).items():
        if v and (best is None or v["mins"] < best["mins"]):
            best = v
    return best


def combo_str(n_idx, nos, ordered):
    return ("→" if ordered else "-").join(str(nos[i]) for i in n_idx)


def predict_race(rid, r, rows, ex_snaps, model):
    card = r.get("card") or {}
    tf = latest(r.get("tf"))
    odds_now = {int(k): v for k, v in (tf or {}).get("odds", {}).items()}
    g = rows[rows.rid == rid].copy()
    if not len(g):
        return None
    for i, x in g.iterrows():
        o = odds_now.get(int(x["no"]))
        if o:
            g.at[i, "odds"] = o.get("win"); g.at[i, "body"] = o.get("body"); g.at[i, "body_diff"] = o.get("body_diff")
    out = {"rid": rid, "post": r.get("post"), "venue": r.get("venue"), "rno": r.get("rno"), "name": card.get("name"),
           "surface": card.get("surface"), "distance": card.get("distance"), "going": card.get("going"), "cls": card.get("cls"),
           "n": len(card.get("runners", [])), "odds_at": tf["at"] if tf else None, "odds_mins": tf["mins"] if tf else None, "horses": []}
    names = {x["no"]: x for x in card.get("runners", []) if x.get("no") is not None}
    gg = g[g["odds"].notna() & (g["odds"] > 1)]
    if len(gg) < 3:
        out["horses"] = [{"no": int(no), "name": x.get("name"), "jockey": x.get("jockey"), "odds": None} for no, x in sorted(names.items())]
        out["note"] = "単勝オッズが取得不可のため、AIの確率はまだ出せません"
        return out
    gg = prod.zscore_race(gg)
    s_ai, s_mk = prod.strengths(gg, model["W3"], model["WM"])
    nos = sorted(int(n) for n in gg["no"])
    idx = {int(no): i for i, no in enumerate(gg["no"])}
    sa = np.array([s_ai[idx[n]] for n in nos]); sm = np.array([s_mk[idx[n]] for n in nos])
    n = len(nos)
    pt = X.pl_tri(sa); P, C, M = X.perms(n)
    top3 = np.zeros(n)
    for j in range(3):
        np.add.at(top3, P[:, j], pt)
    rank = np.argsort(-sa)
    marks = {nos[i]: prod.MARKS[k] for k, i in enumerate(rank[:5])}
    fav = int(gg.loc[gg["odds"].idxmin(), "no"])
    ptm = X.pl_tri(sm)
    for no, x in sorted(names.items()):
        i = nos.index(no) if no in nos else None
        o = odds_now.get(no, {})
        out["horses"].append({
            "no": int(no), "waku": x.get("waku"), "name": x.get("name"), "jockey": x.get("jockey"), "weight": x.get("weight"),
            "sexage": x.get("sexage"), "body": o.get("body"), "body_diff": o.get("body_diff"),
            "odds": o.get("win") if o else x.get("odds"), "mark": marks.get(no, ""),
            "ai": round(float(sa[i]), 4) if i is not None else None, "mkt": round(float(gg.loc[gg.no == no, "p_mkt"].iloc[0]), 4) if i is not None else None,
            "ai3": round(float(top3[i]), 4) if i is not None else None,
            "past": [(p or {}).get("pos_text") for p in (x.get("past") or [])],
        })
    out["fav"] = fav
    out["favout"] = round(float(1 - top3[nos.index(fav)]), 4)
    ex = latest(ex_snaps)
    if ex and sorted(ex["nos"]) == sorted(ex["nos"]):
        enos = sorted(ex["nos"])
        tri = odds_codec.unpack(enos, ex["tri"], True); trio = odds_codec.unpack(enos, ex["trio"], False)
        # 取消の馬を除いた並びに合わせる
        o_t = np.array([tri.get(tuple(nos[i] for i in p), 0.0) for p in P])
        o_f = np.array([trio.get(tuple(nos[i] for i in c), 0.0) for c in C])
        probs, _ = prod.exotic_probs(sa, sm, o_t, o_f, model["BW"])
        pk = prod.picks(n, probs, o_t, o_f)
        p_t, _ = probs["t"]; p_f, _ = probs["f"]
        out["ex_at"], out["ex_mins"] = ex["at"], ex["mins"]
        # ほかの券種（検証中）
        if ex.get("pairs") or odds_now:
            pr = {k: odds_codec.unpack_pairs(enos, v, k) for k, v in (ex.get("pairs") or {}).items()}
            wk = [names.get(no, {}).get("waku") for no in nos]
            out["others"] = prod.pair_picks(sa, sm, nos, wk, {k: v.get("win") for k, v in odds_now.items()},
                                            {k: v.get("pmin") for k, v in odds_now.items()}, pr)
        out["man"] = round(float(p_t[o_t >= 100].sum()), 4) if p_t is not None else None
        out["trio"] = [{"c": combo_str(C[i], nos, False), "p": round(float(p_f[i]), 4), "o": float(o_f[i]), "ev": round(float(p_f[i] * o_f[i]), 3)} for i in pk["trio"]]
        out["tri"] = [{"c": combo_str(P[i], nos, True), "p": round(float(p_t[i]), 5), "o": float(o_t[i]), "ev": round(float(p_t[i] * o_t[i]), 3)} for i in pk["tri"]]
        # 参考：AIの確率が高い3連複の上位（買い目ではない）
        if p_f is not None:
            top = np.argsort(-p_f)[:5]
            out["trio_top"] = [{"c": combo_str(C[i], nos, False), "p": round(float(p_f[i]), 4), "o": float(o_f[i]), "ev": round(float(p_f[i] * o_f[i]), 3)} for i in top]
    return out


def score(paper, race_res):
    pay = race_res.get("pay", {})
    def hit(kind, picks):
        rows = pay.get(kind, [])
        won = {r["combo"].replace(" ", "") for r in rows}
        y = 0
        for pk in picks:
            key = pk["c"].replace("→", "-")
            for r in rows:
                if r["combo"].replace(" ", "") == key:
                    y += r["yen"]
        return y
    t = {"trio": {"pts": len(paper["trio"]), "ret": hit("3連複", paper["trio"])},
         "tri": {"pts": len(paper["tri"]), "ret": hit("3連単", paper["tri"])}}
    t["others"] = {k: {"pts": len(v), "ret": hit(k, v)} for k, v in (paper.get("others") or {}).items()}
    t["trio_pay"] = (pay.get("3連複") or [{}])[0].get("yen"); t["tri_pay"] = (pay.get("3連単") or [{}])[0].get("yen")
    order = sorted([x for x in race_res.get("runners", []) if x.get("pos")], key=lambda x: x["pos"])[:3]
    t["top3"] = [x["no"] for x in order]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--odds", default="odds")
    ap.add_argument("--date", default=None)
    ap.add_argument("--cache", default="/tmp")
    a = ap.parse_args()
    D, O = Store(Path(a.data)), Store(Path(a.odds))
    D.data = Path(a.data)
    date = a.date or dt.datetime.now(JST).strftime("%Y%m%d")
    model = D.read("model.json")
    if not model:
        print("model.json がまだありません（学習を先に）"); return
    live = D.read(f"live/{date}.json")
    if not live or not live.get("races"):
        print(f"{date} の予定がありません"); return
    rows = feature_rows(D, live, model, Path(a.cache))
    exl = O.read(f"live/{date}.json", {"races": {}})["races"]
    res = D.read(f"results/{date}.json", {"races": {}})["races"]
    exf = O.read(f"exotic/{date}.json", {"races": {}})["races"]
    old = D.read(f"pred/{date}.json", {"date": date, "races": {}})
    new = {"date": date, "model": model["version"], "races": {}}
    for rid, r in sorted(live["races"].items(), key=lambda kv: kv[1].get("post") or ""):
        prev = old["races"].get(rid, {})
        p = predict_race(rid, r, rows, exl.get(rid), model) or {"rid": rid, "post": r.get("post"), "venue": r.get("venue"), "rno": r.get("rno")}
        if "paper" in prev:
            p["paper"] = prev["paper"]
        elif p.get("ex_mins") is not None and p["ex_mins"] <= FREEZE_AT:
            p["paper"] = {"mins": p["ex_mins"], "at": p["ex_at"], "trio": p.get("trio", []), "tri": p.get("tri", []), "others": p.get("others", {})}
        elif rid in res and p.get("ex_mins") is not None and p["ex_mins"] <= 30.5:
            # 5分前が取れずに発走した：取れた中で一番遅い締切前のオッズで固定（何分前かを残す）
            p["paper"] = {"mins": p["ex_mins"], "at": p["ex_at"], "trio": p.get("trio", []), "tri": p.get("tri", []), "others": p.get("others", {}), "late": True}
        if rid in res:
            p["result"] = score(p.get("paper") or {"trio": [], "tri": []}, res[rid])
            if "paper" not in p:
                p["result"]["no_paper"] = True  # 締切前の3連オッズが取れなかった
                # 参考：確定オッズで計算した買い目（実際には選べなかったので記録には入れない）
                if "ref" in prev:
                    p["ref"] = prev["ref"]
                elif rid in exf:
                    fin = res[rid].get("odds_final") or {}
                    r2 = dict(r); r2["tf"] = {"確定": {"mins": 0, "at": "確定", "odds": fin}}
                    v = exf[rid]
                    q = predict_race(rid, r2, rows, {"確定": {"mins": 0, "at": "確定", "nos": v["nos"], "tri": v["tri"], "trio": v["trio"]}}, model)
                    if q and "trio" in q:
                        ref = {"trio": q["trio"], "tri": q["tri"]}
                        p["ref"] = {**ref, "result": score(ref, res[rid])}
            # 発走後は、締切前の最後の予想のまま残す（確定オッズで計算し直さない）
            for k in ("horses", "trio", "tri", "trio_top", "others", "man", "favout", "odds_at", "odds_mins", "ex_at", "ex_mins", "fav"):
                if k in prev:
                    p[k] = prev[k]
        new["races"][rid] = p
    if json.dumps(new, sort_keys=True) != json.dumps(old, sort_keys=True):
        D.write(f"pred/{date}.json", new)
        idx = D.read("pred/index.json", {"dates": []})
        if date not in idx["dates"]:
            idx["dates"] = sorted(idx["dates"] + [date]); D.write("pred/index.json", idx)
        # 記録（採点済みのものだけ）
        rec = D.read("paper/records.json", {"races": {}})
        changed = False
        for rid, p in new["races"].items():
            if "result" in p and "paper" in p:
                row = {"date": date, "rid": rid, "venue": p.get("venue"), "rno": p.get("rno"), "name": p.get("name"),
                       "mins": p["paper"]["mins"], "late": p["paper"].get("late", False), **p["result"]}
                if rec["races"].get(rid) != row:
                    rec["races"][rid] = row; changed = True
        if changed:
            D.write("paper/records.json", rec)
        print(f"{date}: 予想 {len(new['races'])}レース / 記録 {sum(1 for p in new['races'].values() if 'paper' in p)} / 採点 {sum(1 for p in new['races'].values() if 'result' in p)}")


if __name__ == "__main__":
    main()
