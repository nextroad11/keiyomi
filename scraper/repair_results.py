"""速報（上位5頭だけ）で保存されてしまった結果の取り直し（11/15〜の不具合の修復。一回きり）。

対象：結果の頭数が出馬表より大きく少ないレース。結果ページ・確定の単複/3連複/3連単/4券種を取り直して上書きする。
使い方（GitHub Actions 内）: APP_PASS=... python scraper/repair_results.py --data data --odds odds 20261115 20261121 ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import daily  # noqa: E402
import jra  # noqa: E402
import odds_codec  # noqa: E402
from store import Store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--odds", default="odds")
    ap.add_argument("dates", nargs="+")
    a = ap.parse_args()
    D, O = Store(Path(a.data)), Store(Path(a.odds))
    cl = jra.Client()
    links = {}
    for ym in sorted({d[:6] for d in a.dates}):
        top = cl.get("/JRADB/accessS.html", "pw01skl00999999/B3")
        months = [c for c in jra.links(top, "pw01skl1") if c[9:15] == ym] or ["pw01skl00999999/B3"]
        html = cl.get("/JRADB/accessS.html", months[0])
        for lc in jra.links(html, "pw01srl1"):
            k = jra.race_key(lc)
            if k and k["date"] in a.dates:
                for rc in jra.links(cl.get("/JRADB/accessS.html", lc), "pw01sde1"):
                    rk = jra.race_key(rc)
                    if rk:
                        links[rk["id"]] = rc
    print("結果ページ", len(links))
    fixed = 0
    for date in a.dates:
        live = D.read(f"live/{date}.json", {"races": {}})
        res = D.read(f"results/{date}.json", {"races": {}}); ex = O.read(f"exotic/{date}.json", {"races": {}}); pr = O.read(f"pairs/{date}.json", {"races": {}})
        n = 0
        for rid, old in sorted(res["races"].items()):
            r = live["races"].get(rid) or {}
            if not daily.incomplete(old, r) and len(old.get("runners", [])) > 5:
                continue
            if rid not in links:
                print("結果ページが見つからない", rid); continue
            race = jra.parse_result(cl.get("/JRADB/accessS.html", links[rid]))
            if daily.incomplete(race, r) or len(race.get("runners", [])) <= len(old.get("runners", [])):
                print("まだ速報のまま", rid, len(race.get("runners", []))); continue
            race.update({k: old[k] for k in ("id", "jcd", "venue", "kai", "day", "date") if k in old})
            race["rno"] = race.get("rno") or old.get("rno")
            tf_html = cl.get("/JRADB/accessO.html", race.get("odds_cname") or old.get("odds_cname"))
            race["odds_final"] = {str(k): v for k, v in jra.parse_odds_tf(tf_html).items()} or old.get("odds_final")
            c7, c8 = daily._exotic_links(tf_html, rid)
            trio = jra.parse_odds_trio(cl.get("/JRADB/accessO.html", c7)) if c7 else {}
            tri = jra.parse_odds_trifecta(cl.get("/JRADB/accessO.html", c8)) if c8 else {}
            nos = sorted({x["no"] for x in race["runners"]})
            res["races"][rid] = race
            ex["races"][rid] = {"nos": nos, "trio": odds_codec.pack(nos, trio, False), "tri": odds_codec.pack(nos, tri, True), "n_trio": len(trio), "n_tri": len(tri)}
            pr["races"][rid] = {"nos": nos, **daily._fetch_pairs(cl, tf_html, rid, nos)}
            n += 1; print(rid, len(old["runners"]), "→", len(race["runners"]), "頭", flush=True)
        if n:
            D.write(f"results/{date}.json", res); O.write(f"exotic/{date}.json", ex, gz=True); O.write(f"pairs/{date}.json", pr, gz=True)
        fixed += n
    print("直したレース", fixed, "アクセス", cl.count)


if __name__ == "__main__":
    main()
