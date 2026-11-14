"""過去レースの確定 3連複・3連単オッズの取り込み（小分けにして途中で保存）。

data ブランチの過去結果（enc/results/）にある各レースについて、単複オッズのページから
3連複・3連単のページへのリンクをたどって取る。新しい日付から順に。
保存先：odds ブランチの enc/exotic/YYYYMMDD.json（gzip→暗号化）。進み具合は state.json（取り終えたレースIDだけ）。

使い方（GitHub Actions 内）: APP_PASS=... python scraper/odds_history.py --data data --odds odds --minutes 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import jra  # noqa: E402
import odds_codec  # noqa: E402
from store import Store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--odds", default="odds")
    ap.add_argument("--minutes", type=float, default=40)
    a = ap.parse_args()
    t0 = time.time()
    src, dst = Store(Path(a.data)), Store(Path(a.odds))
    st_file = Path(a.odds) / "state.json"
    st = json.loads(st_file.read_text()) if st_file.exists() else {"done": [], "failed": []}
    done = set(st["done"])
    cl = jra.Client()
    dates = sorted((p.stem for p in (Path(a.data) / "enc" / "results").glob("*.json")), reverse=True)
    n_new = 0

    def save(date, day):
        dst.write(f"exotic/{date}.json", day, gz=True)
        st["done"] = sorted(done)
        st_file.write_text(json.dumps(st), encoding="utf-8")

    for date in dates:
        races = src.read(f"results/{date}.json", {"races": {}})["races"]
        todo = [r for r in races.values() if r["id"] not in done and r.get("odds_cname") and "障害" not in (r.get("name") or "")]
        if not todo:
            continue
        day = dst.read(f"exotic/{date}.json", {"races": {}})
        for r in todo:
            if time.time() - t0 > a.minutes * 60:
                save(date, day)
                print(f"時間切れ。今回 {n_new}レース / 合計 {len(done)}レース / アクセス {cl.count}回")
                return
            try:
                tf = cl.get("/JRADB/accessO.html", r["odds_cname"])
                c7 = [c for c in jra.links(tf, "pw157ou") if (jra.race_key(c) or {}).get("id") == r["id"]]
                c8 = [c for c in jra.links(tf, "pw158ou") if (jra.race_key(c) or {}).get("id") == r["id"]]
                if not c7 or not c8:
                    raise RuntimeError("3連複・3連単へのリンクがありません")
                trio = jra.parse_odds_trio(cl.get("/JRADB/accessO.html", c7[0]))
                tri = jra.parse_odds_trifecta(cl.get("/JRADB/accessO.html", c8[0]))
            except RuntimeError as e:
                print("  取得できず", r["id"], e)
                st["failed"].append(r["id"]); done.add(r["id"])
                continue
            nos = sorted({x["no"] for x in r["runners"]})
            day["races"][r["id"]] = {"nos": nos, "trio": odds_codec.pack(nos, trio, False), "tri": odds_codec.pack(nos, tri, True),
                                     "n_trio": len(trio), "n_tri": len(tri)}
            done.add(r["id"]); n_new += 1
            if n_new % 50 == 0:
                print(f"  {n_new}レース（{r['id']}）{(time.time()-t0)/60:.0f}分", flush=True)
        save(date, day)
    print(f"すべて取り終えました。今回 {n_new}レース / 合計 {len(done)}レース")


if __name__ == "__main__":
    main()
