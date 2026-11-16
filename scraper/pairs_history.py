"""過去レースの確定の枠連・馬連・ワイド・馬単オッズの取り込み（小分けにして途中で保存）。

data ブランチの過去結果の各レースについて、単複オッズのページから4券種のページへのリンクをたどる。新しい日付から順に。
保存先：odds ブランチの enc/pairs/YYYYMMDD.json（gzip→暗号化）。進み具合は pairs_state.json（取り終えたレースIDだけ）。

使い方（GitHub Actions 内）: APP_PASS=... python scraper/pairs_history.py --data data --odds odds --minutes 48
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import jra  # noqa: E402
from daily import _fetch_pairs  # noqa: E402
from store import Store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--odds", default="odds")
    ap.add_argument("--minutes", type=float, default=48)
    a = ap.parse_args()
    t0 = time.time()
    src, dst = Store(Path(a.data)), Store(Path(a.odds))
    st_file = Path(a.odds) / "pairs_state.json"
    st = json.loads(st_file.read_text()) if st_file.exists() else {"done": [], "failed": []}
    done = set(st["done"])
    cl = jra.Client()
    dates = sorted((p.stem for p in (Path(a.data) / "enc" / "results").glob("*.json")), reverse=True)
    n_new = 0

    def save(date, day):
        dst.write(f"pairs/{date}.json", day, gz=True)
        st["done"] = sorted(done)
        st_file.write_text(json.dumps(st), encoding="utf-8")

    for date in dates:
        races = src.read(f"results/{date}.json", {"races": {}})["races"]
        todo = [r for r in races.values() if r["id"] not in done and r.get("odds_cname") and "障害" not in (r.get("name") or "")]
        if not todo:
            continue
        day = dst.read(f"pairs/{date}.json", {"races": {}})
        todo = [r for r in todo if r["id"] not in day["races"]]
        for r in todo:
            if time.time() - t0 > a.minutes * 60:
                save(date, day)
                print(f"時間切れ。今回 {n_new}レース / 合計 {len(done)}レース / アクセス {cl.count}回")
                return
            nos = sorted({x["no"] for x in r["runners"]})
            try:
                tf = cl.get("/JRADB/accessO.html", r["odds_cname"])
                pairs = _fetch_pairs(cl, tf, r["id"], nos)
                if not pairs:
                    raise RuntimeError("4券種へのリンクがありません")
            except RuntimeError as e:
                print("  取得できず", r["id"], e)
                st["failed"].append(r["id"]); done.add(r["id"])
                continue
            day["races"][r["id"]] = {"nos": nos, **pairs}
            done.add(r["id"]); n_new += 1
            if n_new % 100 == 0:
                print(f"  {n_new}レース（{r['id']}）{(time.time()-t0)/60:.0f}分", flush=True)
        save(date, day)
    print(f"すべて取り終えました。今回 {n_new}レース / 合計 {len(done)}レース")


if __name__ == "__main__":
    main()
