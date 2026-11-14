"""過去のレース結果の取り込み（小分けにして途中で保存）。

今月から1か月ずつさかのぼり、各開催日・各レースの結果ページと、確定の単勝・複勝オッズを取る。
- 1回の実行は --minutes 分まで。続きは次の実行が state.json から再開する。
- 結果は日付ごとに enc/results/YYYYMMDD.json（暗号化）へ。
- state.json（平文）には、どのレースを取り終えたかの ID だけを書く（レースの中身は書かない）。

使い方（GitHub Actions 内）: APP_PASS=... python scraper/history.py --data data --months 24 --minutes 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import jra  # noqa: E402
from store import Store  # noqa: E402

# 入口（JRAのページ内リンクに出てくる固定の cname）
SEARCH = ("/JRADB/accessS.html", "pw01skl00999999/B3")  # 過去レース結果検索（今月）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--minutes", type=float, default=40)
    a = ap.parse_args()
    t0 = time.time()
    data = Path(a.data)
    st_file = data / "history" / "state.json"
    st = json.loads(st_file.read_text()) if st_file.exists() else {"months": [], "done": [], "odds_ok": True, "next": None}
    done = set(st["done"])
    store = Store(data)
    cl = jra.Client()
    cache: dict[str, dict] = {}
    dirty: set[str] = set()

    def day(date):
        if date not in cache:
            cache[date] = store.read(f"results/{date}.json", {"races": {}})
        return cache[date]

    def flush():
        for d in sorted(dirty):
            store.write(f"results/{d}.json", cache[d])
        dirty.clear()
        st["done"] = sorted(done)
        st_file.parent.mkdir(parents=True, exist_ok=True)
        st_file.write_text(json.dumps(st, ensure_ascii=False, indent=0), encoding="utf-8")

    def out_of_time():
        return time.time() - t0 > a.minutes * 60

    # 月のページ（今月 → さかのぼる）
    month = st["next"] or SEARCH[1]
    n_new = 0
    while len(st["months"]) < a.months and not out_of_time():
        html = cl.get("/JRADB/accessS.html", month)
        lists = jra.links(html, "pw01srl1")
        prev = [c for c in jra.links(html, "pw01skl1")]
        months_seen = [k["date"][:6] for k in (jra.race_key(c) for c in lists) if k]
        ym = month[9:15] if month[9:15].isdigit() and month[9:15] != "999999" else (
            max(set(months_seen), key=months_seen.count) if months_seen else None)
        print(f"月 {ym}: 開催 {len(lists)}", flush=True)
        finished = True
        for lc in lists:
            if out_of_time():
                finished = False
                break
            k = jra.race_key(lc)
            if k and ym and k["date"][:6] != ym:
                continue  # 他の月へのリンク
            try:
                rhtml = cl.get("/JRADB/accessS.html", lc)
            except RuntimeError as e:
                print("  開催のページを取得できず", lc, e)
                finished = False
                break
            for rc in jra.links(rhtml, "pw01sde1"):
                rk = jra.race_key(rc)
                if not rk or rk["id"] in done:
                    continue
                if out_of_time():
                    finished = False
                    break
                try:
                    race = jra.parse_result(cl.get("/JRADB/accessS.html", rc))
                except RuntimeError as e:
                    print("  取得できず", rk["id"], e)
                    st.setdefault("failed", []).append(rk["id"])
                    continue
                race.update({k2: rk[k2] for k2 in ("id", "jcd", "venue", "kai", "day")})
                race.setdefault("date", rk["date"])
                race["rno"] = race.get("rno") or rk["rno"]
                race["odds_final"] = None
                if st["odds_ok"] and race.get("odds_cname"):
                    try:
                        odds = jra.parse_odds_tf(cl.get("/JRADB/accessO.html", race["odds_cname"]))
                        race["odds_final"] = {str(k3): v for k3, v in odds.items()} or None
                    except RuntimeError:
                        pass
                    st["odds_fail"] = 0 if race["odds_final"] else st.get("odds_fail", 0) + 1
                    if st["odds_fail"] >= 5:  # 5レース続けて無ければ、古いオッズは残っていないと判断
                        print(f"  確定オッズのページが {rk['date']} 付近から無くなっています。これより前はオッズを取りません")
                        st["odds_ok"] = False
                day(rk["date"])["races"][rk["id"]] = race
                dirty.add(rk["date"]); done.add(rk["id"]); n_new += 1
                if n_new % 50 == 0:
                    flush()
                    print(f"  {n_new}レース（{rk['id']}）{(time.time()-t0)/60:.0f}分", flush=True)
            if not finished:
                break
        if not finished:
            st["next"] = month  # 同じ月の続きから
            break
        st["months"].append(ym)
        # 1か月前へ
        older = [c for c in prev if ym and c[9:15] < ym]
        older.sort(key=lambda c: c[9:15], reverse=True)
        if not older:
            print("これより前の月へのリンクがありません")
            break
        month = st["next"] = older[0]
    flush()
    print(f"今回 {n_new}レース / 合計 {len(done)}レース / 取り終えた月 {len(st['months'])} / アクセス {cl.count}回")


if __name__ == "__main__":
    main()
