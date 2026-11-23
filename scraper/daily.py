"""開催日の取得（1回呼ぶごとに「いまやること」だけをやって終わる。1分おきに呼ばれる前提）。

1. 朝：出馬表の開催選択 → 今日の各場のレース一覧 → 各レースの出馬表（前走4回分・オッズ・人気）。
2. 締切前：発走の 30・20・15・10・5・3・1 分前に単勝・複勝オッズ（馬体重つき）。
   枠連・馬連・ワイド・馬単・3連複・3連単は 30・10・5・3・1 分前。遅れて取れなかった時点は「取得不可」として空けておく（あとから埋めない）。
3. 発走後：結果・払戻、確定の単複・3連複・3連単オッズ。

保存先
- data ブランチ enc/live/YYYYMMDD.json   … 予定・出馬表・単複オッズの時系列
- data ブランチ enc/results/YYYYMMDD.json … 結果（過去データと同じ形）
- odds ブランチ enc/live/YYYYMMDD.json   … 3連複・3連単の締切前オッズ
- odds ブランチ enc/exotic/YYYYMMDD.json … 確定の3連複・3連単（過去データと同じ形）
- odds ブランチ enc/pairs/YYYYMMDD.json  … 確定の枠連・馬連・ワイド・馬単
今日の分が全部終わったら（または開催が無い日は）、--done-flag のファイルを作って知らせる。

使い方（GitHub Actions 内）: APP_PASS=... python scraper/daily.py --data data --odds odds --done-flag /tmp/daily_done
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))
import jra  # noqa: E402
import odds_codec  # noqa: E402
from store import Store  # noqa: E402

JST = ZoneInfo("Asia/Tokyo")
ENTRY = ("/JRADB/accessD.html", "pw01dli00/F3")  # 出馬表 開催選択（固定）
TF_MARKS = [30, 20, 15, 10, 5, 3, 1]
EX_MARKS = [30, 10, 5, 3, 1]
RESULT_AFTER = 12  # 発走から何分後に結果を見にいくか
BUDGET = 50  # 1回で使う秒数の目安（次の分にはみ出さない）


def now_jst() -> dt.datetime:
    return dt.datetime.now(JST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--odds", default="odds")
    ap.add_argument("--done-flag", default="/tmp/daily_done")
    ap.add_argument("--date", default=None, help="テスト用 YYYYMMDD")
    ap.add_argument("--preview", action="store_true", help="前日の夜：翌日の出馬表と前日発売のオッズだけ取る")
    a = ap.parse_args()
    t0 = time.time()
    now = now_jst()
    date = a.date or (now + dt.timedelta(days=1 if a.preview else 0)).strftime("%Y%m%d")
    D, O = Store(Path(a.data)), Store(Path(a.odds))
    cl = jra.Client()
    live = D.read(f"live/{date}.json", None)

    # 1. 予定（朝に1回。読めていなければ読み直す）
    if live is None or not live.get("races"):
        top = cl.get(*ENTRY)
        lists = [c for c in jra.links(top, "pw01drl") if c.split("/")[0].endswith(date)]
        if not lists:
            print(f"{date} は開催がありません")
            Path(a.done_flag).write_text("no race")
            return
        live = {"date": date, "lists": lists, "races": {}, "loaded_at": now.isoformat()}
        for lc in lists:
            rl = cl.get("/JRADB/accessD.html", lc)
            for row in _race_rows(rl):
                k = jra.race_key(row["card"])
                if not k or k["date"] != date:
                    continue
                live["races"][k["id"]] = {**k, "post": row["post"], "card_cname": row["card"], "odds_cname": row["odds"],
                                         "list_cname": lc, "card": None, "tf": {}, "ex": [], "result": False}
        print(f"{date}: {len(live['lists'])}場 {len(live['races'])}レース")
        D.write(f"live/{date}.json", live)

    races = live["races"]
    # 出馬表（未取得のものを、発走の早い順に）
    for rid, r in sorted(races.items(), key=lambda kv: kv[1]["post"] or "99"):
        if time.time() - t0 > BUDGET:
            break
        stale = r["card"] is not None and not a.preview and r.get("card_day", date) != now.strftime("%Y%m%d")
        if r["card"] is None or stale or (a.preview and not r["odds_cname"] and _ago(r.get("card_at")) >= PREVIEW_EVERY):
            try:
                html = cl.get("/JRADB/accessD.html", r["card_cname"])
                r["card"] = jra.parse_card(html); r["card_day"] = now.strftime("%Y%m%d"); r["card_at"] = now.isoformat(timespec="seconds")
                odds = [c for c in jra.links(html, "pw151ou") if (jra.race_key(c) or {}).get("id") == rid]
                r["odds_cname"] = odds[0] if odds else r["odds_cname"]
            except RuntimeError as e:
                print("出馬表を取得できず", rid, e)

    # 2. 締切前のオッズ
    ex_live = None
    if a.preview:
        _preview_odds(cl, O, date, races, t0)
        D.write(f"live/{date}.json", live)
        print(f"アクセス {cl.count}回 {time.time()-t0:.0f}秒")
        return
    for rid, r in sorted(races.items(), key=lambda kv: kv[1]["post"] or "99"):
        if time.time() - t0 > BUDGET or not r["post"]:
            continue
        post = dt.datetime.strptime(date + r["post"], "%Y%m%d%H:%M").replace(tzinfo=JST)
        mins = (post - now).total_seconds() / 60
        if mins <= 0:
            continue
        due_tf = [m for m in TF_MARKS if mins <= m and str(m) not in r["tf"]]
        due_ex = [m for m in EX_MARKS if mins <= m and m not in r["ex"]]
        if not due_tf and not due_ex:
            continue
        if not r["odds_cname"]:
            continue
        try:
            html = cl.get("/JRADB/accessO.html", r["odds_cname"])
            stamp = now_jst().isoformat(timespec="seconds")
            if due_tf:
                r["tf"][str(min(due_tf))] = {"at": stamp, "mins": round(mins, 1), "odds": {str(k): v for k, v in jra.parse_odds_tf(html).items()}}
                for m in due_tf:
                    r["tf"].setdefault(str(m), None)  # 取れなかった時点
            if due_ex:
                c7, c8 = _exotic_links(html, rid)
                if not c7 or not c8:
                    print("  3連複・3連単へのリンクが見つかりません（この時点は取得不可）", rid)
                trio = jra.parse_odds_trio(cl.get("/JRADB/accessO.html", c7)) if c7 else {}
                tri = jra.parse_odds_trifecta(cl.get("/JRADB/accessO.html", c8)) if c8 else {}
                nos = sorted(no for x in (r["card"] or {}).get("runners", []) if (no := x.get("no")))
                pairs = _fetch_pairs(cl, html, rid, nos)
                if ex_live is None:
                    ex_live = O.read(f"live/{date}.json", {"races": {}})
                ex_live["races"].setdefault(rid, {})[str(min(due_ex))] = {
                    "at": stamp, "mins": round(mins, 1), "nos": nos,
                    "trio": odds_codec.pack(nos, trio, False), "tri": odds_codec.pack(nos, tri, True), "pairs": pairs}
                r["ex"] += due_ex
            print(f"{rid} 発走{mins:.0f}分前 単複{'○' if due_tf else '-'} 3連{'○' if due_ex else '-'}")
        except RuntimeError as e:
            print("オッズを取得できず", rid, e)

    # 3. 結果と確定オッズ
    res_day = ex_day = pr_day = None
    live.pop("_res", None)
    for rid, r in sorted(races.items(), key=lambda kv: kv[1]["post"] or "99"):
        if r["result"] or not r["post"] or time.time() - t0 > BUDGET:
            continue
        post = dt.datetime.strptime(date + r["post"], "%Y%m%d%H:%M").replace(tzinfo=JST)
        if (now - post).total_seconds() / 60 < RESULT_AFTER:
            continue
        try:
            rl = cl.get("/JRADB/accessD.html", r["list_cname"])
            sde = [c for c in jra.links(rl, "pw01sde") if (jra.race_key(c) or {}).get("id") == rid]
            if not sde:  # 出馬表の一覧に結果へのリンクが無ければ、レース結果の一覧から探す
                if "_res" not in live:
                    live["_res"] = {}
                    top = cl.get("/JRADB/accessS.html", "pw01sli00/AF")
                    for lc in jra.links(top, "pw01srl"):
                        if lc.split("/")[0].endswith(date):
                            for c in jra.links(cl.get("/JRADB/accessS.html", lc), "pw01sde"):
                                k = jra.race_key(c)
                                if k:
                                    live["_res"][k["id"]] = c
                sde = [live["_res"][rid]] if rid in live["_res"] else []
            if not sde:
                continue  # まだ確定していない
            race = jra.parse_result(cl.get("/JRADB/accessS.html", sde[0]))
            race.update({k: r[k] for k in ("id", "jcd", "venue", "kai", "day")})
            race.setdefault("date", date); race["rno"] = race.get("rno") or r["rno"]
            tf_html = cl.get("/JRADB/accessO.html", race.get("odds_cname") or r["odds_cname"])
            race["odds_final"] = {str(k): v for k, v in jra.parse_odds_tf(tf_html).items()} or None
            c7, c8 = _exotic_links(tf_html, rid)
            trio = jra.parse_odds_trio(cl.get("/JRADB/accessO.html", c7)) if c7 else {}
            tri = jra.parse_odds_trifecta(cl.get("/JRADB/accessO.html", c8)) if c8 else {}
            nos = sorted({x["no"] for x in race["runners"]})
            pairs = _fetch_pairs(cl, tf_html, rid, nos)
            if res_day is None:
                res_day = D.read(f"results/{date}.json", {"races": {}})
                ex_day = O.read(f"exotic/{date}.json", {"races": {}})
                pr_day = O.read(f"pairs/{date}.json", {"races": {}})
            res_day["races"][rid] = race
            pr_day["races"][rid] = {"nos": nos, **pairs}
            ex_day["races"][rid] = {"nos": nos, "trio": odds_codec.pack(nos, trio, False), "tri": odds_codec.pack(nos, tri, True),
                                    "n_trio": len(trio), "n_tri": len(tri)}
            r["result"] = True
            print(f"{rid} 結果・確定オッズ ○")
        except RuntimeError as e:
            print("結果を取得できず", rid, e)

    D.write(f"live/{date}.json", live)
    if ex_live is not None:
        O.write(f"live/{date}.json", ex_live, gz=True)
    if res_day is not None:
        D.write(f"results/{date}.json", res_day)
        O.write(f"exotic/{date}.json", ex_day, gz=True)
        O.write(f"pairs/{date}.json", pr_day, gz=True)
    if races and all(r["result"] for r in races.values()):
        Path(a.done_flag).write_text("all done")
        print(f"{date}: 全{len(races)}レースの結果まで取り終えました")
    print(f"アクセス {cl.count}回 {time.time()-t0:.0f}秒")
    if cl.count >= 3 and cl.ok == 0:
        sys.exit(3)   # この回は1つも取れなかった（JRAに届かない）。ワークフローが続けば別の実行機に切り替える


PREVIEW_EVERY = 30  # 前日発売のオッズを取り直す間隔（分）


def _ago(stamp) -> float:
    """その時刻から何分たったか（無ければとても大きい数）"""
    return 1e9 if not stamp else (now_jst() - dt.datetime.fromisoformat(stamp)).total_seconds() / 60


def _preview_odds(cl, O, date, races, t0):
    """前日の夜：前日発売のオッズを「前日」という時点として残す（発走何分前かも記録。当日の時点とは別枠）。
    買い目の確定（5分前）には使わない。予想を早めに見るためだけ。"""
    ex_live = O.read(f"live/{date}.json", {"races": {}})
    changed = False
    for rid, r in sorted(races.items(), key=lambda kv: kv[1]["post"] or "99"):
        if time.time() - t0 > BUDGET or not r["post"] or not r["odds_cname"]:
            continue
        if _ago((r["tf"].get("前日") or {}).get("at")) < PREVIEW_EVERY:
            continue
        post = dt.datetime.strptime(date + r["post"], "%Y%m%d%H:%M").replace(tzinfo=JST)
        try:
            html = cl.get("/JRADB/accessO.html", r["odds_cname"])
            tf = jra.parse_odds_tf(html)
            if not tf:
                continue  # まだ発売前
            now = now_jst(); stamp = now.isoformat(timespec="seconds"); mins = round((post - now).total_seconds() / 60, 1)
            r["tf"]["前日"] = {"at": stamp, "mins": mins, "odds": {str(k): v for k, v in tf.items()}}
            c7, c8 = _exotic_links(html, rid)
            trio = jra.parse_odds_trio(cl.get("/JRADB/accessO.html", c7)) if c7 else {}
            tri = jra.parse_odds_trifecta(cl.get("/JRADB/accessO.html", c8)) if c8 else {}
            if trio or tri:
                nos = sorted(no for x in (r["card"] or {}).get("runners", []) if (no := x.get("no")))
                ex_live["races"].setdefault(rid, {})["前日"] = {
                    "at": stamp, "mins": mins, "nos": nos, "trio": odds_codec.pack(nos, trio, False), "tri": odds_codec.pack(nos, tri, True)}
                changed = True
            print(f"{rid} 前日発売 単複○ 3連{'○' if trio or tri else '-'}")
        except RuntimeError as e:
            print("前日のオッズを取得できず", rid, e)
    if changed:
        O.write(f"live/{date}.json", ex_live, gz=True)


def _race_rows(html: str):
    from bs4 import BeautifulSoup
    s = BeautifulSoup(html, "html.parser")
    tb = s.find("table")
    for tr in (tb.find_all("tr")[1:] if tb else []):
        card = jra.links(str(tr.find("td", class_="syutsuba") or tr), "pw01dde")
        odds = jra.links(str(tr.find("td", class_="odds") or ""), "pw151ou")
        t = jra._t(tr.find("td", class_="time"))
        import re
        m = re.search(r"(\d+)時(\d+)分", t)
        if card:
            yield {"card": card[0], "odds": odds[0] if odds else None, "post": f"{int(m[1]):02d}:{int(m[2]):02d}" if m else None}


PAIR_PAGES = {"waku": "pw153ou", "umaren": "pw154ou", "wide": "pw155ou", "umatan": "pw156ou"}


def _fetch_pairs(cl, tf_html: str, rid: str, nos: list[int]) -> dict:
    """枠連・馬連・ワイド・馬単のオッズ（取れなかった券種は入れない＝取得不可）"""
    out = {}
    for kind, pre in PAIR_PAGES.items():
        c = [x for x in jra.links(tf_html, pre) if (jra.race_key(x) or {}).get("id") == rid]
        if not c:
            continue
        try:
            out[kind] = odds_codec.pack_pairs(nos, jra.parse_odds_pairs(cl.get("/JRADB/accessO.html", c[0])), kind)
        except RuntimeError as e:
            print("  取得できず", rid, kind, e)
    return out


def _exotic_links(tf_html: str, rid: str):
    c7 = [c for c in jra.links(tf_html, "pw157ou") if (jra.race_key(c) or {}).get("id") == rid]
    c8 = [c for c in jra.links(tf_html, "pw158ou") if (jra.race_key(c) or {}).get("id") == rid]
    return (c7[0] if c7 else None), (c8[0] if c8 else None)


if __name__ == "__main__":
    main()
