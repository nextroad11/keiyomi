"""地方競馬（地方競馬全国協会 www.keiba.go.jp）の公式配布データの取り込み。

公式サイトが月ごとに配布している CSV（zip）をそのまま取り込む。
- RaceDataDownload：racelist（レース）・horselist（出走馬と成績）・payback（払戻）。2017年からある。
- OddsDataDownload：全券種の確定オッズ。2026年6月より前は空（配布されていない）。
毎日1回、前月と今月を取り直す（月の途中は日々追加されるため）。それより前の月は一度取れば取り直さない。
アクセスは数秒に1回。

保存先：nar ブランチ enc/race/YYYYMM.json、enc/odds/YYYYMM.json（中身は {ファイル名: CSV本文}。gzip→暗号化）
進み具合：nar ブランチ state.json（取り終えた月だけ）

使い方（GitHub Actions 内）: APP_PASS=... python scraper/nar.py --out nar --from 2017-01
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

sys.path.insert(0, str(Path(__file__).parent))
from store import Store  # noqa: E402

BASE = "https://www.keiba.go.jp/KeibaWeb/DataDownload/"
UA = "keiyomi-bot/1.0 (personal race analysis; GitHub Actions; low rate)"
WAIT = 3.0


def fetch(s, kind, y, m):
    r = s.get(f"{BASE}{kind}?type=monthly&k_year={y}&k_month={m}", timeout=120)
    r.raise_for_status()
    if r.content[:2] != b"PK":
        raise RuntimeError(f"zip ではありません（{len(r.content)}バイト）")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    files = {n: z.read(n).decode("utf-8-sig") for n in z.namelist()}
    rows = sum(max(0, t.count("\n") - 1) for t in files.values())
    return files, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nar")
    ap.add_argument("--from", dest="start", default="2017-01")
    ap.add_argument("--minutes", type=float, default=40)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out); S = Store(out)
    stf = out / "state.json"
    st = json.loads(stf.read_text()) if stf.exists() else {"race": [], "odds": [], "odds_empty": []}
    s = requests.Session(); s.headers.update({"User-Agent": UA, "Accept-Language": "ja"})
    now = dt.datetime.now(ZoneInfo("Asia/Tokyo"))
    cur = now.year * 12 + now.month - 1
    y0, m0 = map(int, a.start.split("-"))
    months = [(k // 12, k % 12 + 1) for k in range(cur, y0 * 12 + m0 - 2, -1)]  # 新しい月から
    recent = {months[0], months[1]}  # 今月と前月は毎回取り直す
    for kind, key, folder in (("RaceDataDownload", "race", "race"), ("OddsDataDownload", "odds", "odds")):
        for y, m in months:
            ym = f"{y:04d}{m:02d}"
            if time.time() - t0 > a.minutes * 60:
                break
            if (ym in st[key] and (y, m) not in recent) or (key == "odds" and ym in st["odds_empty"]):
                continue
            if key == "odds" and ym < "202606" and st["odds_empty"]:
                continue  # 配布の始まりより前
            try:
                files, rows = fetch(s, kind, y, m)
            except Exception as e:  # noqa: BLE001
                print(f"{kind} {ym}: 取得できず {e}"); time.sleep(WAIT); continue
            if rows == 0:
                print(f"{kind} {ym}: 空（配布なし）")
                if key == "odds" and (y, m) not in recent:
                    st["odds_empty"].append(ym)
            else:
                S.write(f"{folder}/{ym}.json", files, gz=True)
                if ym not in st[key]:
                    st[key].append(ym)
                print(f"{kind} {ym}: {rows}行", flush=True)
            stf.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
            time.sleep(WAIT)
    print(f"結果 {len(st['race'])}か月・オッズ {len(st['odds'])}か月")


if __name__ == "__main__":
    main()
