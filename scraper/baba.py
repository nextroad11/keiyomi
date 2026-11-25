"""JRA の馬場情報（芝のクッション値・芝/ダートの含水率）を、年ごとのページの開催別PDFから取り込む。

保存：data ブランチ enc/baba/YYYY.json（暗号化）。キーは "YYYYMMDD-場コード"、値は
  {"cushion": 芝クッション値, "turf_goal", "turf_4c", "dirt_goal", "dirt_4c"}（取れたものだけ）。
進み具合：baba/state.json（取り終えたPDFの名前だけ）。今年と去年のPDFは毎回取り直す（開催中は更新されるため）。
使い方（GitHub Actions 内）: APP_PASS=... python scraper/baba.py --data data --from 2019
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from store import Store  # noqa: E402

BASE = "https://www.jra.go.jp"
UA = "Mozilla/5.0 (keiyomi research; contact via GitHub nextroad11/keiyomi)"
JCD = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}
WD = "月火水木金土日"
Z2H = str.maketrans("０１２３４５６７８９ＡＢＣＤ．", "0123456789ABCD.")


class Get:
    def __init__(self):
        self.s = requests.Session(); self.s.headers.update({"User-Agent": UA}); self.last = 0.0; self.count = 0

    def __call__(self, path):
        for i in range(3):
            time.sleep(max(0, 1.1 - (time.time() - self.last)))
            self.last = time.time(); self.count += 1
            try:
                r = self.s.get(BASE + path, timeout=30)
                if r.status_code == 200:
                    return r
                if r.status_code in (403, 404):
                    return None
            except requests.RequestException:
                pass
            time.sleep(3 * (i + 1))
        return None


def pdf_text(b: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(b); f.flush()
        return subprocess.run(["pdftotext", "-layout", f.name, "-"], capture_output=True, text=True).stdout.translate(Z2H)


NUM = re.compile(r"(?<![\d:])(\d{1,2}\.\d)(?![\d:])")


def parse(text: str) -> dict:
    """2つの形に対応：①1日1行の表（新しい年）②「第1日・第2日（YYYY年M月D日～D日）」ごとの縦横の表（古い年）"""
    m = re.search(r"(\d{4})年\s*第?\s*\d+\s*回\s*(\S\S)競馬", text)
    if not m or m.group(2) not in JCD:
        return {}
    year, jcd = int(m.group(1)), JCD[m.group(2)]
    out = {}
    # ① 1日1行
    for line in text.splitlines():
        d = re.search(r"(\d{1,2})月\s*(\d{1,2})日\s+(\S)曜日", line)
        if not d:
            continue
        v = NUM.findall(line[d.end():])
        if len(v) == 5:
            key = f"{year}{int(d.group(1)):02d}{int(d.group(2)):02d}-{jcd}"
            out[key] = dict(zip(("cushion", "turf_goal", "turf_4c", "dirt_goal", "dirt_4c"), map(float, v)))
    if out:
        return out
    # ② 週ごとの表
    blocks = re.split(r"\n(?=.*第\s*\d+\s*日.*（\d{4}年)", text)
    for b in blocks:
        h = re.search(r"（(\d{4})年(\d{1,2})月(\d{1,2})日", b)
        wdh = next((l for l in b.splitlines() if "曜日" in l), None)
        if not h or not wdh:
            continue
        start = dt.date(int(h.group(1)), int(h.group(2)), int(h.group(3)))
        wds = [WD.index(c) for c in re.findall(r"(\S)曜日", wdh) if c in WD]
        if not wds:
            continue
        dates = [start + dt.timedelta(days=(w - wds[0]) % 7) for w in wds]
        rows = {}
        kind = None
        for line in b.splitlines():
            vals = NUM.findall(line)
            if "クッション" in line:
                rows["cushion"] = vals; continue
            if "芝" in line and "含水" in line:
                kind = "turf"
            elif "ダート" in line and "含水" in line:
                kind = "dirt"
            if kind and ("ゴール" in line or "4コーナー" in line or "４コーナー" in line):
                rows[f"{kind}_{'goal' if 'ゴール' in line else '4c'}"] = vals
        for k, vals in rows.items():
            # クッション値は金曜に無いことがある：右に寄せる
            vals = vals[-len(dates):] if len(vals) <= len(dates) else vals[:len(dates)]
            for d, v in zip(dates[len(dates) - len(vals):], vals):
                out.setdefault(f"{d:%Y%m%d}-{jcd}", {})[k] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--from", dest="y0", type=int, default=2019)
    a = ap.parse_args()
    D = Store(Path(a.data)); g = Get()
    st_file = Path(a.data) / "baba" / "state.json"; st_file.parent.mkdir(parents=True, exist_ok=True)
    st = json.loads(st_file.read_text()) if st_file.exists() else {"done": []}
    done = set(st["done"]); this = dt.date.today().year
    for y in range(a.y0, this + 1):
        r = g(f"/keiba/baba/archive/{y}.html")
        if r is None:
            print(y, "年のページなし"); continue
        pdfs = sorted(set(re.findall(rf"/keiba/baba/archive/{y}pdf/[a-z]+\d+\.pdf", r.content.decode("utf-8", "replace"))))
        year = D.read(f"baba/{y}.json", {})
        n0 = len(year)
        for p in pdfs:
            if p in done and y < this - 1:
                continue
            rp = g(p)
            if rp is None:
                print("取れず", p); continue
            got = parse(pdf_text(rp.content))
            year.update(got); done.add(p)
            print(p, len(got), "日", flush=True)
        D.write(f"baba/{y}.json", year)
        print(y, "年", len(year), "日（+", len(year) - n0, ")", flush=True)
    st["done"] = sorted(done); st_file.write_text(json.dumps(st))
    print("アクセス", g.count)


if __name__ == "__main__":
    main()
