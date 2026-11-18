"""JRA公式サイトの下見（一回きり）。

GitHub Actions から JRA のページに届くか、出馬表・オッズ・結果のページがどう作られているかを確かめる。
- アクセスは1.2秒に1回まで、最大60ページ。
- 取ったページは keys/probe_pub.txt の公開鍵で封をして保存する（平文はコミットしない）。
  開けるのは、その公開鍵と組の秘密鍵を持つ Claude の作業環境だけ。
- 平文で残すのは「どのURLに何を送って、何番が返ったか」の一覧だけ。

使い方: python scraper/probe.py keys/probe_pub.txt 出力フォルダ [入口のJSON]
  入口のJSON（任意）: {"seeds": [["POST", "/JRADB/accessS.html", "cname"], ...], "max_pages": 30, "max_depth": 1}
"""
from __future__ import annotations

import io
import json
import re
import sys
import tarfile
import time
from collections import Counter, deque
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from crypt import seal  # noqa: E402

BASE = "https://www.jra.go.jp"
H = {"User-Agent": "keiyomi-bot/1.0 (personal race analysis; GitHub Actions; 1 req/sec)",
     "Accept-Language": "ja"}
WAIT = 1.2
MAX_PAGES = 60
PER_KIND = 10  # accessD / accessO などの種類ごとの上限

# 入口。cname は JRA のページ内のリンクで使われている値（違っていれば下のリンク探索で見つける）
SEEDS = [
    ("GET", "/", None),
    ("GET", "/keiba/", None),
    ("POST", "/JRADB/accessD.html", "pw01dli00/F3"),  # 出馬表
    ("POST", "/JRADB/accessO.html", "pw15oli00/6D"),  # オッズ
    ("POST", "/JRADB/accessS.html", "pw01sli00/AF"),  # レース結果
    ("POST", "/JRADB/accessH.html", "pw01hli00/03"),  # 払戻金
]

RE_DO = re.compile(r"doAction\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]")
RE_GETC = re.compile(r"href=['\"](/JRADB/access\w\.html)\?CNAME=([^'\"&]+)", re.I)


def decode(b: bytes, ctype: str) -> str:
    m = re.search(rb"charset=['\"]?([\w_-]+)", b[:2000]) or re.search(r"charset=([\w_-]+)", ctype or "")
    enc = (m.group(1).decode() if isinstance(m.group(1), bytes) else m.group(1)) if m else "shift_jis"
    try:
        return b.decode(enc, errors="replace")
    except LookupError:
        return b.decode("cp932", errors="replace")


def main():
    pub = Path(sys.argv[1]).read_text().strip()
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    seeds, max_pages, max_depth, follow = SEEDS, MAX_PAGES, 2, re.compile(".")
    if len(sys.argv) > 3 and Path(sys.argv[3]).exists():
        cfg = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
        seeds = [tuple(x) for x in cfg["seeds"]]
        max_pages, max_depth = cfg.get("max_pages", MAX_PAGES), cfg.get("max_depth", 2)
        follow = re.compile(cfg.get("follow", "."))
    s = requests.Session(); s.headers.update(H)
    q = deque((m, u, c, 0) for m, u, c in seeds)
    seen, kinds, log, files = set(), Counter(), [], {}
    while q and len(log) < max_pages:
        method, url, cname, depth = q.popleft()
        key = (url, cname)
        if key in seen:
            continue
        kind = url.split("?")[0].rstrip("/").split("/")[-1]
        if depth > 0 and kinds[kind] >= (PER_KIND if cname else 3):
            continue
        seen.add(key); kinds[kind] += 1
        try:
            full = url if url.startswith("http") else BASE + url
            if method == "POST":
                r = s.post(full, data={"cname": cname}, timeout=20)
            else:
                r = s.get(full, timeout=20)
            status, body, ctype = r.status_code, r.content, r.headers.get("content-type", "")
        except Exception as e:  # 届かない・タイムアウト
            status, body, ctype = f"ERR {type(e).__name__}", b"", ""
        n = len(log)
        name = f"{n:02d}_{re.sub(r'[^A-Za-z0-9_]', '', kind.replace('.html', '')) or 'index'}.html"
        files[name] = body
        text = decode(body, ctype) if body else ""
        title = (re.search(r"<title>(.*?)</title>", text, re.S) or [None, ""])[1].strip()[:60]
        found = RE_DO.findall(text) + RE_GETC.findall(text)
        log.append({"n": n, "method": method, "url": url, "cname": cname, "status": status,
                    "bytes": len(body), "title": title, "links": len(found), "depth": depth})
        print(n, method, url, cname, status, len(body), title, f"links={len(found)}", flush=True)
        if depth < max_depth and url.startswith("https://www.keiba.go.jp"):
            import html as _h
            for href in re.findall(r'href="([^"#]+)"', text):
                href = _h.unescape(href)
                if href.startswith("/"):
                    href = "https://www.keiba.go.jp" + href
                if href.startswith("https://www.keiba.go.jp/KeibaWeb/") and follow.search(href) and (href, None) not in seen:
                    q.append(("GET", href, None, depth + 1))
        if depth < max_depth:
            for u, c in found:
                if u.startswith("/JRADB/") and follow.search(c):
                    q.append(("POST", u, c, depth + 1))
        time.sleep(WAIT)

    # 一覧（平文）と、ページ本体（封をしたもの）
    (out / "index.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, body in files.items():
            info = tarfile.TarInfo(name); info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    (out / "pages.sealed.json").write_text(json.dumps(seal(pub, buf.getvalue())), encoding="utf-8")
    ok = sum(1 for x in log if x["status"] == 200)
    print(f"取得 {len(log)}ページ（200: {ok}）")


if __name__ == "__main__":
    main()
