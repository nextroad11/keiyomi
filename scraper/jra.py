"""JRA公式サイト（www.jra.go.jp/JRADB/）の取得と読み取り。

- ページは cname を POST して開く。cname の末尾（/XX）はサイトが付ける検査用の文字なので、自分では作らず、
  必ずページ内のリンクからたどる。
- アクセスは WAIT 秒に1回まで。
- 読み取れない項目は None（推測で埋めない）。
"""
from __future__ import annotations

import re
import time

import requests
from bs4 import BeautifulSoup

BASE = "https://www.jra.go.jp"
UA = "keiyomi-bot/1.0 (personal race analysis; GitHub Actions; 1 req/sec)"
WAIT = 1.1
VENUES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
          "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
VENUE_CODE = {v: k for k, v in VENUES.items()}
CNAME_RE = re.compile(r"(pw\w+?/[0-9A-F]{2})")


class Client:
    def __init__(self, wait: float = WAIT):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "ja"})
        self.wait, self.last, self.count = wait, 0.0, 0

    def get(self, path: str, cname: str) -> str:
        """path は /JRADB/accessS.html など。失敗したら2回までやり直す。"""
        for i in range(3):
            dt = time.time() - self.last
            if dt < self.wait:
                time.sleep(self.wait - dt)
            self.last = time.time(); self.count += 1
            try:
                r = self.s.post(BASE + path, data={"cname": cname}, timeout=30)
                if r.status_code == 200 and len(r.content) > 2000:
                    return r.content.decode("cp932", errors="replace")
            except requests.RequestException:
                pass
            time.sleep(5 * (i + 1))
        raise RuntimeError(f"取得できません {path} {cname}")


# ---------- 共通 ----------
def _t(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ")).strip() if el is not None else ""


def _num(s, typ=float):
    if s is None:
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", str(s))
    if not m:
        return None
    try:
        return typ(m.group(0).replace(",", ""))
    except ValueError:
        return None


def links(html: str, kind: str) -> list[str]:
    """ページ内の cname のうち、kind（例 'pw01sde'）で始まるものを出てきた順に（重複なし）。"""
    out = []
    for c in CNAME_RE.findall(html):
        if c.startswith(kind) and c not in out:
            out.append(c)
    return out


def race_key(cname: str) -> dict | None:
    """pw01sde1005202605030120261114/87 → 場・回・日・R・日付。出馬表・オッズの cname も同じ並び。"""
    m = re.search(r"(\d{20,})Z?\d*/[0-9A-F]{2}$", cname)
    if not m:
        return None
    d = m.group(1)[-20:]
    jcd, year, kai, day, rno, date = d[0:2], d[2:6], d[6:8], d[8:10], d[10:12], d[12:20]
    return {"id": f"{date}-{jcd}-{rno}", "date": date, "jcd": jcd, "venue": VENUES.get(jcd),
            "kai": int(kai), "day": int(day), "rno": int(rno)}


def parse_header(s: BeautifulSoup) -> dict:
    h = s.find(class_="race_header")
    if h is None:
        return {}
    out = {}
    d = _t(h.find(class_="date"))
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", d)
    if m:
        out["date"] = f"{int(m[1]):04d}{int(m[2]):02d}{int(m[3]):02d}"
    m = re.search(r"(\d+)時(\d+)分", _t(h.find(class_="time")))
    out["post"] = f"{int(m[1]):02d}:{int(m[2]):02d}" if m else None
    w = h.find(class_="weather")
    out["weather"] = _t(w.find(class_="txt")) if w else None
    out["surface"] = out["going"] = None
    baba = h.find(class_="baba")
    if baba:
        for li in baba.find_all("li"):
            cap = _t(li.find(class_="cap"))
            if cap in ("芝", "ダート", "障害"):
                out["surface"] = cap if cap != "障害" else out.get("surface") or "障害"
                out["going"] = _t(li.find(class_="txt")) or None
    rn = h.find(class_="race_number")
    img = rn.find("img") if rn else None
    out["rno"] = _num(img.get("alt"), int) if img else None
    out["name"] = _t(h.find(class_="race_name")) or None
    typ = h.find(class_="type")
    if typ:
        out["age"] = _t(typ.find(class_="category")) or None
        out["cls"] = _t(typ.find(class_="class")) or None
        out["rule"] = _t(typ.find(class_="rule")) or None
        out["wcond"] = _t(typ.find(class_="weight")) or None
        c = typ.find(class_="course")
        if c:
            out["distance"] = _num(_t(c), int)
            det = _t(c.find(class_="detail"))
            out["course_detail"] = det.strip("（）") or None
            if out.get("surface") is None:
                for k in ("芝", "ダート", "障害"):
                    if k in det:
                        out["surface"] = k
                        break
    return out


# ---------- レース結果 ----------
def parse_result(html: str) -> dict:
    s = BeautifulSoup(html, "html.parser")
    race = parse_header(s)
    tb = s.find("table", class_="basic")
    runners = []
    if tb:
        for tr in tb.find_all("tr")[1:]:
            td = {(c.get("class") or [""])[0]: c for c in tr.find_all(["td", "th"])}
            if "num" not in td:
                continue
            a = td.get("horse").find("a") if td.get("horse") else None
            hid = re.search(r"pw01dud\d{2}(\d+)/", a["href"]) if a and a.get("href") else None
            jk = td.get("jockey"); ja = jk.find("a") if jk else None
            jid = re.search(r"pw04kmk(\d+)/", ja.get("onclick", "")) if ja else None
            hw = _t(td.get("h_weight"))
            bw = re.match(r"(\d+)\s*\(([-+]?\d+|計不|前計不)?\)?", hw)
            waku = td.get("waku").find("img") if td.get("waku") else None
            place = _t(td.get("place"))
            runners.append({
                "pos": _num(place, int) if place.isdigit() else None,
                "pos_text": place or None,               # 中止・除外・取消などはここに文字で残る
                "waku": _num(waku.get("alt"), int) if waku else None,
                "no": _num(_t(td["num"]), int),
                "name": _t(td.get("horse")) or None,
                "hid": hid.group(1) if hid else None,
                "sexage": _t(td.get("age")) or None,
                "weight": _num(_t(td.get("weight"))),
                "jockey": _t(td.get("jockey")) or None,
                "jid": jid.group(1) if jid else None,
                "time": _t(td.get("time")) or None,
                "margin": _t(td.get("margin")) or None,
                "corner": _t(td.get("corner")) or None,
                "f3": _num(_t(td.get("f_time"))),
                "body": int(bw.group(1)) if bw else None,
                "body_diff": _num(bw.group(2), int) if bw and bw.group(2) and bw.group(2)[-1].isdigit() else None,
                "trainer": _t(td.get("trainer")) or None,
                "pop": _num(_t(td.get("pop")), int),
            })
    race["runners"] = runners
    race["n"] = len(runners)
    # ラップ・上がり
    td = s.find(class_="result_time_data")
    race["laps"] = race["last"] = None
    if td:
        for tr in td.find_all("tr"):
            k, v = _t(tr.find("th")), _t(tr.find("td"))
            if k == "ハロンタイム":
                race["laps"] = [float(x) for x in re.findall(r"\d+\.\d", v)]
            elif k == "上り":
                race["last"] = v
    cp = s.find(class_="result_corner_place")
    race["corners"] = {_t(tr.find("th")): _t(tr.find("td")) for tr in cp.find_all("tr")} if cp else {}
    # 払戻
    pay = {}
    ra = s.find(class_="refund_area")
    if ra:
        for li in ra.find_all("li", class_=True):
            dt = _t(li.find("dt"))
            if not dt:
                continue
            rows = []
            for ln in li.find_all(class_="line"):
                combo, yen, pop = _t(ln.find(class_="num")), _num(_t(ln.find(class_="yen")), int), _num(_t(ln.find(class_="pop")), int)
                if combo and yen:
                    rows.append({"combo": combo, "yen": yen, "pop": pop})
            if rows:
                pay[dt.replace("３", "3")] = rows
    race["pay"] = pay
    rel = links(html, "pw151ou")
    race["odds_cname"] = rel[0] if rel else None
    card = links(html, "pw01dde")
    race["card_cname"] = card[0] if card else None
    return race


# ---------- 単勝・複勝オッズ ----------
def parse_odds_tf(html: str) -> dict:
    s = BeautifulSoup(html, "html.parser")
    tb = s.find("table", class_="basic")
    out = {}
    if not tb:
        return out
    for tr in tb.find_all("tr")[1:]:
        num = tr.find("td", class_="num")
        if not num:
            continue
        no = _num(_t(num), int)
        tan = _num(_t(tr.find("td", class_="odds_tan")))
        fk = tr.find("td", class_="odds_fuku")
        fmin = _num(_t(fk.find(class_="min"))) if fk and fk.find(class_="min") else None
        fmax = _num(_t(fk.find(class_="max"))) if fk and fk.find(class_="max") else None
        hw = _t(tr.find("td", class_="h_weight"))
        bw = re.match(r"(\d+)\s*\(?([-+]?\d+)?", hw)
        out[no] = {"win": tan, "pmin": fmin, "pmax": fmax,
                   "body": int(bw.group(1)) if bw else None,
                   "body_diff": _num(bw.group(2), int) if bw and bw.group(2) else None}
    return out


# ---------- 出馬表 ----------
def parse_card(html: str) -> dict:
    s = BeautifulSoup(html, "html.parser")
    race = parse_header(s)
    tb = s.find("table", class_="basic")
    runners = []
    if tb:
        for tr in tb.find_all("tr")[1:]:
            num = tr.find("td", class_="num")
            if not num:
                continue
            h = tr.find("td", class_="horse")
            a = h.find("div", class_="name").find("a") if h and h.find("div", class_="name") else None
            hid = re.search(r"pw01dud\d{2}(\d+)/", a["href"]) if a and a.get("href") else None
            jk = tr.find("td", class_="jockey")
            for img in (jk.find_all("img") if jk else []):
                img.decompose()
            jt = _t(jk)
            m = re.match(r"(\S+?)/(\S+)\s+([\d.]+)\s*kg\s+(.*)", jt)
            ja = jk.find("a") if jk else None
            jid = re.search(r"pw04kmk(\d+)/", ja.get("onclick", "")) if ja else None
            odds = h.find(class_="odds_line") if h else None
            past = []
            for p in tr.find_all("td", class_="past"):
                if not _t(p):
                    past.append(None); continue
                d = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", _t(p.find(class_="date")))
                pl = p.find(class_="place_line")
                dist = _t(p.find(class_="dist"))
                corners = [_num(_t(li), int) for li in p.select(".corner_list li")]
                fin = p.find(class_="fin")
                link = p.find("a")
                past.append({
                    "date": f"{int(d[1]):04d}{int(d[2]):02d}{int(d[3]):02d}" if d else None,
                    "venue": _t(p.find(class_="rc")) or None,
                    "name": _t(p.find(class_="name")) or None,
                    "pos": _num(_t(pl.find(class_="place")), int) if pl else None,
                    "pos_text": _t(pl.find(class_="place")) if pl else None,
                    "n": _num(_t(p.find(class_="max")), int),
                    "no": _num(_t(p.find(class_="gate")), int),
                    "pop": _num(_t(p.find(class_="pop")), int),
                    "jockey": _t(p.find(class_="jockey")) or None,
                    "weight": _num(_t(p.find(class_="weight"))),
                    "distance": _num(dist, int),
                    "surface": ("芝" if "芝" in dist else "ダート" if "ダ" in dist else "障害" if "障" in dist else None),
                    "time": _t(p.find(class_="time")) or None,
                    "going": _t(p.find(class_="condition")) or None,
                    "body": _num(_t(p.find(class_="h_weight")), int),
                    "corners": corners,
                    "f3": _num(_t(p.find(class_="f3")).replace("3F", "")),
                    "winner": _t(fin).split("(")[0].strip() if fin else None,
                    "diff": _num(_t(fin.find(class_="time"))) if fin and fin.find(class_="time") else None,
                    "result_cname": (CNAME_RE.search(link.get("href", "")) or [None])[0] if link else None,
                })
            waku = tr.find("td", class_="waku")
            wimg = waku.find("img") if waku else None
            runners.append({
                "no": _num(_t(num), int),
                "waku": _num(wimg.get("alt"), int) if wimg else None,
                "name": _t(a) or None,
                "hid": hid.group(1) if hid else None,
                "odds": _num(_t(odds.find(class_="num"))) if odds else None,
                "pop": _num(_t(odds.find(class_="pop_rank")), int) if odds else None,
                "sexage": m.group(1) if m else None,
                "weight": float(m.group(3)) if m else None,
                "jockey": m.group(4).strip() if m else None,
                "jid": jid.group(1) if jid else None,
                "trainer": _t(h.find(class_="trainer")).split("(")[0].strip() if h else None,
                "sire": _t(h.find(class_="sire")).replace("父：", "").strip() if h else None,
                "past": past,
            })
    race["runners"] = runners
    race["n"] = len(runners)
    return race
