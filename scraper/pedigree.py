"""血統（父・母の父）の取り込み。

1) 結果ページ（2022-03 以降、新しい月から）を回って、出走馬のページへのリンク（pw01dud…）を集める
2) 馬のページから 父・母の父・性別・生年月日 を取る（1頭1回）
保存：data ブランチ enc/pedigree/horses.json（暗号化）{馬ID: {"sire","bms","sex","born"}}
進み具合：pedigree/state.json（取り終えた月・レースID、馬ID→ページ名。中身は書かない）
1回の実行は --minutes 分まで。続きは次の実行が再開する。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
import jra  # noqa: E402
from store import Store  # noqa: E402

SEARCH = "pw01skl00999999/B3"
HREF = re.compile(r"pw01dud\d{2}(\d+)/[0-9A-F]{2}")


def horse_links(html: str) -> dict:
    """{馬ID: cname}（ページ内の馬のリンク）"""
    out = {}
    for m in re.finditer(r"(pw01dud\d{2}(\d+)/[0-9A-F]{2})", html):
        out.setdefault(m.group(2), m.group(1))
    return out


def parse_horse(html: str) -> dict:
    s = BeautifulSoup(html, "html.parser")
    out = {}
    for dl in s.select("div.profile dl"):
        dt_, dd = dl.find("dt"), dl.find("dd")
        if not dt_ or not dd:
            continue
        k = dt_.get_text(strip=True); a = dd.find("a"); v = (a or dd).get_text(" ", strip=True)
        if k == "父":
            out["sire"] = v
        elif k == "母の父":
            out["bms"] = v
        elif k == "性別":
            out["sex"] = v
        elif k == "生年月日":
            out["born"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--since", default="202203")
    ap.add_argument("--minutes", type=float, default=40)
    a = ap.parse_args()
    t0 = time.time(); out_of_time = lambda: time.time() - t0 > a.minutes * 60
    D = Store(Path(a.data)); cl = jra.Client()
    sf = Path(a.data) / "pedigree" / "state.json"; sf.parent.mkdir(parents=True, exist_ok=True)
    st = json.loads(sf.read_text()) if sf.exists() else {"months": [], "races": [], "link": {}, "next": None, "finished_links": False}
    races = set(st["races"]); link = st["link"]
    H = D.read("pedigree/horses.json", {})

    def save():
        st["races"] = sorted(races); st["link"] = link
        sf.write_text(json.dumps(st)); D.write("pedigree/horses.json", H)

    # 1) 先に、まだ取っていない馬のページ（新しく走った馬ほど大事なので、リンクを集めた順に）
    def fetch_horses():
        n = 0
        for hid, c in list(link.items()):
            if hid in H:
                continue
            if out_of_time():
                return n
            try:
                H[hid] = parse_horse(cl.get("/JRADB/accessU.html", c)) or {"ng": 1}
            except RuntimeError as e:
                print("馬のページを取れず", hid, e); H[hid] = {"ng": 1}
            n += 1
            if n % 200 == 0:
                save(); print(f"  馬 {n}頭 {(time.time()-t0)/60:.0f}分", flush=True)
        return n

    nh = fetch_horses()
    # 2) 結果ページを回ってリンクを集める（新しい月から）
    month = st["next"] or SEARCH
    while not st["finished_links"] and not out_of_time():
        html = cl.get("/JRADB/accessS.html", month)
        lists = jra.links(html, "pw01srl1")
        ym = month[9:15] if month[9:15].isdigit() and month[9:15] != "999999" else None
        print("月", ym or "今月", "開催", len(lists), flush=True)
        stop = False
        for lc in lists:
            k = jra.race_key(lc)
            if ym and k and k["date"][:6] != ym:
                continue
            rh = cl.get("/JRADB/accessS.html", lc)
            for rc in jra.links(rh, "pw01sde1"):
                rk = jra.race_key(rc)
                if not rk or rk["id"] in races:
                    continue
                if out_of_time():
                    stop = True; break
                try:
                    link.update({h: c for h, c in horse_links(cl.get("/JRADB/accessS.html", rc)).items() if h not in link})
                    races.add(rk["id"])
                except RuntimeError as e:
                    print("結果ページを取れず", rk["id"], e)
            if stop:
                break
        if stop:
            st["next"] = month; break
        st["months"].append(ym or "now")
        older = sorted((c for c in jra.links(html, "pw01skl1") if c[9:15].isdigit() and (not ym or c[9:15] < ym)), key=lambda c: c[9:15], reverse=True)
        if not older or older[0][9:15] < a.since:
            st["finished_links"] = True; break
        month = st["next"] = older[0]
        save()
    nh += fetch_horses()
    save()
    print(f"レース {len(races)} / リンク {len(link)}頭 / 血統 {len(H)}頭（今回 {nh}頭） / アクセス {cl.count}回")


if __name__ == "__main__":
    main()
