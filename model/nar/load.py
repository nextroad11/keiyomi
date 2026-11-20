"""地方競馬：公式CSV（月ごと）を表にまとめる。ばんえい（帯広ば）は除く。"""
import glob, re, numpy as np, pandas as pd

RAW = "/tmp/narraw"


def _t(s):
    """'1:35.0' や '1350'(=1分35秒0) を秒に"""
    if pd.isna(s) or s == "":
        return np.nan
    s = str(s)
    m = re.match(r"(\d+):(\d+)\.(\d)", s)
    if m:
        return int(m[1]) * 60 + int(m[2]) + int(m[3]) / 10
    if s.isdigit():
        v = int(s); return (v // 1000) * 60 + (v % 1000) / 10
    return np.nan


def races():
    out = []
    for f in sorted(glob.glob(f"{RAW}/race/*/*_racelist.csv")):
        d = pd.read_csv(f, dtype=str)
        out.append(d[["競馬場", "競走年月日", "レース番号", "発走時刻", "芝ダート区分", "距離", "天候", "馬場", "頭数", "条件", "レース名", "上がり3F"]])
    d = pd.concat(out)
    d = d[d["競馬場"] != "帯広ば"].copy()
    d.columns = ["venue", "date", "rno", "post", "surface", "dist", "weather", "going", "n", "cond", "name", "race3f"]
    d["rno"] = d.rno.astype(int); d["dist"] = pd.to_numeric(d.dist, errors="coerce"); d["n"] = pd.to_numeric(d.n, errors="coerce")
    d["rid"] = d.date + "-" + d.venue + "-" + d.rno.astype(str).str.zfill(2)
    return d.drop_duplicates("rid")


def runners():
    out = []
    for f in sorted(glob.glob(f"{RAW}/race/*/*_horselist.csv")):
        d = pd.read_csv(f, dtype=str)
        out.append(d[["競馬場", "競走年月日", "レース番号", "枠番", "馬番", "馬名", "生年月日", "性", "齢", "騎手名", "調教師", "負担重量",
                      "馬体重", "馬体重増減", "着順", "タイム", "着差", "上がり3F", "人気"]])
    d = pd.concat(out)
    d = d[d["競馬場"] != "帯広ば"].copy()
    d.columns = ["venue", "date", "rno", "waku", "no", "horse", "born", "sex", "age", "jockey", "trainer", "wt", "body", "body_diff",
                 "pos", "time", "margin", "l3f", "pop"]
    d["rid"] = d.date + "-" + d.venue + "-" + d.rno.astype(int).astype(str).str.zfill(2)
    d["hid"] = d.horse + "|" + d.born.fillna("")
    for c in ("waku", "no", "age", "body", "pop"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["wt"] = pd.to_numeric(d.wt.str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    d["body_diff"] = pd.to_numeric(d.body_diff.str.replace("+", "", regex=False), errors="coerce")
    d["pos"] = pd.to_numeric(d.pos, errors="coerce")
    d["sec"] = d.time.map(_t)
    d["l3f"] = pd.to_numeric(d.l3f, errors="coerce")
    d.loc[d.l3f > 100, "l3f"] = d.loc[d.l3f > 100, "l3f"] / 10  # 念のため（'385' 形式）
    return d


if __name__ == "__main__":
    R = races(); H = runners()
    print(len(R), len(H)); print(R.head(3).T); print(H.head(3).T)
    print(H.time.dropna().head(10).tolist(), H.l3f.describe(), H.sec.describe())
    R.to_pickle("/tmp/nar_races.pkl"); H.to_pickle("/tmp/nar_runners.pkl")
