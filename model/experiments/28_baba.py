"""第10回：馬場情報（芝クッション値・含水率）を入れて当てはまりと回収率が上がるか。
学習 2022-03〜2025-05、基準決め 2025-06〜2026-05（ここで判断）、採点 2026-06〜（1回だけ）。"""
import sys, json, gzip, pathlib, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/keiyomi/model"); sys.path.insert(0, "/home/claude/keiyomi/scraper")
import pl, prod, exotic as X
from crypt import unseal
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
priv = X25519PrivateKey.from_private_bytes(bytes.fromhex(pathlib.Path("~/.keiyomi/probe_priv.hex").expanduser().read_text().strip()))
def op(p):
    raw = unseal(priv, json.loads(pathlib.Path(p).read_text()))
    if raw[:2] == b"\x1f\x8b": raw = gzip.decompress(raw)
    return json.loads(raw)
B = {}
for f in sorted(pathlib.Path("/tmp/kd4/dev/baba").glob("*.json")): B.update(op(f))
print("馬場情報", len(B), "日", flush=True)
df = pd.read_pickle("/tmp/ten/df.pkl")
key = df.date + "-" + df.jcd.astype(str).str.zfill(2)
for k in ("cushion", "turf_goal", "dirt_goal"):
    df[k] = key.map(lambda x: (B.get(x) or {}).get(k))
turf = df.surface == "芝"
df["moist"] = np.where(turf, df.turf_goal, df.dirt_goal)
# 場ごとに標準化（場によって含水率の基準が違う）。基準は学習期間だけで作る
tr_mask = (df.date >= "20220301") & (df.date <= "20250531")
for k in ("cushion", "moist"):
    g = df[tr_mask].groupby(["jcd", "surface"])[k].agg(["mean", "std"])
    m = df.set_index(["jcd", "surface"]).index.map(lambda i: g["mean"].get(i, np.nan)); s = df.set_index(["jcd", "surface"]).index.map(lambda i: g["std"].get(i, np.nan))
    df[k + "_c"] = ((df[k] - np.asarray(m)) / np.asarray(s)).clip(-3, 3)
df["cushion_c"] = np.where(turf, df.cushion_c, 0); df = df.fillna({"cushion_c": 0, "moist_c": 0})
print("馬場情報あり（学習期間）", f"{df[tr_mask].cushion.notna().mean():.0%}（芝のクッション）", f"{df[tr_mask].moist.notna().mean():.0%}（含水率）", flush=True)
# ① 交互作用：今日の馬場 × 先行力・上がり・指数
for a in ("cushion_c", "moist_c"):
    for b in ("z_front", "z_f3_best", "z_fig_best"):
        df[f"{a}*{b}"] = df[a] * df[b]
# ② その馬の「似た馬場」での走り：過去走の指数のうち、今日と同じ側（硬い/軟らかい・乾き/湿り）の平均 − 全体の平均（前走まで）
df = df.sort_values(["hid", "dt"]).reset_index(drop=True)
side = np.sign(np.where(turf, df.cushion_c, -df.moist_c))  # ＋＝硬い/乾き（速い馬場）
df["side"] = np.where((df.cushion.isna() & turf) | (df.moist.isna() & ~turf), 0, side)
g = df.groupby("hid")
fig = df.fig.where(df.side != 0)
for s_, nm in ((1, "fast"), (-1, "slow")):
    v = fig.where(df.side == s_)
    df[f"sum_{nm}"] = v.fillna(0).groupby(df.hid).cumsum() - v.fillna(0); df[f"n_{nm}"] = v.notna().astype(int).groupby(df.hid).cumsum() - v.notna().astype(int)
fm = df.fig_mean
aff = np.where(df.side > 0, df.sum_fast / df.n_fast.clip(lower=1) - fm, np.where(df.side < 0, df.sum_slow / df.n_slow.clip(lower=1) - fm, 0))
nn = np.where(df.side > 0, df.n_fast, np.where(df.side < 0, df.n_slow, 0))
df["aff"] = np.where(nn > 0, aff * nn / (nn + 2), 0)  # 回数が少ないほど0に寄せる
df["z_aff"] = df.groupby("rid").aff.transform(lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1)).fillna(0).clip(-4, 4)
INT = [c for c in df.columns if "*" in c]
VA, TE = ("20250601", "20260531"), ("20260601", "20261131")
tr = df[tr_mask]; va = df[(df.date >= VA[0]) & (df.date <= VA[1])]; te = df[(df.date >= TE[0]) & (df.date <= TE[1])]
res = {}
for nm, cols in (("今の形", []), ("+交互作用", INT), ("+似た馬場での走り", ["z_aff"]), ("+両方", INT + ["z_aff"])):
    C = ["logp_mkt"] + prod.Z + cols
    W = pl.fit(pl.pack(tr, C), l2=3, stages=3)
    nl = pl._loglik(W, pl.pack(va, C), 0, 3)[0] / va.rid.nunique()
    n1 = pl._loglik(W, pl.pack(va, C), 0, 1)[0] / va.rid.nunique()
    res[nm] = (W, C, nl); print(nm, "基準決め NLL 3着まで", round(nl, 4), "1着", round(n1, 4), "| 追加分の重み", dict(zip(cols, np.round(W[-len(cols):], 3))) if cols else "", flush=True)
pickle.dump((res, df[["rid", "date"] + INT + ["z_aff"]]), open("/tmp/baba/res.pkl", "wb"))
