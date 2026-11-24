exec(open("/tmp/audit_live.py").read().split("stats = collections")[0])
wr=[]; allr=[]; info=[]
for D in ("20261115","20261121","20261122","20261123"):
    OL = op(f"/tmp/ko4/dev/live/{D}.json"); FX = op(f"/tmp/ko4/dev/exotic/{D}.json"); RS = op(f"/tmp/kd4/dev/results/{D}.json")["races"]
    for rid, snaps in OL["races"].items():
        res=RS.get(rid); fx=FX["races"].get(rid)
        if not res or not fx: continue
        top=tuple(sorted(x["no"] for x in res["runners"] if x.get("pos") in (1,2,3)))
        f=odds_codec.unpack(sorted(fx["nos"]), fx["trio"], False)
        pay=(res["pay"].get("3連複") or [{}])[0].get("yen")
        for m in ("30","10","5","1"):
            s=snaps.get(m)
            if not s: continue
            a=odds_codec.unpack(sorted(s["nos"]), s["trio"], False)
            if top in a and top in f:
                wr.append((m, np.log(f[top]/a[top])))
                if m=="5": info.append((rid, a[top], f[top], pay/100 if pay else None, s["mins"], sorted(s["nos"])==sorted(fx["nos"])))
            if m=="5":
                ks=[k for k in a if k in f and a[k]>0 and f[k]>0]
                allr += [np.log(f[k]/a[k]) for k in ks]
for m in ("30","10","5","1"):
    v=[x for mm,x in wr if mm==m]; print(m,"分前→確定 当たり組の倍率 中央値", round(float(np.exp(np.median(v))),3), "n",len(v))
print("全組 中央値", round(float(np.exp(np.median(allr))),3))
for x in info[:15]: print(x)
