exec(open("/tmp/audit_live.py").read().split("stats = collections")[0])
import itertools
bad=0; tot=0
for D in ("20261115","20261121","20261122","20261123"):
    OL = op(f"/tmp/ko4/dev/live/{D}.json"); FX = op(f"/tmp/ko4/dev/exotic/{D}.json")
    for rid, snaps in sorted(OL["races"].items()):
        fx=FX["races"].get(rid); s=snaps.get("5") or snaps.get("3")
        if not fx or not s: continue
        a=odds_codec.unpack(sorted(s["nos"]), s["trio"], False); f=odds_codec.unpack(sorted(fx["nos"]), fx["trio"], False)
        ks=[k for k in a if a[k]>0 and f.get(k,0)>0]
        if len(ks)<10: continue
        x=np.log([a[k] for k in ks]); y=np.log([f[k] for k in ks]); c=np.corrcoef(x,y)[0,1]; tot+=1
        if c<0.97: bad+=1; print(rid, round(c,3), len(s["nos"]), len(fx["nos"]), s["nos"][:20], fx["nos"][:20], s["mins"])
print("低い相関", bad, "/", tot)
