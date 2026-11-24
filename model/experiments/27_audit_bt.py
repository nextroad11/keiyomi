import sys
src=open("/home/claude/keiyomi/model/experiments/26_ten_years.py").read()
exec(src.split("rng = np.random")[0])
import collections
W3, WM, BW = pickle.load(open(C / "models.pkl", "rb"))["A 2022-03〜"]
R = prod.RULES["trio"]
def reason(pf, okf, of):
    if pf is None or okf.sum()==0: return "オッズなし"
    ev=pf*of; a=okf&(ev>=1)
    if not a.any(): return "期待値1以上なし"
    b=a&(pf>=R["pmin"])
    if not b.any(): return "確率1%未満のみ"
    if not (b&(of>=R["omin"])).any(): return "オッズ10倍未満のみ"
    return "買い"
for nm,(a,b) in (("基準決め 2025-06〜2026-05",VA),("採点 2026-06〜11",TE)):
    rids=df[(df.date>=a)&(df.date<=b)].rid.unique()
    rc=collections.Counter(); grp={"買い":[], "見送り":[]}; calib=[]; top1={"買い":[0,0,0],"見送り":[0,0,0]}
    for rid in rids:
        A=arrays(rid,W3,WM)
        if not A: rc["データ不足で除外"]+=1; continue
        pr,_=prod.exotic_probs(A["s_ai"],A["s_mk"],A["tri"],A["trio"],BW); pf,okf=pr["f"]
        rs=reason(pf,okf,A["trio"]); rc[rs]+=1
        q,_=X.market_prob(A["trio"]); g="買い" if rs=="買い" else "見送り"
        grp[g].append((np.log(pf[A["wf"]]+1e-12), np.log(q[A["wf"]]+1e-12)))
        if rs=="買い":
            pk=prod.picks(A["n"],pr,A["tri"],A["trio"])["trio"]
            for i in pk: calib.append((pf[i], i==A["wf"], A["trio"][i]))
        # 仮想予想：AIの確率が一番高い3連複を100円
        i=int(np.argmax(np.where(okf,pf,-1))); t=top1[g]; t[0]+=1; t[1]+= i==A["wf"]; t[2]+= A["pay_f"] if i==A["wf"] else 0
    tot=sum(v for k,v in rc.items() if k!="データ不足で除外")
    print("==",nm); print(" 理由別", {k:f"{v}（{v/tot:.0%}）" for k,v in rc.items()})
    for g,v in grp.items():
        v=np.array(v); print(f" {g} {len(v)}R 勝った組の平均対数確率 AI {v[:,0].mean():.3f} 市場 {v[:,1].mean():.3f} 差 {v[:,0].mean()-v[:,1].mean():+.3f}")
    for g,(n,h,r) in top1.items(): print(f" 仮想予想（AI確率1位の3連複を100円）{g}: {n}R 的中率{h/n:.1%} 回収率{r/(n*100):.1%}")
    c=np.array([(p,h) for p,h,o in calib]); 
    for lo,hi in ((0.01,0.02),(0.02,0.04),(0.04,0.08),(0.08,1)):
        m=(c[:,0]>=lo)&(c[:,0]<hi)
        if m.sum(): print(f"  買い目の予測確率 {lo:.0%}〜{hi:.0%}: {m.sum()}点 予測平均{c[m,0].mean():.2%} 実際の的中率{c[m,1].mean():.2%}")
