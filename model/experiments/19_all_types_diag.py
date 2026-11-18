exec(open('/home/claude/keiyomi/model/experiments/18_all_types_rules.py').read().split('chosen = {}')[0])
for t in ["単勝","馬連","馬単"]:
    for a in [dict(th=1.0,pmin=0,omin=0),dict(th=1.1,pmin=0,omin=0),dict(th=0.95,pmin=0,omin=0)]:
        print(t,a,'基準決め',f(run(VA,t,a)),'| 採点',f(run(TE,t,a)))
# 組み合わせ：複勝以外で前後半とも100%超えの券種だけ（3連複＋3連単）を同じレースで
import numpy as np
def port(R,types):
    cost=ret=0; races=0; hit=0
    for d in R:
        n=0
        for t in types:
            x=d['types'].get(t); a=CH[t]
            if not x: continue
            c=pick(x,a,t); n+=len(c); cost+=len(c)*100; ret+=x['ret'][c].sum(); hit+=int((x['ret'][c]>0).sum())
        races+=n>0
    return f"{races}R 的中{hit} 回収率{ret/cost:.1%}" if cost else "なし"
import pickle; CH=pickle.load(open('/tmp/alltypes_chosen.pkl','rb'))
for ts in [["3連複","3連単"],["3連複","3連単","複勝","ワイド"],list(CH)]:
    print(ts,'基準決め',port(VA,ts),'| 採点',port(TE,ts))
