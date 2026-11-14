import pickle,sys,numpy as np,pandas as pd; sys.path.insert(0,'model')
import features as F, pl
df=pd.read_pickle('/tmp/df.pkl')
df=df[(df.surface!='障害') & df.pos.notna() & df.odds.notna()].copy()
df=df.reset_index(drop=True)
# re-normalize market within remaining
inv=1/df.odds; df['p_mkt']=inv/inv.groupby(df.rid).transform('sum'); df['logp_mkt']=np.log(df.p_mkt)
P={}
per={'train':('20250301','20251130'),'valid':('20251201','20260531'),'test':('20260601','20261131')}
Z=['z_'+c for c in F.FEATURES]
sets={'market':['logp_mkt'],'index':Z,'combo':['logp_mkt']+Z}
for name,(a,b) in per.items():
    d=df[(df.date>=a)&(df.date<=b)]
    P[name]={k:pl.pack(d,c) for k,c in sets.items()}
    print(name,len(P[name]['market']['rids']),'races')
W={}
for k in sets:
    for l2 in [0.3,3,30]:
        w=pl.fit(P['train'][k],l2=l2)
        ll={n:pl.winner_logloss(pl.win_prob(w,P[n][k]),P[n][k]) for n in ['train','valid']}
        print(k,l2,{n:round(v,4) for n,v in ll.items()})
        W[(k,l2)]=w
pickle.dump((W,sets),open('/tmp/W.pkl','wb'))
# uniform baseline
import math
print('uniform valid',np.mean([math.log(m.sum()) for m in P['valid']['market']['M']]))
