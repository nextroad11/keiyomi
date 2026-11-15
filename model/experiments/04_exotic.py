"""3連複・3連単：確定オッズ（実際）とAI確率の組み合わせ。前半で学習・基準決め、後半（未使用）で採点。"""
import pickle,sys,itertools,numpy as np,pandas as pd
sys.path.insert(0,'/home/claude/keiyomi/model')
import features as F, pl
from scipy.optimize import minimize
df=pd.read_pickle('/tmp/df.pkl')
df=df[(df.surface!='障害') & df.pos.notna() & df.odds.notna()].copy().reset_index(drop=True)
inv=1/df.odds; df['p_mkt']=inv/inv.groupby(df.rid).transform('sum'); df['logp_mkt']=np.log(df.p_mkt)
R={r['id']:r for r in pickle.load(open('/tmp/races.pkl','rb'))}
EX=pickle.load(open('/tmp/ex.pkl','rb'))
Z=['z_'+c for c in F.FEATURES]
tr=df[(df.date>='20250301')&(df.date<='20251130')]
W3=pl.fit(pl.pack(tr,['logp_mkt']+Z),l2=3,stages=3)
WM=pl.fit(pl.pack(tr,['logp_mkt']),l2=3,stages=3)
PERM={}; COMB={}
def idx(n):
    if n not in PERM:
        PERM[n]=np.array(list(itertools.permutations(range(n),3))); COMB[n]=np.array(list(itertools.combinations(range(n),3)))
    return PERM[n],COMB[n]
def pl_tri(s,P):
    a,b,c=P[:,0],P[:,1],P[:,2]
    return s[a]*s[b]/np.maximum(1e-12,1-s[a])*s[c]/np.maximum(1e-12,1-s[a]-s[b])
rows=[]
for rid,g in df.groupby('rid',sort=False):
    if rid not in EX: continue
    nos,trio,tri=EX[rid]; n=len(nos)
    if len(tri)!=n*(n-1)*(n-2): continue
    pos={no:i for i,no in enumerate(nos)}
    s_ai=np.zeros(n); s_mk=np.zeros(n)
    X=g[['logp_mkt']+Z].to_numpy()
    sc=X@W3; e=np.exp(sc-sc.max()); e/=e.sum()
    sm=g['logp_mkt'].to_numpy()*WM[0]; em=np.exp(sm-sm.max()); em/=em.sum()
    for i,no in enumerate(g['no']): s_ai[pos[no]]=e[i]; s_mk[pos[no]]=em[i]
    P,C=idx(n)
    pt=pl_tri(s_ai,P); pmt=pl_tri(s_mk,P)
    # 3連複＝6通りの合計
    key=lambda arr: (arr[:,0]*10000+arr[:,1]*100+arr[:,2])
    srt=np.sort(P,1); kp=key(srt); kc=key(C); order=np.searchsorted(kc,kp) if np.all(np.diff(kc)>0) else None
    pf=np.bincount(order,weights=pt,minlength=len(C)); pmf=np.bincount(order,weights=pmt,minlength=len(C))
    race=R[rid]; win3=[int(x) for x in race['pay']['3連単'][0]['combo'].split('-')] if '3連単' in race['pay'] else None
    if not win3: continue
    wi=(pos[win3[0]],pos[win3[1]],pos[win3[2]])
    wt=int(np.where((P[:,0]==wi[0])&(P[:,1]==wi[1])&(P[:,2]==wi[2]))[0][0]); sw=sorted(wi)
    wf=int(np.where((C[:,0]==sw[0])&(C[:,1]==sw[1])&(C[:,2]==sw[2]))[0][0])
    rows.append(dict(rid=rid,date=race['date'],n=n,pt=pt,pmt=pmt,ot=tri,pf=pf,pmf=pmf,of=trio,wt=wt,wf=wf,
                     pay_t=race['pay']['3連単'][0]['yen'],pay_f=race['pay']['3連複'][0]['yen'] if '3連複' in race['pay'] else 0,
                     dead=len(race['pay']['3連単'])>1 or len(race['pay'].get('3連複',[]))>1))
pickle.dump(rows,open('/tmp/exrows.pkl','wb'))
print('races',len(rows))
