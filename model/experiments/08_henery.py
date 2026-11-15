"""2着・3着の割引（lam2, lam3）を、学習期間で重みを合わせ、基準決め期間の当てやすさで選ぶ。未使用期間は見ない。"""
import pickle,sys,numpy as np,pandas as pd
sys.path.insert(0,'/home/claude/keiyomi/model')
import features as F, pl, exotic as X
from scipy.optimize import minimize
df=pd.read_pickle('/tmp/df.pkl')
df=df[(df.surface!='障害') & df.pos.notna() & df.odds.notna()].copy().reset_index(drop=True)
inv=1/df.odds; df['p_mkt']=inv/inv.groupby(df.rid).transform('sum'); df['logp_mkt']=np.log(df.p_mkt)
R={r['id']:r for r in pickle.load(open('/tmp/races.pkl','rb'))}
EX=pickle.load(open('/tmp/ex.pkl','rb'))
Z=['z_'+c for c in F.FEATURES]
tr=df[(df.date>='20250301')&(df.date<='20251130')]
W3=pl.fit(pl.pack(tr,['logp_mkt']+Z),l2=3,stages=3)
W1=pl.fit(pl.pack(tr,['logp_mkt']+Z),l2=3,stages=1)
WM=pl.fit(pl.pack(tr,['logp_mkt']),l2=3,stages=1)
LAMS=[(1,1),(0.9,0.8),(0.85,0.7),(0.8,0.6),(0.75,0.5)]
races=[]
for rid,g in df.groupby('rid',sort=False):
    if rid not in EX or g.date.iloc[0]>'20260531' or g.date.iloc[0]<'20250301': continue
    nos,trio,tri=EX[rid]; n=len(nos)
    if len(tri)!=n*(n-1)*(n-2): continue
    race=R[rid]
    if '3連単' not in race['pay'] or len(race['pay']['3連単'])>1 or len(race['pay'].get('3連複',[]))>1: continue
    pos={no:i for i,no in enumerate(nos)}
    def strength(w,cols):
        sc=g[cols].to_numpy()@w; e=np.exp(sc-sc.max()); e/=e.sum(); s=np.zeros(n)
        for i,no in enumerate(g['no']): s[pos[no]]=e[i]
        return s
    s1=strength(W1,['logp_mkt']+Z); s3=strength(W3,['logp_mkt']+Z); sm=strength(WM,['logp_mkt'])
    qt,okt=X.market_prob(tri); qf,okf=X.market_prob(trio)
    P,C,_=X.perms(n)
    w3=[pos[int(x)] for x in race['pay']['3連単'][0]['combo'].split('-')]
    wt=int(np.where((P[:,0]==w3[0])&(P[:,1]==w3[1])&(P[:,2]==w3[2]))[0][0]); sw=sorted(w3)
    wf=int(np.where((C[:,0]==sw[0])&(C[:,1]==sw[1])&(C[:,2]==sw[2]))[0][0])
    rec={'date':race['date'],'n':n,'wt':wt,'wf':wf,'qt':np.log(qt+1e-9),'qf':np.log(qf+1e-9),'okt':okt,'okf':okf}
    for nm,s in [('s1',s1),('s3',s3),('sm',sm)]:
        for lam in LAMS:
            pt=X.pl_tri(s,*lam); rec[(nm,lam,'t')]=np.log(pt+1e-12); rec[(nm,lam,'f')]=np.log(X.trio_from_tri(pt,n)+1e-12)
    races.append(rec)
TR=[r for r in races if r['date']<='20251130']; VA=[r for r in races if r['date']>='20251201']
print('races',len(TR),len(VA))
def nll(w,D,keys,kind):
    L=0;G=np.zeros(len(w))
    for r in D:
        Xs=np.stack([r['q'+kind]]+[r[k+(kind,)] for k in keys],1); ok=r['ok'+kind]
        s=np.where(ok,Xs@w,-1e9); m=s.max(); e=np.exp(s-m)*ok; Z_=e.sum(); p=e/Z_; wi=r['w'+kind]
        if not ok[wi]: continue
        L+=s[wi]-m-np.log(Z_); G+=Xs[wi]-(p[:,None]*Xs).sum(0)
    return -L,-G
res=[]
for kind,name in [('t','3連単'),('f','3連複')]:
    base=None
    for ai in ['s1','s3']:
        for lam in LAMS:
            for lm in [(1,1),lam]:
                keys=[(ai,lam),('sm',lm)]
                w=minimize(nll,np.array([0.8,0.7,-0.5]),args=(TR,keys,kind),jac=True,method='L-BFGS-B').x
                v=nll(w,VA,keys,kind)[0]/len(VA)
                res.append((name,ai,lam,lm,np.round(w,3),round(v,4)))
                print(name,ai,'AI割引',lam,'人気PL割引',lm,np.round(w,3),'valid',round(v,4),flush=True)
pickle.dump(res,open('/tmp/henery.pkl','wb'))
