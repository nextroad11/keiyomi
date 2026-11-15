import pickle,numpy as np
from scipy.optimize import minimize
rows=pickle.load(open('/tmp/exrows.pkl','rb'))
TAKE={'t':0.275,'f':0.25}
def feats(r,kind):
    o=r['o'+kind]; ok=o>0
    q=np.where(ok,1/np.where(ok,o,1),0); q=q/q.sum()
    p=r['p'+kind]; pm=r['pm'+kind]
    eps=1e-9
    X=np.stack([np.log(q+eps),np.log(p+eps),np.log(pm+eps)],1)
    return X,ok
def per(a,b): return [r for r in rows if a<=r['date']<=b and not r['dead']]
TR,VA,TE=per('20250301','20251130'),per('20251201','20260531'),per('20260601','20261131')
print('races',len(TR),len(VA),len(TE))
def nll(w,data,kind,cols):
    L=0;G=np.zeros(len(w))
    for r in data:
        X,ok=r['X'+kind]; X=X[:,cols]
        s=X@w; s=np.where(ok,s,-1e9); m=s.max(); e=np.exp(s-m)*ok; Z=e.sum(); p=e/Z
        wi=r['w'+kind]
        if not ok[wi]: continue
        L+=s[wi]-m-np.log(Z); G+=X[wi]-(p[:,None]*X).sum(0)
    return -L,-G
for r in rows:
    for kind in 'tf': r['X'+kind]=feats(r,kind)
models={'人気(実オッズ)のみ':[0],'AIのPLのみ':[1],'実オッズ+AI':[0,1],'実オッズ+AI+単勝人気PL':[0,1,2]}
FIT={}
for kind,name in [('t','3連単'),('f','3連複')]:
    for mn,cols in models.items():
        w=minimize(nll,np.ones(len(cols))*0.5,args=(TR,kind,cols),jac=True,method='L-BFGS-B').x
        FIT[(kind,mn)]=(w,cols)
        res={nm:nll(w,d,kind,cols)[0]/len(d) for nm,d in [('valid',VA),('test',TE)]}
        print(name,mn,np.round(w,3),{k:round(v,4) for k,v in res.items()})
pickle.dump(FIT,open('/tmp/exfit.pkl','wb'))
