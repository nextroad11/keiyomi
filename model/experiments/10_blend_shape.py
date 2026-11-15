"""組み合わせ方の改良：オッズの大きさによる偏り（2乗の項）、AIと人気の食い違いの大きさ。基準決め期間で判断。"""
import pickle,numpy as np
from scipy.optimize import minimize
rows=pickle.load(open('/tmp/exrows.pkl','rb'))
def per(a,b): return [r for r in rows if a<=r['date']<=b and not r['dead']]
TR,VA=per('20250301','20251130'),per('20251201','20260531')
def base_feats(r,kind):
    o=r['o'+kind]; ok=o>0; q=np.where(ok,1/np.where(ok,o,1),0); q=q/q.sum()
    lq=np.log(q+1e-9); lp=np.log(r['p'+kind]+1e-9); lm=np.log(r['pm'+kind]+1e-9)
    c=lq.mean()
    return {'lq':lq,'lp':lp,'lm':lm,'lq2':(lq-c)**2/10,'dev':(lp-lm)**2/10,'lqdev':(lq-c)*(lp-lm)/10,'n':np.full(len(lq),r['n']/10)*(lp-lm)},ok
for r in TR+VA:
    for k in 'tf': r['F'+k]=base_feats(r,k)
def nll(w,D,kind,cols):
    L=0;G=np.zeros(len(w))
    for r in D:
        Fd,ok=r['F'+kind]; X=np.stack([Fd[c] for c in cols],1)
        s=np.where(ok,X@w,-1e9); m=s.max(); e=np.exp(s-m)*ok; Z=e.sum(); p=e/Z; wi=r['w'+kind]
        if not ok[wi]: continue
        L+=s[wi]-m-np.log(Z); G+=X[wi]-(p[:,None]*X).sum(0)
    return -L,-G
sets={'今の形':['lq','lp','lm'],'+オッズの2乗':['lq','lp','lm','lq2'],'+食い違いの2乗':['lq','lp','lm','dev'],
      '+オッズ×食い違い':['lq','lp','lm','lqdev'],'+頭数×食い違い':['lq','lp','lm','n'],'全部':['lq','lp','lm','lq2','dev','lqdev','n']}
for kind,name in [('t','3連単'),('f','3連複')]:
    b=None
    for nm,cols in sets.items():
        w=minimize(nll,np.r_[0.85,0.7,-0.5,np.zeros(len(cols)-3)],args=(TR,kind,cols),jac=True,method='L-BFGS-B').x
        v=nll(w,VA,kind,cols)[0]/len(VA)
        if b is None: b=v
        print(name,nm,np.round(w,3),'valid',round(v,4),'上乗せ',round(b-v,4),flush=True)
