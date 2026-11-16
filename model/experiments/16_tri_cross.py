"""3連単の改良1：3連複のオッズ（同じ3頭の組）を3連単の予想に加える。学習期間で重み、基準決め期間で判断。"""
import pickle,sys,numpy as np
sys.path.insert(0,'/home/claude/keiyomi/model')
import exotic as X
from scipy.optimize import minimize
D=pickle.load(open('/tmp/five.pkl','rb')); rows=D['rows']
TRr=[r for r in rows if r['date']<='20250531']; VA=[r for r in rows if '20250601'<=r['date']<='20260531']
def feats(r):
    n=r['n']; _,_,M=X.perms(n)
    o=r['o_t']; ok=o>0; q=np.where(ok,1/np.where(ok,o,1),0); q/=q.sum()
    of=r['o_f']; okf=of>0; qf=np.where(okf,1/np.where(okf,of,1),0); qf/=qf.sum()
    lqf=np.log(qf[M]+1e-9)                      # 同じ3頭の3連複人気
    lpf=np.log(np.exp(r['lp_f'].astype(np.float64))[M]+1e-12)  # 同じ3頭のAI 3連複確率
    return np.stack([np.log(q+1e-9),r['lp_t'],r['lm_t'],lqf,lpf],1),ok
for r in TRr+VA: r['Xc']=feats(r)
def nll(w,Dd,cols):
    L=0;G=np.zeros(len(cols))
    for r in Dd:
        Xm,ok=r['Xc']; Xm=Xm[:,cols]; s=np.where(ok,Xm@w,-1e9); m=s.max(); e=np.exp(s-m)*ok; Z=e.sum(); p=e/Z; wi=r['wt']
        if not ok[wi]: continue
        L+=s[wi]-m-np.log(Z); G+=Xm[wi]-(p[:,None]*Xm).sum(0)
    return -L,-G
res={}
for nm,cols in [('今の形',[0,1,2]),('+3連複の人気',[0,1,2,3]),('+3連複の人気+AIの3連複確率',[0,1,2,3,4])]:
    w=minimize(nll,np.r_[0.86,0.68,-0.52,np.zeros(len(cols)-3)],args=(TRr,cols),jac=True,method='L-BFGS-B').x
    v=nll(w,VA,cols)[0]/len(VA); res[nm]=(w,cols); print(nm,np.round(w,3),'基準決めNLL',round(v,4),flush=True)
pickle.dump(res,open('/tmp/tricross.pkl','wb'))
