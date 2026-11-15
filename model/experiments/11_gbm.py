"""勾配ブースティング（非線形）の上乗せ。学習期間の中は時期で3分割して「使っていない時期の予測」を作る。基準決め期間で判断。"""
import sys,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/keiyomi/model')
import features as F, features2 as F2, pl
from sklearn.ensemble import HistGradientBoostingClassifier
df=pd.read_pickle('/tmp/df2.pkl')
df=df[(df.surface!='障害') & df.pos.notna() & df.odds.notna()].copy().reset_index(drop=True)
inv=1/df.odds; df['p_mkt']=inv/inv.groupby(df.rid).transform('sum'); df['logp_mkt']=np.log(df.p_mkt)
df['mkt_rank']=df.groupby('rid')['odds'].rank()/df['n']
df['turf']=(df.surface=='芝').astype(float)
RAW=F.FEATURES+F2.FEATURES2; Z=['z_'+c for c in F.FEATURES]; Z2=['z_'+c for c in F2.FEATURES2]
COLS=['logp_mkt','mkt_rank','n','cls','distance','turf']+RAW+Z+Z2
tr=df[(df.date>='20250301')&(df.date<='20251130')].copy(); va=df[(df.date>='20251201')&(df.date<='20260531')].copy()
def gbm(target, lr=0.05, it=300, leaf=40):
    return HistGradientBoostingClassifier(learning_rate=lr,max_iter=it,min_samples_leaf=leaf,l2_regularization=1.0,max_leaf_nodes=15,random_state=0)
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
months=sorted(tr.date.str[:6].unique()); folds=[months[i::3] for i in range(3)]
for target,name in [('win',lambda d:(d.pos==1).astype(int)),('top3',lambda d:(d.pos<=3).astype(int))]:
    for it in [200,400]:
        oof=np.zeros(len(tr))
        for f in folds:
            m=tr.date.str[:6].isin(f).to_numpy()
            g=gbm(target,it=it).fit(tr.loc[~m,COLS],name(tr[~m])); oof[m]=logit(g.predict_proba(tr.loc[m,COLS])[:,1])
        g=gbm(target,it=it).fit(tr[COLS],name(tr)); vg=logit(g.predict_proba(va[COLS])[:,1])
        tr['g'],va['g']=oof,vg
        for cols in [['logp_mkt','g'],['logp_mkt','g']+Z]:
            w=pl.fit(pl.pack(tr,cols),l2=3,stages=3); Pv=pl.pack(va,cols)
            print(target,it,cols[:2],'+Z' if len(cols)>2 else '','valid NLL',round(pl._loglik(w,Pv,0,3)[0]/len(Pv['rids']),4),np.round(w[:2],3),flush=True)
print('参考: 今のAI 5.9875 / 人気のみ 6.0041')
