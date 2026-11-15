"""追加特徴量の上乗せ（学習期間で学習、基準決め期間の上位3着の当てやすさで判断）。未使用期間は見ない。"""
import sys,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/keiyomi/model')
import features as F, features2 as F2, pl
df=pd.read_pickle('/tmp/df2.pkl')
df=df[(df.surface!='障害') & df.pos.notna() & df.odds.notna()].copy().reset_index(drop=True)
inv=1/df.odds; df['p_mkt']=inv/inv.groupby(df.rid).transform('sum'); df['logp_mkt']=np.log(df.p_mkt)
tr=df[(df.date>='20250301')&(df.date<='20251130')]; va=df[(df.date>='20251201')&(df.date<='20260531')]
Z=['z_'+c for c in F.FEATURES]; Z2=['z_'+c for c in F2.FEATURES2]
def score(cols,l2=3):
    Pt,Pv=pl.pack(tr,cols),pl.pack(va,cols); w=pl.fit(Pt,l2=l2,stages=3)
    return pl._loglik(w,Pv,0,3)[0]/len(Pv['rids']), w
base,_=score(['logp_mkt']+Z); mk,_=score(['logp_mkt'])
print('上位3着NLL(valid)  人気のみ',round(mk,4),' 今のAI',round(base,4),' 差',round(mk-base,4))
allv,w=score(['logp_mkt']+Z+Z2); print('全部足す',round(allv,4),'今のAIからの上乗せ',round(base-allv,4))
for c,v in zip(['logp_mkt']+Z+Z2,w):
    if c in Z2: print(f'  {c:16s}{v:+.3f}')
print('1つずつ足す:')
for c in Z2:
    v,_=score(['logp_mkt']+Z+[c]); print(f'  +{c:16s} 上乗せ {base-v:+.4f}')
