import pickle,sys,numpy as np,pandas as pd; sys.path.insert(0,'model')
import features as F, pl, backtest as B
exec(open('/tmp/exp1.py').read().split("W={}")[0])
R={r['id']:r for r in pickle.load(open('/tmp/races.pkl','rb'))}
W,sets=pickle.load(open('/tmp/W.pkl','rb'))
w3=W[('combo',3)]
w1=pl.fit(P['train']['combo'],l2=3,stages=1); wm1=pl.fit(P['train']['market'],l2=3,stages=1)
print('win-only market alpha',wm1, 'logloss valid market1',pl.winner_logloss(pl.win_prob(wm1,P['valid']['market']),P['valid']['market']),'combo1',pl.winner_logloss(pl.win_prob(w1,P['valid']['combo']),P['valid']['combo']))
ANA_MIN={"単勝":0.05,"複勝":0.15,"馬連":0.03,"ワイド":0.06,"馬単":0.02,"3連複":0.015,"3連単":0.01}
def run(period):
    Pc,Pm=P[period]['combo'],P[period]['market']
    p1=pl.win_prob(w1,Pc); p3=pl.win_prob(w3,Pc); pm1=pl.win_prob(wm1,Pm)
    out=[]
    for r in range(len(Pc['rids'])):
        m=Pc['M'][r]; n=m.sum(); idx=Pc['idx'][r][:n]
        sub=df.loc[idx]; nos=sub['no'].tolist(); odds=sub['odds'].to_numpy(); pmin=sub['pmin'].to_numpy()
        race=R[Pc['rids'][r]]
        T=B.pl_tables(p3[r][:n]); Mk=B.pl_tables(sub['p_mkt'].to_numpy())
        rec={'rid':Pc['rids'][r]}
        # 単勝
        rec['win']=[(p1[r][i]*odds[i], B.payout(race,'単勝',(i,),nos)) for i in range(n)]
        rec['winM']=[(pm1[r][i]*odds[i], B.payout(race,'単勝',(i,),nos)) for i in range(n)]
        rec['place']=[(T['複勝'][(i,)]*pmin[i] if pmin[i]==pmin[i] else 0, B.payout(race,'複勝',(i,),nos)) for i in range(n)]
        # 本線 3連複
        for nm,TT in [('main',T),('mainM',B.pl_tables(np.exp(np.log(sub['p_mkt'].to_numpy())*wm1[0])/np.exp(np.log(sub['p_mkt'].to_numpy())*wm1[0]).sum()))]:
            rows=sorted(TT['3連複'].items(),key=lambda kv:-kv[1]); pick=[];cum=0
            for k,p in rows:
                if len(pick)>=6 or (len(pick)>=3 and cum>=0.30): break
                pick.append(k); cum+=p
            rec[nm]=[B.payout(race,'3連複',k,nos) for k in pick]
        # 穴
        cand=[]
        for typ in ['単勝','複勝','馬連','ワイド','馬単','3連複','3連単']:
            for k,p in T[typ].items():
                q=Mk[typ].get(k);
                if not q: continue
                od=odds[k[0]] if typ=='単勝' else (1-B.TAKE[typ])/q
                ev=p*od
                if ev>=1.2 and p/q>=1.5 and p>=ANA_MIN[typ]: cand.append((ev,typ,k))
        cand.sort(key=lambda t:-t[0])
        rec['ana']=[(t[1],B.payout(race,t[1],t[2],nos)) for t in cand[:3]]
        # 荒れ狙い
        hi=[(p,k,Mk['3連単'][k]) for k,p in T['3連単'].items() if (1-.275)/Mk['3連単'][k]>=100]
        man=sum(h[0] for h in hi); manM=sum(h[2] for h in hi)
        rec['man']=(man,manM,B.payout(race,'3連単',max(T['3連単'],key=lambda k:0) if False else (0,1,2),nos))
        pay3=race['pay']['3連単'][0]['yen']; rec['pay3']=pay3
        are=[]
        if man>=0.4 and man>manM:
            for p,k,_ in sorted(hi,key=lambda h:-h[0]):
                if p<0.01 or len(are)>=12: break
                are.append(B.payout(race,'3連単',k,nos))
        rec['are']=are
        # 1番人気が飛ぶ
        fav=int(np.argmin(odds)); rec['favout']=(1-T['複勝'][(fav,)], 1-Mk['複勝'][(fav,)], race_pos(sub,fav))
        out.append(rec)
    return out
def race_pos(sub,i): return sub['pos'].iloc[i]
res={p:run(p) for p in ['valid','test']}
pickle.dump(res,open('/tmp/res.pkl','wb')); print('saved')
