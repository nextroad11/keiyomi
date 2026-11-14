import pickle,numpy as np
res=pickle.load(open('/tmp/res.pkl','rb'))
rng=np.random.default_rng(0)
def roi(per_race):  # list of (cost, ret, hits, bets) per race
    a=np.array(per_race,float); c,rt=a[:,0].sum(),a[:,1].sum()
    if c==0: return None
    bs=[]
    for _ in range(2000):
        i=rng.integers(0,len(a),len(a)); cc=a[i,0].sum(); bs.append(a[i,1].sum()/cc if cc else np.nan)
    return dict(bets=int(a[:,3].sum()),hit=int(a[:,2].sum()),cost=int(c),roi=rt/c,lo=np.nanpercentile(bs,2.5),hi=np.nanpercentile(bs,97.5),races=int((a[:,3]>0).sum()))
def fmt(d): return 'なし' if not d else f"{d['bets']}点/{d['races']}R 的中{d['hit']} 回収率{d['roi']:.1%} (95%: {d['lo']:.0%}〜{d['hi']:.0%})"
def thr(key,th,period):
    return [(100*sum(1 for ev,p in r[key] if ev>=th), sum(p for ev,p in r[key] if ev>=th), sum(1 for ev,p in r[key] if ev>=th and p>0), sum(1 for ev,p in r[key] if ev>=th)) for r in res[period]]
for key,name in [('win','単勝 AI'),('winM','単勝 人気のみ'),('place','複勝 AI')]:
    print(name)
    best=None
    for th in [0.8,0.9,1.0,1.1,1.2,1.3,1.5,1.8,2.2]:
        v=roi(thr(key,th,'valid'))
        if v: print(f'  基準{th}: valid {fmt(v)}')
        if v and v['bets']>=300 and (best is None or v['roi']>best[1]): best=(th,v['roi'])
    if best: print(f'  → validで決めた基準 {best[0]} を test に: {fmt(roi(thr(key,best[0],"test")))}')
for key,name in [('main','本線 3連複 AI'),('mainM','本線 3連複 人気のみ'),('are','荒れ狙い 3連単')]:
    for per in ['valid','test']:
        print(name,per,fmt(roi([(100*len(r[key]),sum(r[key]),sum(1 for p in r[key] if p>0),len(r[key])) for r in res[per]])))
for per in ['valid','test']:
    print('穴',per,fmt(roi([(100*len(r['ana']),sum(p for t,p in r['ana']),sum(1 for t,p in r['ana'] if p>0),len(r['ana'])) for r in res[per]])))
    # 全レース3連単を1点ずつランダム ≒ 72.5%
# 1番人気が飛ぶ 較正
for per in ['valid','test']:
    a=np.array([r['favout'] for r in res[per]],float)
    out=(a[:,2]>3)|np.isnan(a[:,2])
    print('1番人気飛び',per,f"AI予測平均{a[:,0].mean():.1%} 人気予測{a[:,1].mean():.1%} 実際{out.mean():.1%}")
    for lo,hi in [(0,.3),(.3,.4),(.4,.5),(.5,1)]:
        m=(a[:,0]>=lo)&(a[:,0]<hi); 
        if m.sum(): print(f'   AI {lo:.0%}-{hi:.0%}: {m.sum()}R 予測{a[m,0].mean():.1%} 実際{out[m].mean():.1%}')
    man=np.array([r['man'][:2] for r in res[per]]); p3=np.array([r['pay3'] for r in res[per]])
    print('  万馬券確率 AI平均',f'{man[:,0].mean():.1%}','人気',f'{man[:,1].mean():.1%}','実際',f'{(p3>=10000).mean():.1%}')
