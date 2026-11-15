import pickle,numpy as np
exec(open('/home/claude/keiyomi/model/experiments/05_exotic_fit.py').read().split("models=")[0])
FIT=pickle.load(open('/tmp/exfit.pkl','rb'))
rng=np.random.default_rng(1)
def probs(r,kind,mn):
    w,cols=FIT[(kind,mn)]; X,ok=r['X'+kind]; s=np.where(ok,X[:,cols]@w,-1e9); e=np.exp(s-s.max())*ok; return e/e.sum(),ok
# 1) 有意性
for kind,name in [('t','3連単'),('f','3連複')]:
    for nm,D in [('valid',VA),('test',TE)]:
        d=[]
        for r in D:
            a,_=probs(r,kind,'人気(実オッズ)のみ'); b,_=probs(r,kind,'実オッズ+AI+単勝人気PL'); wi=r['w'+kind]
            if a[wi]>0 and b[wi]>0: d.append(np.log(b[wi])-np.log(a[wi]))
        d=np.array(d); bs=[rng.choice(d,len(d)).mean() for _ in range(2000)]
        print(name,nm,'上乗せ(大きいほど良い)',round(d.mean(),4),'95%',np.round(np.percentile(bs,[2.5,97.5]),4))
BEST='実オッズ+AI+単勝人気PL'
def sim(D,kind,rule,**kw):
    out=[]
    for r in D:
        p,ok=probs(r,kind,BEST); o=r['o'+kind]; wi=r['w'+kind]; pay=r['pay_'+kind]
        ev=p*o
        if rule=='ev':
            cand=np.where(ok&(ev>=kw['th'])&(p>=kw.get('pmin',0)))[0]
            cand=cand[np.argsort(-ev[cand])][:kw['k']]
        elif rule=='main':
            order=np.argsort(-p); cum=0; cand=[]
            for i in order:
                if len(cand)>=kw['k'] or (len(cand)>=3 and cum>=kw['target']): break
                cand.append(i); cum+=p[i]
            cand=np.array(cand)
        elif rule=='are':
            hi=ok&(o>=100); man=p[hi].sum()
            if man<kw['man']: cand=np.array([],int)
            else:
                idx=np.where(hi)[0]; idx=idx[np.argsort(-p[idx])]; cand=[]
                for i in idx:
                    if p[i]<0.01 or len(cand)>=12: break
                    cand.append(i)
                cand=np.array(cand,int)
        hit=int(wi in set(cand.tolist())) if len(cand) else 0
        out.append((100*len(cand), pay if hit else 0, hit, len(cand)))
    a=np.array(out,float); c=a[:,0].sum()
    if c==0: return None
    bs=[]
    for _ in range(1500):
        i=rng.integers(0,len(a),len(a)); cc=a[i,0].sum(); bs.append(a[i,1].sum()/cc if cc else np.nan)
    return dict(pts=int(a[:,3].sum()),races=int((a[:,3]>0).sum()),hit=int(a[:,2].sum()),roi=a[:,1].sum()/c,lo=np.nanpercentile(bs,2.5),hi=np.nanpercentile(bs,97.5))
def f(d): return 'なし' if not d else f"{d['pts']}点/{d['races']}R 的中{d['hit']} 回収率{d['roi']:.1%} (95%: {d['lo']:.0%}〜{d['hi']:.0%})"
for kind,name in [('f','3連複'),('t','3連単')]:
    print('=====',name)
    grid=[]
    for th in [1.0,1.1,1.2,1.3,1.5,1.8]:
        for k in [3,6,12]:
            for pmin in [0,0.002,0.005,0.01]:
                v=sim(VA,kind,'ev',th=th,k=k,pmin=pmin)
                if v and v['pts']>=300: grid.append(((th,k,pmin),v))
    grid.sort(key=lambda t:-t[1]['roi'])
    for g,v in grid[:5]: print(' 期待値 valid',g,f(v))
    (th,k,pmin),_=grid[0]
    print(' → validで1位の基準',(th,k,pmin),'test:',f(sim(TE,kind,'ev',th=th,k=k,pmin=pmin)))
    # 頑健さ：上位5つの基準のtest平均
    print(' → validで上位5つの基準それぞれのtest:',[round(sim(TE,kind,'ev',th=g[0],k=g[1],pmin=g[2])['roi'],3) for g,_ in grid[:5]])
    for tgt in ([0.2,0.3,0.4] if kind=='f' else [0.08,0.12,0.2]):
        print(f' 本線(合計{tgt:.0%}まで3〜6点) valid',f(sim(VA,kind,'main',target=tgt,k=6)),'| test',f(sim(TE,kind,'main',target=tgt,k=6)))
    if kind=='t':
        for man in [0.6,0.7,0.8,0.85]:
            print(f' 荒れ狙い(万馬券確率≥{man:.0%}) valid',f(sim(VA,kind,'are',man=man)),'| test',f(sim(TE,kind,'are',man=man)))
