exec(open('/home/claude/keiyomi/model/experiments/18_all_types_rules.py').read().split('chosen = {}')[0])
import pickle; CH=pickle.load(open('/tmp/alltypes_chosen.pkl','rb'))
def dep(R,t,a):
    cost=0; hits=[]
    for d in R:
        x=d['types'].get(t)
        if not x: continue
        c=pick(x,a,t); cost+=len(c)*100
        for i in c:
            if x['ret'][i]>0: hits.append((int(x['ret'][i]),d['rid'],round(float(x['o'][i]),1),round(float(x['p'][i]),3)))
    hits.sort(reverse=True); s=sum(h[0] for h in hits)
    return f"{s/cost:.0%} 上位1つ除く{(s-hits[0][0])/cost:.0%} 上位3つ除く{(s-sum(h[0] for h in hits[:3]))/cost:.0%} 上位{hits[:3]}" if hits else '-'
for t,a in [('馬連',dict(th=1.0,pmin=0,omin=0)),('馬単',dict(th=1.0,pmin=0,omin=0)),('3連複',CH['3連複']),('3連単',CH['3連単']),('複勝',CH['複勝']),('ワイド',CH['ワイド'])]:
    print(t,'基準決め',dep(VA,t,a)); print('   採点',dep(TE,t,a))
