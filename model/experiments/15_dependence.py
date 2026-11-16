import sys; sys.argv=['x']
exec(open('/home/claude/keiyomi/model/experiments/13_rules.py').read().split('if __name__ == "__main__":')[0])
import numpy as np
def detail(R,k,rule,a):
    cost=0; pays=[]
    for r in R:
        c=pick(r,k,rule,a); cost+=len(c)*100
        if r['w'+k] in set(c.tolist()): pays.append(r['pay_'+k])
    pays=sorted(pays,reverse=True)
    s=sum(pays)
    return f"回収率{s/cost:.0%} 最大の払戻を除くと{(s-pays[0])/cost:.0%} 上位3つを除くと{(s-sum(pays[:3]))/cost:.0%} 上位3つの払戻{pays[:3]}"
for nm,R in [('基準決め',VA),('採点',TE)]:
    print(nm,'3連複 EV≥1.0 確率≥1% オッズ≥10:',detail(R,'f','ev',dict(th=1.0,pmin=0.01,omin=10)))
    print(nm,'3連単 3連複→3連単 EV≥1.0:',detail(R,'t','f2t',dict(th=1.0)))
    print(nm,'3連単 3連複→3連単 EV≥0.9:',detail(R,'t','f2t',dict(th=0.9)))
