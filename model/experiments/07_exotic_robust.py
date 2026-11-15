exec(open('/home/claude/keiyomi/model/experiments/06_exotic_bet.py').read().split("for kind,name in [('f','3連複'),('t','3連単')]:\n    print('=====',name)")[0].split("# 1) 有意性")[0])
BEST='実オッズ+AI+単勝人気PL'
exec("def sim"+open('/home/claude/keiyomi/model/experiments/06_exotic_bet.py').read().split("def sim")[1].split("for kind,name in [('f','3連複'),('t','3連単')]:\n    print('=====',name)")[0])
print('3連複 期待値ルールのまわり（valid | test）')
for th in [1.0,1.1,1.2,1.3]:
    for pmin in [0.005,0.01,0.02]:
        v=sim(VA,'f','ev',th=th,k=12,pmin=pmin); t=sim(TE,'f','ev',th=th,k=12,pmin=pmin)
        print(f' EV≥{th} 確率≥{pmin:.1%}: {f(v)} | {f(t)}')
# 全期間（valid+test）をまとめて
for th,pmin in [(1.1,0.01),(1.2,0.01),(1.1,0.02)]:
    print('valid+test',th,pmin,f(sim(VA+TE,'f','ev',th=th,k=12,pmin=pmin)))
# 実オッズだけ(AIなし)で同じルール → 0.75付近になるはず
def sim_mk(D,th,pmin):
    out=[]
    for r in D:
        p,ok=probs(r,'f','人気(実オッズ)のみ'); o=r['of']; ev=p*o; c=np.where(ok&(ev>=th)&(p>=pmin))[0][:12]
        out.append((100*len(c), r['pay_f'] if r['wf'] in set(c.tolist()) else 0))
    a=np.array(out,float); return a[:,1].sum()/max(1,a[:,0].sum()), int(a[:,0].sum()/100)
print('人気のみで同じルール（valid,test）', sim_mk(VA,1.1,0.01), sim_mk(TE,1.1,0.01))
