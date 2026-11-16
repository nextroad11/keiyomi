"""採点期間に1回だけ当てる（基準決め期間で選んだ設定のまま）。"""
import pickle,sys
sys.argv=['x']; exec(open('/home/claude/keiyomi/model/experiments/13_rules.py').read().split('if __name__ == "__main__":')[0])
final=pickle.load(open('/tmp/chosen.pkl','rb'))
for (k,rule),lst in final.items():
    if rule=='are': continue
    name={'f':'3連複','t':'3連単'}[k]
    print(f'===== {name} {rule}')
    for i,a in enumerate(lst):
        v=run(VA,k,rule,a); t=run(TE,k,rule,a,boot=(i==0))
        print(('★1位 ' if i==0 else '   ')+str(a)); print('     基準決め:',fmt(v)); print('     採点    :',fmt(t))
# 参考: 全部まとめ（基準決め＋採点）で1位設定
for (k,rule) in [('f','ev'),('t','ev'),('t','f2t')]:
    a=final[(k,rule)][0]; print('基準決め＋採点の合計',k,rule,a,fmt(run(VA+TE,k,rule,a,boot=True)))
