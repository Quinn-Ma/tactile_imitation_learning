"""Check the saved simulator dataset and report measured descriptive statistics."""
import json,pathlib,hashlib
from sim_adapter import np


def main():
    out=pathlib.Path(__file__).parent
    logs=[json.loads(s) for s in (out/'generation_log.jsonl').read_text().splitlines()]
    files=sorted((out/'episodes').glob('ep_*.npz'))
    assert len(logs)==len(files)==160
    lookup={r['episode_id']:r for r in logs}; seeds=set();sizes=set();rows=[]
    for file in files:
        d=np.load(file,allow_pickle=False);eid=str(d['episode_id']);split=str(d['split'])
        seed=int(d['seed']);assert seed not in seeds;seeds.add(seed)
        assert hashlib.sha256(file.read_bytes()).hexdigest()==lookup[eid]['sha256']
        assert d['rgb'].shape==(151,64,64,3) and d['rgb'].dtype==np.uint8
        for key,width in [('tactile',6),('proprio',9),('height',1),('contact',2),('substep_normal_peak',2)]:
            assert d[key].shape==(151,width) and np.isfinite(d[key]).all(),(eid,key)
        assert d['action'].shape==(150,7) and d['reward'].shape==(150,1)
        assert np.isfinite(d['action']).all() and np.abs(d['action']).max()<=1
        assert np.isfinite(d['reward']).all()
        assert (d['substep_normal_peak']>=0).all() and (d['tactile'][:,[0,3]]>=0).all()
        p=json.loads(str(d['params_json']));sizes.add(tuple(p['half_size']))
        run=0;longest=0
        for v in d['height'][1:136,0]>=.10:
            run=run+1 if v else 0;longest=max(longest,run)
        peak=float(d['substep_normal_peak'].max())
        rows.append(dict(split=split,strict10cm=longest>=10,native=lookup[eid]['sustained_success_10steps'],
                         force_exceeded=peak>8,safe_strict10cm=longest>=10 and peak<=8,
                         peak=peak,setup44_exceeded=bool(d['substep_normal_peak'][:45].max()>8)))
    assert len(sizes)==1
    groups={}
    for split in ['train','val','calibration','test_id','test_ood']:
        subset=[r for r in rows if r['split']==split]
        groups[split]=dict(n=len(subset),**{key:int(sum(r[key] for r in subset)) for key in [
            'native','strict10cm','force_exceeded','safe_strict10cm','setup44_exceeded']},
            mean_full_episode_peak_N=float(np.mean([r['peak'] for r in subset])))
    result=dict(episodes=160,transitions=24000,states=24160,bytes=sum(f.stat().st_size for f in files),
        unique_seeds=len(seeds),fixed_compiled_half_size_m=list(next(iter(sizes))),
        checks='PASS: 160 hashes, disjoint episode seeds, required shapes/dtypes, finite states/actions/rewards, action bounds, fixed geometry',
        groups=groups)
    (out/'dataset_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
