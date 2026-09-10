"""Freeze all policy contrasts before any revision-v2 test execution."""
from pathlib import Path
import argparse,datetime,hashlib,json
OUT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--amend-pretest',action='store_true');args=p.parse_args()
    rows=[]
    for variant in ['vision','visuotactile']:
        for seed in range(3):
            suffix=f'{variant}_seed{seed}'
            rows.append(dict(name='WM_'+suffix,group='WM_'+variant,kind='world_model',seed=seed,
                model=f'outputs/revision_v2/world_models/{suffix}/best.pt',
                actor=f'outputs/revision_v2/rl/{suffix}/rl_actor.pt'))
            rows.append(dict(name='WM_BC_'+suffix,group='WM_BC_'+variant,kind='world_model',seed=seed,
                model=f'outputs/revision_v2/world_models/{suffix}/best.pt',
                actor=f'outputs/revision_v2/rl/{suffix}/bc_actor.pt'))
            for algorithm in ['BC','IQL']:
                rows.append(dict(name=algorithm+'_'+suffix,group=algorithm+'_'+variant,kind='reactive',seed=seed,
                    checkpoint=f'outputs/revision_v2/reactive_runs/{suffix}/{algorithm.lower()}_policy.pt'))
    for name in ['script','force_feedback']:
        rows.append(dict(name=name,group=name,kind=name))
    for mode in ['model_guard','model_no_guard']:
        for seed in range(3):
            rows.append(dict(name=f'{mode}_seed{seed}',group=mode,kind='adaptive',mode=mode,seed=seed,
                             model=f'outputs/revision_v2/world_models/visuotactile_seed{seed}/best.pt'))
    for mode in ['reactive_guard','reactive_no_guard']:
        rows.append(dict(name=mode,group=mode,kind='adaptive',mode=mode))
    assert len(rows)==34 and len({r['name'] for r in rows})==34
    path=OUT/'methods.json';payload=json.dumps(rows,indent=2)
    if path.exists() and path.read_text()!=payload:
        assert args.amend_pretest,'Do not silently modify frozen methods.'
        assert not list((OUT/'evaluations').rglob('episodes.jsonl')),'Cannot amend after test execution'
        prior=json.loads(path.read_text())
        assert len(prior)==28 and all(r in rows for r in prior)
        amendment={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'old_methods_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'new_methods_logical_sha256':hashlib.sha256(payload.encode()).hexdigest(),
            'reason':'Independent causal/literature review before test execution: isolate imagination updates from frozen predictive representation.',
            'addition':'Six frozen-world-model BC policies, paired with the exact corresponding pre-imagination checkpoint.',
            'main_original_28_retained':True,'test_outcomes_available':False,
            'collection_and_training_and_test_seed_protocol_unchanged':True,
            'counterfactual_addition':'Forecast identical independently executed branches with vision and visuotactile models, three seeds each.'}
        (OUT/'protocol_amendment.json').write_text(json.dumps(amendment,indent=2))
        path.write_text(payload)
    elif path.exists():assert path.read_text()==payload,'Do not silently modify frozen methods.'
    else:path.write_text(payload)
    print(f'{len(rows)} methods frozen')

if __name__=='__main__':main()
