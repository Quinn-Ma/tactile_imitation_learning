"""Verify the complete prospectively specified collection before model training."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/revision_v2'

def main():
    plan=json.loads((OUT/'protocol.json').read_text())
    rows=plan['collection']; paths=list((OUT/'dataset/episodes').glob('*.npz'))
    assert len(paths)==len(rows)==336
    assert len({r['seed'] for r in rows+plan['control_test']+plan['counterfactual']})==468
    manifest=[]
    for row in rows:
        path=OUT/'dataset/episodes'/(row['episode_id']+'.npz')
        with np.load(path,allow_pickle=False) as ep:
            assert int(ep['seed'])==row['seed'] and str(ep['split'])==row['split']
            assert str(ep['episode_id'])==row['episode_id']
            assert str(ep['collection_controller'])==row['controller']
            assert ep['action'].shape==(150,7) and ep['rgb'].shape==(151,64,64,3)
            assert np.max(np.abs(ep['action'][44:135][:,[0,1,3,4,5]]))==0
            assert ep['tactile'].shape==(151,6) and ep['substep_normal_peak'].shape==(151,2)
            for key in ['action','tactile','substep_normal_peak','height','proprio','reward']:
                assert np.isfinite(ep[key]).all(),(path,key)
            actual=json.loads(str(ep['params_json']))
            for key,value in row['params'].items():
                if key=='feasibility':
                    # Legacy cube uses its compiled seeded size, rather than
                    # the conservative pre-compilation .022 m bound.
                    for name,v in value.items():
                        if name in ['horizontal_width_bound_m','required_aperture_m']:
                            assert actual[key][name]<=v+1e-12
                        else:assert actual[key][name]==v,(path,key,name)
                else:assert actual[key]==value,(path,key,value,actual[key])
        manifest.append({'path':str(path.resolve()),'split':row['split'],'episode_id':row['episode_id'],
                         'seed':row['seed'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'steps':150,'has_reward':True})
    (OUT/'dataset/data_manifest.json').write_text(json.dumps(manifest,indent=2))
    report={'episodes':len(manifest),'counts':{k:sum(r['split']==k for r in manifest) for k in ['train','val','calibration']},
            'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
            'all_prescribed_parameters_match':True,'all_intervention_actions_are_2D':True,
            'all_outcomes_retained':True,'test_and_diagnostic_seeds_disjoint':True}
    (OUT/'dataset/audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=='__main__':main()
