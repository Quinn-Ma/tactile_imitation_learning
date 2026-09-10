from pathlib import Path
import hashlib
import json
from sim_adapter_v2 import parameters

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/revision_v2'
TRAIN_GEOMS=['legacy_cube','squat_box','tall_box']
OOD_GEOMS=['wide_box','narrow_tall_box']

def main():
    rows=[]
    for split,count,start in [('train',240,20310000),('val',48,20320000),('calibration',48,20330000)]:
        for i in range(count):
            seed=start+i; geom=TRAIN_GEOMS[(i//2)%3]
            rows.append({'episode_id':f'{split}_{i:04d}','seed':seed,'split':split,
                         'domain':'id','controller':['script','force_feedback'][i%2],
                         'params':parameters(seed,'id',geometry=geom)})
    tests=[]
    for domain,start in [('id',20350000),('geometry_ood',20351000),('physics_ood',20352000),('combined_ood',20353000)]:
        geometries=OOD_GEOMS if domain in ['geometry_ood','combined_ood'] else TRAIN_GEOMS
        for i in range(30):
            seed=start+i
            tests.append({'episode_id':f'{domain}_{i:03d}','seed':seed,'domain':domain,
                          'params':parameters(seed,domain,geometry=geometries[i%len(geometries)])})
    diagnostic=[]
    for i in range(12):
        seed=20364000+i
        diagnostic.append({'episode_id':f'diagnostic_{i:03d}','seed':seed,'domain':'id',
                           'params':parameters(seed,'id',geometry=TRAIN_GEOMS[i%3])})
    plan={'protocol_version':1,'prospective':True,'public_preregistration':False,
          'primary_outcome':'10cm for 10 consecutive observations before lowering, AND full-episode per-finger peak<=8N',
          'train_geometry_keys':TRAIN_GEOMS,'heldout_geometry_keys':OOD_GEOMS,
          'collection':rows,'control_test':tests,'counterfactual':diagnostic,
          'training_seeds':[0,1,2],'takeover':44,'control_end':135,'horizon':150,
          'bootstrap_resamples':2000,'bootstrap_seed':20401010,
          'main_control_dimensions':[2,6],'zero_intervention_dimensions':[0,1,3,4,5],
          'collection_force_feedback_target_N':5.,
          'notes':['All splits and original study seeds disjoint.',
                   'Five box geometries: aspect/size generalization, not arbitrary object classes.',
                   'Analytic friction screening is not an empirical success guarantee.'],
          'source_sha256':hashlib.sha256(Path(__file__).with_name('sim_adapter_v2.py').read_bytes()).hexdigest()}
    path=OUT/'protocol.json'
    payload=json.dumps(plan,indent=2,sort_keys=True)
    if path.exists():assert path.read_text()==payload, 'Protocol frozen; do not silently revise.'
    else:path.write_text(payload,encoding='utf-8')
    print(json.dumps({'collection':len(rows),'control_test':len(tests),'diagnostic':len(diagnostic),
                      'sha256':hashlib.sha256(payload.encode()).hexdigest()}))

if __name__=='__main__':main()
