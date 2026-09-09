"""Standard-library integrity and recorded-protocol checks; no experiment runs."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parent
index=json.loads((root/'FILES_SHA256.json').read_text())
for name,digest in index['files'].items():
    path=root/name
    assert path.is_file(),name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,(name,'hash mismatch')
    assert path.suffix not in ('.pt','.pth','.npz','.npy','.pkl','.pdf','.tex','.zip'),name
    if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'),filename=name)
    text=path.read_text(encoding='utf-8-sig')
    assert not re.search(r'\b[A-Za-z]:[/\\]',text),(name,'absolute Windows path')
split=json.loads((root/'recorded/outputs/world_model/runs/data_manifest.json').read_text())
assert Counter(row['split'] for row in split)=={'train':80,'val':16,'calibration':24,'test_id':20,'test_ood':20}
total=0
for cohort,start in [('evaluation_main',20281000),('evaluation_goal_aligned',20291000)]:
    directory=root/'recorded/outputs/simulation'/cohort
    summaries=sorted(directory.glob('*/summary.json'))
    assert len(summaries)==17,(cohort,len(summaries))
    for path in summaries:
        summary=json.loads(path.read_text())
        rows=[json.loads(line) for line in path.with_name('episodes.jsonl').read_text().splitlines() if line]
        assert len(rows)==20
        assert {r['seed'] for r in rows}==set(range(start,start+20))
        assert Counter(r['domain'] for r in rows)=={'id':10,'ood':10}
        assert summary['takeover']==44 and summary['intervention_end']==135
        assert summary['control_dimensions']==[2,6]
        assert summary['zero_dimensions_during_intervention']==[0,1,3,4,5]
        for domain in ('all','id','ood'):
            selected=[r for r in rows if domain=='all' or r['domain']==domain]
            for key in ('sustained_10cm_lift_success','sustained_10cm_lift_within_force_budget','force_budget_exceeded'):
                assert abs(sum(r[key] for r in selected)/len(selected)-summary['metrics'][domain][key])<1e-12
        total+=len(rows)
print(json.dumps({'integrity':'PASS','files':len(index['files']),'model_dataset_episodes':len(split),
                  'controller_runs':34,'executions':total,'distinct_evaluation_environment_seeds':40,
                  'note':'Recorded-result verification only; no new training or simulation.'},indent=2))
