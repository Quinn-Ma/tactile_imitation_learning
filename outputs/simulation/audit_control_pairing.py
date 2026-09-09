"""Verify shared setup and identical 2D action domain from saved trajectories."""
import argparse,pathlib,json
from sim_adapter import np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).parent/'evaluation_main')
    args=parser.parse_args();root=args.root.resolve()
    reference=root/'script';rows=[]
    for summary_file in sorted(root.glob('*/summary.json')):
        summary=json.loads(summary_file.read_text());folder=summary_file.parent
        assert summary['takeover']==44 and summary['intervention_end']==135
        assert summary['zero_dimensions_during_intervention']==[0,1,3,4,5]
        maxima={key:0. for key in ['rgb','proprio','tactile','height','action']}
        for seed in summary['seeds']:
            data=np.load(folder/f'episode_{seed}.npz',allow_pickle=False)
            ref=np.load(reference/f'episode_{seed}.npz',allow_pickle=False)
            assert str(data['params_json'])==str(ref['params_json']),(folder.name,seed,'params')
            for key in maxima:
                end=44 if key=='action' else 45
                error=float(np.abs(data[key][:end].astype(float)-ref[key][:end]).max())
                maxima[key]=max(maxima[key],error)
            assert np.isfinite(data['action']).all()
            assert np.abs(data['action'][44:135][:,[0,1,3,4,5]]).max()==0,(folder.name,seed,'nonzero excluded action')
        assert max(maxima.values())==0,(folder.name,maxima)
        rows.append(dict(run=folder.name,episodes=len(summary['seeds']),max_shared_setup_error=maxima,
                         restricted_dimensions_exact_zero=True))
    result=dict(completed_runs=len(rows),checks='PASS: paired parameters, bit-identical pre-intervention RGB/proprio/tactile/height/actions, exact zero excluded actions',runs=rows)
    (root/'pairing_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
