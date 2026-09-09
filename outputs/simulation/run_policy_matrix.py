"""Run the fixed 15-policy test plan; never alter checkpoints or tune on tests."""
import argparse,json,pathlib,subprocess,time,sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu-python',required=True)
parser.add_argument('--parallel',type=int,default=2)
parser.add_argument('--out',type=pathlib.Path,default=ROOT/'work/reproduce_policy_matrix')
parser.add_argument('--runs',type=pathlib.Path,default=ROOT/'outputs/world_model/runs')
parser.add_argument('--actors',type=pathlib.Path,default=ROOT/'outputs/world_model/rl')
args=parser.parse_args()
if args.parallel<1:parser.error('--parallel must be positive')
GPU=args.gpu_python
OUT=args.out.resolve();RUNS=args.runs.resolve();ACTORS=args.actors.resolve()
PLAN=[]
for seed in range(3):
    for variant in ['vision','visuotactile']:
        for kind in ['bc','rl']:
            name=f'{variant}_seed{seed}_{kind}'
            PLAN.append(dict(name=name,model=str(RUNS/f'{variant}_seed{seed}'/'best.pt'),
                actor=str(ACTORS/f'{variant}_seed{seed}'/f'{kind}_actor.pt')))
    PLAN.append(dict(name=f'visuotactile_margin_seed{seed}_rl',
        model=str(RUNS/f'visuotactile_seed{seed}'/'best.pt'),
        actor=str(ACTORS/f'visuotactile_margin_seed{seed}'/'rl_actor.pt')))
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'policy_evaluation_plan.json').write_text(json.dumps(dict(
    models=PLAN,episodes=20,seed_start=20281000,domain='mixed',takeover=44,intervention_end=135,
    control_dimensions=[2,6],zero_dimensions=[0,1,3,4,5],
    no_adaptation='All hyperparameters/checkpoints fixed before testing; no tuning from evaluation results'),indent=2))
pending=PLAN.copy();active={};finished=[];started=time.time();last_status=0
while pending or active:
    for name,entry in list(active.items()):
        code=entry['process'].poll()
        if code is None:continue
        entry['log'].close();del active[name]
        summary=OUT/name/'summary.json'
        if code!=0 or not summary.exists():
            print(json.dumps(dict(event='failed',name=name,exit_code=code,console=str(OUT/name/'console.log'))),flush=True)
            for e in active.values():e['process'].terminate();e['log'].close()
            raise SystemExit(1)
        measured=json.loads(summary.read_text())
        finished.append(name)
        print(json.dumps(dict(event='complete',name=name,elapsed_sec=round(time.time()-started,1),
                              metrics=measured['metrics'])),flush=True)
    for row in list(pending):
        if len(active)>=args.parallel:break
        out=OUT/row['name'];summary=out/'summary.json'
        if summary.exists():
            print(json.dumps(dict(event='existing_complete',name=row['name'])),flush=True)
            pending.remove(row);finished.append(row['name']);continue
        model=ROOT/row['model'];actor=ROOT/row['actor']
        if not all(p.exists() and p.stat().st_size>0 and time.time()-p.stat().st_mtime>2 for p in [model,actor]):continue
        out.mkdir(parents=True,exist_ok=True)
        log=(out/'console.log').open('w',encoding='utf-8')
        command=[sys.executable,str(ROOT/'outputs/simulation/evaluate_controller.py'),
            '--controller','policy','--episodes','20','--takeover','44','--intervention-end','135',
            '--seed-start','20281000','--out',str(out),'--gpu-python',GPU,
            '--model',str(model),'--actor',str(actor),'--save-trajectories']
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        active[row['name']]=dict(process=process,log=log);pending.remove(row)
        print(json.dumps(dict(event='started',name=row['name'],pid=process.pid)),flush=True)
    if time.time()-last_status>=30:
        print(json.dumps(dict(event='status',finished=len(finished),active=list(active),pending=len(pending),
                              elapsed_sec=round(time.time()-started,1))),flush=True)
        last_status=time.time()
    if pending and not active and time.time()-started>1800:
        raise TimeoutError('Required checkpoints did not appear within30minutes')
    if pending or active:time.sleep(2)
print(json.dumps(dict(event='all_complete',evaluations=finished,elapsed_sec=time.time()-started)),flush=True)
