"""Fixed round2: align imagined task reward with10cm; use fresh environment seeds.

The round1 dataset, models and evaluations remain unchanged. This script never
alters a checkpoint or trains a model in response to round2 test outcomes.
"""
import argparse,datetime,hashlib,json,pathlib,subprocess,sys,time

ROOT=pathlib.Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu-python',required=True)
parser.add_argument('--parallel',type=int,default=2)
parser.add_argument('--out',type=pathlib.Path,default=ROOT/'work/reproduce_goal_aligned_matrix')
parser.add_argument('--runs',type=pathlib.Path,default=ROOT/'outputs/world_model/runs')
parser.add_argument('--bc-actors',type=pathlib.Path,default=ROOT/'outputs/world_model/rl')
parser.add_argument('--new-actors',type=pathlib.Path,default=ROOT/'outputs/world_model/rl_goal_aligned')
args=parser.parse_args()
if args.parallel<1:parser.error('--parallel must be positive')
OUT=args.out.resolve();RUNS=args.runs.resolve()
BC=args.bc_actors.resolve();NEW=args.new_actors.resolve()
PLAN=[dict(name='script',controller='script'),dict(name='force_feedback',controller='force_feedback')]
for seed in range(3):
    PLAN.append(dict(name=f'visuotactile_seed{seed}_bc',controller='policy',
        model=str(RUNS/f'visuotactile_seed{seed}'/'best.pt'),
        actor=str(BC/f'visuotactile_seed{seed}'/'bc_actor.pt')))
for reward_name,actors in [('native',BC),('height',NEW)]:
    for seed in range(3):
        for variant in ['visuotactile','visuotactile_margin']:
            PLAN.append(dict(name=f'{variant}_{reward_name}_seed{seed}_rl',controller='policy',
                model=str(RUNS/f'visuotactile_seed{seed}'/'best.pt'),
                actor=str(actors/f'{variant}_seed{seed}'/'rl_actor.pt')))
OUT.mkdir(parents=True,exist_ok=True)
protocol=dict(experiment='goal_aligned_round2',models=PLAN,episodes=20,seed_start=20291000,
    domain='mixed',takeover=44,intervention_end=135,control_dimensions=[2,6],
    zero_dimensions=[0,1,3,4,5],
    single_intended_change='Imagined actor task reward changes from predicted native reward to clamp(pred_height/0.1,0,1); other specified training settings fixed',
    prior_observation='Round1 native Lift height threshold is lower than strict10cm evaluation; results and failure cases retained',
    no_adaptation='Round2 settings and new environment seeds fixed before reading round2 test outcomes; no test-driven parameter changes',
    revision_reason='Expanded11 to17 runs for paired reward-only comparison on identical new environment seeds; completed baselines and BC episode prefixes retained',
    prior_plan='goal_aligned_evaluation_plan_initial11.json',
    simulator_sha256=hashlib.sha256((ROOT/'outputs/simulation/sim_adapter.py').read_bytes()).hexdigest(),
    evaluator_sha256=hashlib.sha256((ROOT/'outputs/simulation/evaluate_controller.py').read_bytes()).hexdigest())
planfile=OUT/'goal_aligned_evaluation_plan.json'
if planfile.exists():
    previous=json.loads(planfile.read_text())
    comparison={k:v for k,v in previous.items() if k!='created_utc'}
    if comparison!=protocol:raise ValueError('Existing output has a different fixed plan; select a new --out')
else:
    protocol['created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    planfile.write_text(json.dumps(protocol,indent=2))
pending=PLAN.copy();active={};finished=[];started=time.time();last_status=0
while pending or active:
    for name,entry in list(active.items()):
        code=entry['process'].poll()
        if code is None:continue
        entry['log'].close();del active[name]
        summary=OUT/name/'summary.json'
        if code!=0 or not summary.exists():
            print(json.dumps(dict(event='failed',name=name,exit_code=code,console=str(OUT/name/'console.log'))),flush=True)
            for item in active.values():item['process'].terminate();item['log'].close()
            raise SystemExit(1)
        measured=json.loads(summary.read_text());finished.append(name)
        print(json.dumps(dict(event='complete',name=name,elapsed_sec=round(time.time()-started,1),
            strict10cm=measured['metrics']['all']['sustained_10cm_lift_success'],
            strict10cm_within_budget=measured['metrics']['all']['sustained_10cm_lift_within_force_budget'],
            force_exceeded=measured['metrics']['all']['force_budget_exceeded'])),flush=True)
    for row in list(pending):
        if len(active)>=args.parallel:break
        out=OUT/row['name'];summary=out/'summary.json'
        if summary.exists():
            completed=json.loads(summary.read_text())
            if completed['seeds']!=list(range(20291000,20291020)):raise ValueError('Mismatched completed test seeds')
            print(json.dumps(dict(event='existing_complete',name=row['name'])),flush=True)
            pending.remove(row);finished.append(row['name']);continue
        if row['controller']=='policy':
            sources=[pathlib.Path(row[key]) for key in ['model','actor']]
            if not all(path.exists() and path.stat().st_size>0 and time.time()-path.stat().st_mtime>2 for path in sources):continue
        out.mkdir(parents=True,exist_ok=True)
        resume=(out/'episodes.jsonl').exists()
        log=(out/'console.log').open('a' if resume else 'w',encoding='utf-8')
        command=[sys.executable,str(ROOT/'outputs/simulation/evaluate_controller.py'),
            '--controller',row['controller'],'--episodes','20','--takeover','44','--intervention-end','135',
            '--seed-start','20291000','--domain','mixed','--out',str(out),'--save-trajectories']
        if resume:command+=['--resume']
        if row['controller']=='policy':
            command+=['--gpu-python',args.gpu_python,'--model',row['model'],'--actor',row['actor']]
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        active[row['name']]=dict(process=process,log=log);pending.remove(row)
        print(json.dumps(dict(event='started',name=row['name'],pid=process.pid)),flush=True)
    if time.time()-last_status>=30:
        print(json.dumps(dict(event='status',finished=len(finished),active=list(active),pending=len(pending),
            elapsed_sec=round(time.time()-started,1))),flush=True);last_status=time.time()
    if pending and not active and time.time()-started>1800:
        raise TimeoutError('Required checkpoints did not appear within30minutes')
    if pending or active:time.sleep(2)
print(json.dumps(dict(event='all_complete',evaluations=finished,elapsed_sec=time.time()-started)),flush=True)
