"""Paired-seed evaluation in actual MuJoCo; no model-generated outcome scores.

All policies receive a common privileged scripted approach (steps0:44).
Only z / continuous gripper control changes during steps44:135. Delta xy and
rotation are zero for every controller; the final lowering segment is common. Full-episode
force peaks include setup and lowering. Reset(seed), not partial state snapshots,
is used separately for every controller.
"""
import argparse, base64, hashlib, json, pathlib, subprocess, time
from sim_adapter import SimAdapter, parameters, np


class PolicyBridge:
    def __init__(self, python, model, actor, out):
        self.log=(out/'policy_server_stderr.txt').open('w',encoding='utf-8')
        server=pathlib.Path(__file__).parents[1]/'world_model'/'controller_server.py'
        self.process=subprocess.Popen([python,str(server),'--model',model,'--actor',actor,'--device','cuda'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        ready=self.process.stdout.readline()
        if not ready or not json.loads(ready).get('ready'):
            raise RuntimeError('GPU bridge did not start; inspect policy_server_stderr.txt: '+ready)

    def request(self, payload):
        self.process.stdin.write(json.dumps(payload)+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line: raise RuntimeError('GPU bridge stopped; inspect stderr log')
        result=json.loads(line)
        if 'error' in result: raise RuntimeError(result['error'])
        return result

    def reset(self): self.request({'command':'reset'})

    def action(self,obs):
        result=self.request(dict(rgb_base64=base64.b64encode(obs['rgb'].tobytes()).decode('ascii'),
            tactile=obs['tactile'].tolist(),proprio=obs['proprio'].tolist()))
        action=np.asarray(result['action'],np.float32)
        if action.shape!=(7,) or not np.isfinite(action).all(): raise ValueError('Invalid policy action')
        return action

    def close(self):
        try:
            self.process.stdin.write(json.dumps({'command':'quit'})+'\n');self.process.stdin.flush()
            self.process.wait(timeout=15)
        finally:
            if self.process.poll() is None:self.process.terminate()
            self.log.close()


def evaluate(args):
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    bridge=PolicyBridge(args.gpu_python,args.model,args.actor,out) if args.controller=='policy' else None
    env=SimAdapter(); rows=[];tic=time.time()
    if args.resume and (out/'episodes.jsonl').exists():
        rows=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()]
        if len(rows)>args.episodes:raise ValueError('Resume prefix exceeds requested episode count')
        for index,row in enumerate(rows):
            expected_seed=args.seed_start+index
            expected_domain='ood' if args.domain=='ood' or (args.domain=='mixed' and index%2) else 'id'
            if row['seed']!=expected_seed or row['domain']!=expected_domain or row['controller']!=args.controller:
                raise ValueError('Resume prefix does not match requested seed/controller/domain')
            previous=np.load(out/f'episode_{expected_seed}.npz',allow_pickle=False)
            if previous['action'].shape!=(150,7) or previous['rgb'].shape!=(151,64,64,3):
                raise ValueError('Incomplete saved trajectory in resume prefix')
            if np.abs(previous['action'][args.takeover:args.intervention_end][:,[0,1,3,4,5]]).max()!=0:
                raise ValueError('Resume prefix uses a different action subspace')
            previous.close()
    resumed_episodes=len(rows)
    try:
        for index in range(resumed_episodes,args.episodes):
            seed=args.seed_start+index
            ood=(args.domain=='ood') or (args.domain=='mixed' and index%2==1)
            params=parameters(seed,ood=ood);obs=env.reset(seed,params)
            if bridge:bridge.reset()
            states=[obs];actions=[];rewards=[]
            for t in range(env.horizon):
                # Call the common script in every mode, preserving RNG sequence.
                action=env.script_action()
                predicted=None
                if bridge and args.takeover-2<=t<args.intervention_end:
                    predicted=bridge.action(obs)
                if args.takeover<=t<args.intervention_end:
                    # All compared controllers use the same 2D action subspace,
                    # matching the actor and learned-model imagination interface.
                    action[[0,1,3,4,5]]=0.
                    if args.controller=='policy': action[[2,6]]=predicted[[2,6]]
                    elif args.controller=='force_feedback':
                        normal=float(obs['tactile'][[0,3]].mean())
                        action[6]=np.clip(.12*(args.target_force-normal),-.25,.35)
                obs,reward,done,info=env.step(action)
                actions.append(action.copy());rewards.append(reward);states.append(obs)
            arrays={key:np.stack([s[key] for s in states]) for key in states[0]}
            run=0;longest=0
            for value in arrays['success'][1:136,0]:
                run=run+1 if value else 0;longest=max(longest,run)
            run=0;strict_longest=0
            for value in arrays['height'][1:136,0]>=.10:
                run=run+1 if value else 0;strict_longest=max(strict_longest,run)
            peak=float(arrays['substep_normal_peak'].max())
            setup_peak=float(arrays['substep_normal_peak'][:args.takeover+1].max())
            sustained=longest>=10
            row=dict(seed=seed,domain='ood' if ood else 'id',controller=args.controller,
                any_native_success=bool(arrays['success'].max()),
                sustained_native_success=sustained,native_success_longest_run_before_lower=longest,
                sustained_10cm_lift_success=bool(strict_longest>=10),
                sustained_10cm_lift_within_force_budget=bool(strict_longest>=10 and peak<=8.),
                final_native_success=bool(arrays['success'][-1,0]),
                force_budget=8.,full_episode_peak_normal_N=peak,force_budget_exceeded=peak>8.,
                sustained_success_within_force_budget=bool(sustained and peak<=8.),
                setup_peak_normal_N=setup_peak,setup_force_budget_exceeded=setup_peak>8.,
                intervention_peak_normal_N=float(arrays['substep_normal_peak'][args.takeover+1:args.intervention_end+1].max()),
                max_relative_height_m=float(arrays['height'].max()),native_shaped_return=float(sum(rewards)),
                bilateral_contact_at_takeover=bool(arrays['contact'][args.takeover].all()),
                sampled_slip_proxy_steps=int(arrays['slip'].sum()),params=env.params)
            rows.append(row)
            with (out/'episodes.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(row)+'\n')
            if args.save_trajectories or index==0:
                arrays.update(action=np.stack(actions),reward=np.asarray(rewards,np.float32)[:,None],
                    seed=np.array(seed),params_json=np.array(json.dumps(env.params)))
                np.savez_compressed(out/f'episode_{seed}.npz',**arrays)
            print(json.dumps(dict(seed=seed,controller=args.controller,domain=row['domain'],
                sustained_success=sustained,peak_normal_N=peak,elapsed_sec=round(time.time()-tic,1))),flush=True)
    finally:
        env.close()
        if bridge:bridge.close()
    metrics={}
    for domain in ['all','id','ood']:
        subset=[r for r in rows if domain=='all' or r['domain']==domain]
        if not subset:continue
        metrics[domain]=dict(n=len(subset),**{key:float(np.mean([r[key] for r in subset])) for key in [
            'any_native_success','sustained_native_success','final_native_success','force_budget_exceeded',
            'sustained_success_within_force_budget','setup_force_budget_exceeded','full_episode_peak_normal_N',
            'sustained_10cm_lift_success','sustained_10cm_lift_within_force_budget',
            'native_shaped_return','bilateral_contact_at_takeover']})
    summary=dict(controller=args.controller,metrics=metrics,seeds=[r['seed'] for r in rows],
        takeover=args.takeover,intervention_end=args.intervention_end,
        setup='Common privileged scripted approach before takeover; reset same seed per controller; no state snapshot branching',
        intervention='Only z/gripper allowed; delta xy/rotation identically zero for all controllers; common scripted lowering after intervention_end',
        control_dimensions=[2,6],zero_dimensions_during_intervention=[0,1,3,4,5],
        success='Native robosuite1.5.1 Lift cube center z > table z +0.04m for >=10 consecutive macrosteps ending by step135; strict10cm relative-lift metric reported separately',
        force='Maximum of per-finger summed normal contact force at all 0.002s substeps over complete150-step episode; 8N budget',
        caveat='Rigid-body simulation extension, not real robot, physical damage, or deployed safety guarantee',
        source_sha256=hashlib.sha256(pathlib.Path(__file__).with_name('sim_adapter.py').read_bytes()).hexdigest(),
        model=args.model,actor=args.actor,elapsed_sec=time.time()-tic,resumed_episodes=resumed_episodes)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--controller',choices=['script','force_feedback','policy'],default='script')
    p.add_argument('--out',required=True);p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed-start',type=int,default=20281000)
    p.add_argument('--domain',choices=['id','ood','mixed'],default='mixed')
    p.add_argument('--takeover',type=int,default=44);p.add_argument('--intervention-end',type=int,default=135)
    p.add_argument('--target-force',type=float,default=5.)
    p.add_argument('--gpu-python');p.add_argument('--model');p.add_argument('--actor')
    p.add_argument('--save-trajectories',action='store_true')
    p.add_argument('--resume',action='store_true',help='Continue a verified complete episode prefix without replaying it')
    args=p.parse_args()
    if args.controller=='policy' and not all([args.gpu_python,args.model,args.actor]):
        p.error('policy requires --gpu-python --model --actor')
    if (pathlib.Path(args.out)/'episodes.jsonl').exists() and not args.resume:
        p.error('Use a fresh --out directory to avoid mixing evaluations')
    evaluate(args)
