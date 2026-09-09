"""Generate deterministic episode files from actual robosuite/MuJoCo dynamics."""
import argparse, pathlib, time, json, hashlib
from sim_adapter import SimAdapter, parameters, np, mujoco, suite


def split_for(i):
    cycle=['train','train','val','train','calibration','train','test_id','train','calibration','test_ood']*3
    cycle+=['train','train','val','train','test_id','train','test_ood','train','test_id','test_ood']
    return cycle[i%40]


def generate(out, n, start=0):
    out=pathlib.Path(out); (out/'episodes').mkdir(parents=True,exist_ok=True)
    if start==0: (out/'generation_log.jsonl').write_text('',encoding='utf-8')
    env=SimAdapter(); summaries=[]; tic=time.time()
    for i in range(start,start+n):
        split=split_for(i); seed=20270915+i; p=parameters(seed,ood=split=='test_ood')
        obs=env.reset(seed,p); p=env.params.copy(); states=[obs]; actions=[];rewards=[]
        for t in range(env.horizon):
            action=env.script_action()
            obs,r,done,info=env.step(action)
            actions.append(action);rewards.append([r]);states.append(obs)
        arrays={key:np.stack([s[key] for s in states]) for key in states[0]}
        arrays.update(action=np.stack(actions),reward=np.asarray(rewards,np.float32),
            episode_id=np.array(f'ep_{i:04d}'),split=np.array(split),seed=np.array(seed),dt=np.array(env.dt),
            params_json=np.array(json.dumps(p,sort_keys=True)))
        f=out/'episodes'/f'ep_{i:04d}.npz';np.savez_compressed(f,**arrays)
        runs=0; sustained=False
        for v in arrays['success'][1:136,0]:
            runs=runs+1 if v else 0
            sustained=sustained or runs>=10
        row=dict(episode_id=f.stem,seed=seed,split=split,success=bool(arrays['success'].max()),
            sustained_success_10steps=bool(sustained),
            final_success=bool(arrays['success'][-1,0]),max_height=float(arrays['height'].max()),
            peak_normal_force=float(arrays['substep_normal_peak'].max()),
            force_budget_exceeded=bool(arrays['substep_normal_peak'].max()>p['force_budget']),
            contact_steps=int(arrays['contact'].all(axis=1).sum()),mass=p['mass'],friction=p['friction'],
            actuator_cap=p['actuator_cap'],sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        summaries.append(row)
        with (out/'generation_log.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(row)+'\n')
        print(json.dumps(dict(**row,elapsed_sec=round(time.time()-tic,1))),flush=True)
        if i==start:
            import imageio.v2 as imageio
            imageio.mimsave(out/'first_episode.gif',[np.repeat(np.repeat(im,3,0),3,1) for im in arrays['rgb'][::3]],duration=.15,loop=0)
    env.close()
    manifest=dict(task='robosuite Lift with explicitly documented extensions',robosuite_version=suite.__version__,
        mujoco_version=mujoco.__version__,horizon=150,control_hz=20,
        extensions=['continuous gripper rate replaces sign-only rate','mass/friction and actuator force cap randomized; compiled cube geometry fixed; mj_setConst after mass/inertia changes',
                    'fixed task-centered camera','solver contact signals and full internal substep normal peaks logged'],
        no_claims=['not soft-body deformation','not physical object damage','not measured tactile sensor imagery','not physical robot experiments'],
        tactile_channels=['left_normal_N','left_shear_up_N','left_shear_cross_N','right_normal_N','right_shear_up_N','right_shear_cross_N'],
        action='7D OSC_POSE delta xyz and rotation, scaled [0.05m,0.5rad], then continuous gripper rate [-1,1]',
        alignment='state[t],action[t],reward[t],state[t+1]; reward is unmodified robosuite shaped Lift reward',
        scripted_setup='privileged cube pose used by collection script; not model input; main control evaluation common approach actions0:44 and intervention44:135',
        phase={'0':'approach','1':'close','2':'micro_lift_probe','3':'lift','4':'hold','5':'lower'},
        sampling='force peak for each policy interval measured at all 25 internal 0.002s MuJoCo steps',
        success='native robosuite1.5.1: cube center z > table z +0.04m; any-time, sustained10steps and final metrics distinct; strict relative10cm sustained10steps in dataset_audit/evaluation',
        model_construction_seed=20270915,force_budget_N=8.,
        split_policy='episode index mod40:20train4val6calibration5test_id5test_ood; all seeds disjoint; OOD mass/friction distribution shifted; geometry fixed',
        seeds=[r['seed'] for r in summaries],generated_this_invocation=len(summaries),seconds=time.time()-tic,
        sources=['https://robosuite.ai/docs/overview.html','https://robosuite.ai/docs/installation.html',
                 'https://github.com/ARISE-Initiative/robosuite','https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(pathlib.Path(__file__).parent));p.add_argument('--episodes',type=int,default=30);p.add_argument('--start',type=int,default=0)
    args=p.parse_args();generate(args.out,args.episodes,args.start)
