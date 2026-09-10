"""Execute the frozen revision-v2 protocol; outcomes always come from MuJoCo.

Common privileged approach (0..43), then identical z/gripper action space
(44..134), then common lowering (135..149). All150 intervals contribute to
force peaks. No policy receives object pose/height, physical parameters, native
reward, episode seed or future sensor values. RGB/touch/proprio only; adaptive
controllers also get a Boolean marking the common intervention boundary.
"""
from pathlib import Path
import argparse,base64,hashlib,json,subprocess,time
from sim_adapter_v2 import SimAdapterV2,np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def longest_run(values):
    run=best=0
    for value in values:
        run=run+1 if bool(value) else 0;best=max(best,run)
    return best

class Bridge:
    def __init__(self,method,python,out):
        kind=method['kind'];self.adaptive=kind=='adaptive'
        if kind=='world_model':
            command=[ROOT/'outputs/world_model/controller_server.py','--model',ROOT/method['model'],
                     '--actor',ROOT/method['actor']]
        elif kind=='reactive':command=[OUT/'reactive_server.py','--checkpoint',ROOT/method['checkpoint']]
        elif kind=='adaptive':
            command=[OUT/'adaptive_controller.py','--mode',method['mode']]
            if method.get('model'):command+=['--model',ROOT/method['model']]
        else:raise ValueError(kind)
        command+=['--device','cuda']
        self.log=(out/'policy_stderr.log').open('a',encoding='utf-8')
        self.process=subprocess.Popen([python]+list(map(str,command)),cwd=ROOT,stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        ready=self.process.stdout.readline()
        if not ready or not json.loads(ready).get('ready'):
            self.close();raise RuntimeError('Policy bridge failed; inspect '+str(out/'policy_stderr.log'))
        self.ready=json.loads(ready)

    def request(self,payload):
        self.process.stdin.write(json.dumps(payload)+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Policy bridge closed unexpectedly')
        result=json.loads(line)
        if 'error' in result:raise RuntimeError(result['error'])
        return result

    def reset(self):self.request({'command':'reset'})

    def action(self,obs,active):
        payload=dict(rgb_base64=base64.b64encode(obs['rgb'].tobytes()).decode('ascii'),
                     tactile=obs['tactile'].tolist(),proprio=obs['proprio'].tolist())
        if self.adaptive:payload['control_active']=active
        start=time.perf_counter();result=self.request(payload);elapsed=time.perf_counter()-start
        action=np.asarray(result['action'],np.float32)
        assert action.shape==(7,) and np.isfinite(action).all()
        assert np.max(np.abs(action[[0,1,3,4,5]]))==0
        return action,result,elapsed

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.write('{"command":"quit"}\n');self.process.stdin.flush()
                self.process.wait(timeout=15)
        except (BrokenPipeError,subprocess.TimeoutExpired):
            if self.process.poll() is None:self.process.terminate()
        finally:self.log.close()

def evaluate(args):
    plan=json.loads((OUT/'protocol.json').read_text())
    method=next(m for m in json.loads((OUT/'methods.json').read_text()) if m['name']==args.method)
    cases=plan['control_test']
    if args.smoke:
        cases=[r for r in plan['collection'] if r['split']=='val'][:1]
    out=Path(args.out) if args.out else OUT/'evaluations'/method['name']
    if args.smoke and not args.out:raise ValueError('Smoke must use separate --out')
    out.mkdir(parents=True,exist_ok=True)
    inputs={'method':method,'smoke':args.smoke,'protocol_sha256':sha(OUT/'protocol.json'),
            'evaluator_sha256':sha(__file__),'adapter_sha256':sha(OUT/'sim_adapter_v2.py'),
            'original_adapter_sha256':sha(ROOT/'outputs/simulation/sim_adapter.py'),
            'cases':[c['seed'] for c in cases],
            'model_inputs':{k:sha(ROOT/method[k]) for k in ['model','actor','checkpoint'] if k in method}}
    inputs_file=out/'inputs.json'
    if inputs_file.exists():assert json.loads(inputs_file.read_text())==inputs,'Changed run inputs; refuse mixed resume'
    else:inputs_file.write_text(json.dumps(inputs,indent=2))
    rows=[];records=out/'episodes.jsonl'
    if records.exists():
        if not args.resume:raise ValueError('Use --resume for a completed prefix')
        rows=[json.loads(line) for line in records.read_text().splitlines() if line]
        assert len(rows)<=len(cases)
        for row,case in zip(rows,cases):
            assert row['seed']==case['seed'] and row['method']==method['name']
            saved=out/f'episode_{row["seed"]}.npz'
            assert sha(saved)==row['trajectory_sha256']
    if len(rows)==len(cases):return
    bridge=Bridge(method,args.gpu_python,out) if method['kind'] in ['world_model','reactive','adaptive'] else None
    env=SimAdapterV2(render=True);started=time.time()
    try:
        for index,case in list(enumerate(cases))[len(rows):]:
            obs=env.reset(case['seed'],params=case['params'])
            if bridge:bridge.reset()
            states=[obs];actions=[];rewards=[];latencies=[];guard_count=0;fallback_count=0
            for t in range(150):
                action=env.script_action() # Keeps nuisance RNG common in all methods.
                if bridge and 42<=t<135:
                    predicted,reply,elapsed=bridge.action(obs,t>=44)
                    if t>=44:
                        latencies.append(elapsed);guard_count+=int(reply.get('guard_intervened',False))
                        fallback_count+=int(reply.get('fallback_to_reactive',False))
                if 44<=t<135:
                    action[[0,1,3,4,5]]=0.
                    if bridge:action[[2,6]]=predicted[[2,6]]
                    elif method['kind']=='force_feedback':
                        action[6]=np.clip(.12*(5.-float(obs['tactile'][[0,3]].mean())),-.25,.35)
                obs,reward,_,info=env.step(action)
                assert info['substeps']==25
                states.append(obs);actions.append(action.copy());rewards.append([reward])
            arrays={key:np.stack([s[key] for s in states]) for key in states[0]}
            arrays.update(action=np.stack(actions),reward=np.asarray(rewards,np.float32),seed=np.array(case['seed']),
                episode_id=np.array(case['episode_id']),split=np.array('test_'+case['domain']),
                params_json=np.array(json.dumps(env.params,sort_keys=True)),dt=np.array(.05))
            assert np.max(np.abs(arrays['action'][44:135][:,[0,1,3,4,5]]))==0
            strict_longest=longest_run(arrays['height'][1:136,0]>=.10)
            peak=float(arrays['substep_normal_peak'].max())
            path=out/f'episode_{case["seed"]}.npz'
            np.savez_compressed(path,**arrays) # Persist trajectory before recording completion.
            row=dict(seed=case['seed'],domain=case['domain'],geometry=env.params['geometry'],method=method['name'],
                group=method['group'],strict_lift=bool(strict_longest>=10),joint_success=bool(strict_longest>=10 and peak<=8.),
                force_violation=bool(peak>8.),peak_N=peak,max_height_m=float(arrays['height'].max()),strict_longest=strict_longest,
                setup_peak_N=float(arrays['substep_normal_peak'][:45].max()),
                intervention_peak_N=float(arrays['substep_normal_peak'][45:136].max()),
                lower_peak_N=float(arrays['substep_normal_peak'][136:].max()),
                bilateral_contact_at_takeover=bool(arrays['contact'][44].all()),
                native_success=bool(longest_run(arrays['success'][1:136,0])>=10),
                native_return=float(np.sum(rewards)),guard_interventions=guard_count,model_fallbacks=fallback_count,
                policy_roundtrip_ms_mean=float(np.mean(latencies)*1000) if latencies else None,
                policy_roundtrip_ms_p95=float(np.percentile(latencies,95)*1000) if latencies else None,
                timing_caveat='Concurrent local experiments; IPC and synchronization included; not an isolated latency benchmark',
                params=env.params,trajectory_sha256=sha(path))
            with records.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
            rows.append(row)
            if (index+1)%10==0 or index+1==len(cases):
                print(json.dumps({'method':method['name'],'completed':index+1,'total':len(cases),
                                  'elapsed_seconds':round(time.time()-started,1)}),flush=True)
    finally:
        env.close()
        if bridge:bridge.close()
    (out/'complete.json').write_text(json.dumps({'episodes':len(rows),'elapsed_seconds':time.time()-started,
        'inputs':inputs,'bridge_ready':bridge.ready if bridge else None},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',required=True)
    p.add_argument('--gpu-python');p.add_argument('--out');p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true');evaluate(p.parse_args())
