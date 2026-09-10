from pathlib import Path
import argparse
import hashlib
import json
import time
from sim_adapter_v2 import SimAdapterV2, np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/revision_v2'

def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    args=p.parse_args()
    plan=json.loads((OUT/'protocol.json').read_text())
    rows=plan['collection'][args.shard::args.shards]
    target=OUT/'dataset/episodes';target.mkdir(parents=True,exist_ok=True)
    env=SimAdapterV2(render=True)
    tic=time.time()
    try:
        for i,row in enumerate(rows):
            path=target/(row['episode_id']+'.npz')
            if path.exists():
                with np.load(path,allow_pickle=False) as old:
                    assert int(old['seed'])==row['seed'] and old['action'].shape==(150,7)
                continue
            obs=env.reset(row['seed'],row['params']); states=[obs];actions=[];rewards=[]
            for t in range(150):
                action=env.script_action()
                if 44<=t<135:
                    action[[0,1,3,4,5]]=0.
                    if row['controller']=='force_feedback':
                        normal=float(obs['tactile'][[0,3]].mean())
                        action[6]=np.clip(.12*(5.-normal),-.25,.35)
                obs,reward,_,info=env.step(action)
                assert info['substeps']==25
                states.append(obs);actions.append(action.copy());rewards.append([reward])
            arrays={key:np.stack([s[key] for s in states]) for key in states[0]}
            arrays.update(action=np.stack(actions),reward=np.asarray(rewards,np.float32),
                          episode_id=np.array(row['episode_id']),seed=np.array(row['seed']),
                          split=np.array(row['split']),dt=np.array(.05),
                          params_json=np.array(json.dumps(env.params,sort_keys=True)),
                          collection_controller=np.array(row['controller']))
            assert np.max(np.abs(arrays['action'][44:135][:,[0,1,3,4,5]]))==0
            np.savez_compressed(path,**arrays)
            with (OUT/f'dataset/shard_{args.shard}.jsonl').open('a') as f:
                f.write(json.dumps({'episode_id':row['episode_id'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'height_max':float(arrays['height'].max()),'peak_N':float(arrays['substep_normal_peak'].max())})+'\n')
            if i%10==0 or i==len(rows)-1:
                print(json.dumps({'shard':args.shard,'completed':i+1,'total':len(rows),
                                  'elapsed':round(time.time()-tic,1)}),flush=True)
    finally:env.close()

if __name__=='__main__':main()
