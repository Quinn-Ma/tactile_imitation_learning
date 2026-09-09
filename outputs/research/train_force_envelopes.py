"""Real public GelSight force regression and offline selective envelope evaluation.
No robot control, measured slip success, material damage, or world-model claim.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn

MUS=np.array([0.3,0.5,0.8],dtype=np.float64)
CEILINGS=[0.5,1.0,2.0,3.0]  # Analysis parameters, not measured damage thresholds.
ALPHAS=[0.20,0.10,0.05]


class SmallForceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[];cin=3
        for cout in [24,48,96,128]:
            layers += [nn.Conv2d(cin,cout,3,stride=2,padding=1),nn.GroupNorm(8,cout),nn.SiLU()]
            cin=cout
        self.features=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(128*4*3,128),nn.SiLU(),nn.Linear(128,3))
    def forward(self,x):
        return self.head(self.features(x))


def group_splits(names):
    """Fixed trajectory split reused across all model seeds and calibrators."""
    rng=np.random.default_rng(314159)
    result={k:[] for k in ["train","validation","calibration","test_id","test_batch6","test_flat","test_sharp"]}
    for batch in range(1,6):
        ids=np.array([i for i,name in enumerate(names) if name.startswith(f"sphere/batch_{batch}/")])
        ids=rng.permutation(ids);n=len(ids);a=int(.60*n);b=int(.70*n);c=int(.85*n)
        for split,part in zip(["train","validation","calibration","test_id"],[ids[:a],ids[a:b],ids[b:c],ids[c:]]):
            result[split].extend(part.tolist())
    for i,name in enumerate(names):
        if name.startswith("sphere/batch_6/"):result["test_batch6"].append(i)
        if name.startswith("flat/"):result["test_flat"].append(i)
        if name.startswith("sharp/"):result["test_sharp"].append(i)
    all_ids=[i for part in result.values() for i in part]
    assert len(all_ids)==len(set(all_ids))==len(names), "Trajectory split overlaps or incomplete."
    return result


def qfinite(scores,alpha):
    scores=np.sort(np.asarray(scores));k=math.ceil((len(scores)+1)*(1-alpha))
    return float(max(0,scores[k-1])) if k<=len(scores) else float("inf")


def group_max(values,groups):
    return np.array([values[groups==g].max() for g in np.unique(groups)])


def gforce(force):
    return np.linalg.norm(force[:,:2],axis=1)[:,None]-force[:,2,None]*MUS[None,:]


def calibrated_bounds(pred,method,q):
    ghat=gforce(pred)
    if method=="trajectory_coordinate_box":
        upper=np.linalg.norm(np.abs(pred[:,:2])+q,axis=1)[:,None]-(pred[:,2,None]-q)*MUS[None,:]
        normal_upper=pred[:,2]+q
    else:
        upper=ghat+q;normal_upper=pred[:,2]+q
    return upper,normal_upper


def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--batch-size",type=int,default=256)
    ap.add_argument("--seeds",type=int,nargs="+",default=[17,29,43])
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError("This experiment requires CUDA; no silent fallback.")
    start=time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    names=json.loads((args.data/"trajectory_names.json").read_text(encoding="utf-8"))
    splits=group_splits(names)
    traj=np.load(args.data/"trajectory_ids.npy")
    labels=np.load(args.data/"forces.npy").astype(np.float32)
    indexes={k:np.flatnonzero(np.isin(traj,ids)) for k,ids in splits.items()}
    frames=np.load(args.data/"images.npy",mmap_mode="r")
    images_gpu=torch.as_tensor(np.array(frames),device="cuda",dtype=torch.uint8)
    forces_gpu=torch.as_tensor(labels,device="cuda")
    train_ix=torch.as_tensor(indexes["train"],device="cuda")
    mean=forces_gpu[train_ix].mean(0);std=forces_gpu[train_ix].std(0).clamp_min(.01)
    # Fixed pixel normalization uses no calibration or test information.
    def x_at(ix):return images_gpu[ix].float().div_(127.5).sub_(1)
    split_record={k:{"trajectory_ids":ids,"trajectory_names":[names[i] for i in ids],"frames":len(indexes[k])} for k,ids in splits.items()}
    (args.out/"real_force_splits.json").write_text(json.dumps(split_record,indent=2),encoding="utf-8")
    metric_rows=[];envelope_rows=[];selection_rows=[];cal_rows=[];logs=[]
    all_ix=torch.arange(len(labels),device="cuda")
    for seed in args.seeds:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=SmallForceCNN().cuda()
        optim=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=args.epochs,eta_min=1e-4)
        best_loss=float("inf");best_epoch=-1;best=None
        gen=torch.Generator(device="cuda").manual_seed(seed+1000)
        val_ix=torch.as_tensor(indexes["validation"],device="cuda")
        def predict(indices):
            model.eval();outs=[]
            with torch.no_grad():
                for chunk in indices.split(args.batch_size*2):
                    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                        pred=model(x_at(chunk))
                    outs.append(pred.float()*std+mean)
            return torch.cat(outs)
        seed_start=time.perf_counter()
        for epoch in range(args.epochs):
            model.train();order=train_ix[torch.randperm(len(train_ix),generator=gen,device="cuda")]
            running=0.
            for batch in order.split(args.batch_size):
                optim.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    out=model(x_at(batch));target=(forces_gpu[batch]-mean)/std
                    loss=nn.functional.mse_loss(out,target)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.0);optim.step()
                running+=float(loss.detach().item())*len(batch)
            scheduler.step()
            pred=predict(val_ix)
            val_loss=float(((pred-forces_gpu[val_ix]).abs().mean()).item())
            logs.append({"seed":seed,"epoch":epoch+1,"train_standardized_mse":running/len(train_ix),"validation_mae_N":val_loss})
            if val_loss<best_loss:
                best_loss=val_loss;best_epoch=epoch+1
                best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch==0 or (epoch+1)%5==0:
                print(f"seed {seed}, epoch {epoch+1}/{args.epochs}, val MAE {val_loss:.4f} N, elapsed {time.perf_counter()-seed_start:.1f}s",flush=True)
        model.load_state_dict(best)
        prediction=predict(all_ix).cpu().numpy()
        torch.save({"state_dict":best,"force_mean_N":mean.cpu(),"force_std_N":std.cpu(),"seed":seed,"selected_epoch":best_epoch,"input_shape":[3,64,48]},args.out/f"real_force_model_seed{seed}.pt")
        split_code=np.zeros(len(labels),dtype=np.int8)
        for i,(split,ix) in enumerate(indexes.items()):split_code[ix]=i
        np.savez_compressed(args.out/f"real_force_predictions_seed{seed}.npz",prediction_N=prediction,label_N=labels,trajectory_id=traj,split_code=split_code,source_frame_index=np.load(args.data/"source_frame_indexes.npy"))
        train_mean=mean.cpu().numpy()
        for split,ix in indexes.items():
            for method,hat in [("cnn",prediction[ix]),("train_mean",np.broadcast_to(train_mean,labels[ix].shape))]:
                err=hat-labels[ix]
                metric_rows.append({"seed":seed,"split":split,"method":method,"frames":len(ix),"trajectories":len(splits[split]),"mae_Fx_N":float(abs(err[:,0]).mean()),"mae_Fy_N":float(abs(err[:,1]).mean()),"mae_Fn_N":float(abs(err[:,2]).mean()),"mae_xyz_mean_N":float(abs(err).mean()),"rmse_xyz_N":float(np.sqrt((err**2).mean())),"selected_epoch":best_epoch})
        cal_ix=indexes["calibration"];cy=labels[cal_ix];cp=prediction[cal_ix];cg=traj[cal_ix]
        # g_mu errors are affine in mu, so endpoint calibration covers every
        # mu in [.3,.8] with this common additive bound, including .5.
        constraint_errors=np.column_stack([gforce(cy)[:,[0,2]]-gforce(cp)[:,[0,2]],cy[:,2]-cp[:,2]])
        frame_score=constraint_errors.max(1)
        traj_score=group_max(frame_score,cg)
        coord_score=group_max(np.abs(cy-cp).max(1),cg)
        for alpha in ALPHAS:
            qs={"point":0.0,"frame_constraint":qfinite(frame_score,alpha),"trajectory_coordinate_box":qfinite(coord_score,alpha),"trajectory_constraint":qfinite(traj_score,alpha)}
            for method,q in qs.items():
                cal_rows.append({"seed":seed,"alpha":alpha,"method":method,"q_N":q,"calibration_trajectories":len(np.unique(cg)),"calibration_frames":len(cg)})
                for split in ["test_id","test_batch6","test_flat","test_sharp"]:
                    ix=indexes[split];y=labels[ix];p=prediction[ix];groups=traj[ix]
                    upper,normal_upper=calibrated_bounds(p,method,q)
                    true_g=gforce(y)
                    covered=(true_g<=upper+1e-8).all(1)&(y[:,2]<=normal_upper+1e-8)
                    group_coverage=np.array([covered[groups==g].all() for g in np.unique(groups)])
                    envelope_rows.append({"seed":seed,"alpha":alpha,"split":split,"method":method,"frame_joint_constraint_coverage":float(covered.mean()),"trajectory_joint_constraint_coverage":float(group_coverage.mean()),"mean_normal_upper_margin_N":float((normal_upper-p[:,2]).mean()),"mean_g_upper_margin_N":float((upper-gforce(p)).mean()),"q_N":q,"trajectories":len(np.unique(groups)),"frames":len(ix)})
                    for j,mu in enumerate(MUS):
                        for ceiling in CEILINGS:
                            accept=(upper[:,j]<=0)&(normal_upper<=ceiling)
                            truly_valid=(true_g[:,j]<=0)&(y[:,2]<=ceiling)
                            false_certificate=accept&~truly_valid
                            group_false=np.array([false_certificate[groups==g].any() for g in np.unique(groups)])
                            selection_rows.append({"seed":seed,"alpha":alpha,"split":split,"method":method,"assumed_mu":float(mu),"assumed_normal_ceiling_N":ceiling,"certification_coverage":float(accept.mean()),"true_constraint_valid_fraction":float(truly_valid.mean()),"false_certificate_per_frame":float(false_certificate.mean()),"violation_given_certification":float(false_certificate.sum()/accept.sum()) if accept.any() else None,"valid_certificates_per_frame":float((accept&truly_valid).mean()),"trajectory_any_false_certificate":float(group_false.mean()),"certified_frames":int(accept.sum()),"frames":len(ix)})
        write_csv(args.out/"real_force_training_log.csv",logs)
        write_csv(args.out/"real_force_predictor_metrics.csv",metric_rows)
        write_csv(args.out/"real_force_envelope_metrics.csv",envelope_rows)
        write_csv(args.out/"real_force_selective_metrics.csv",selection_rows)
        write_csv(args.out/"real_force_calibration.csv",cal_rows)
        print(f"Completed seed {seed}: selected epoch {best_epoch}, val MAE {best_loss:.4f}N",flush=True)
    torch.cuda.synchronize()
    metadata={"scope":"offline force prediction / calibrated proxy-constraint analysis on public real tactile recordings; no executed robot control",
              "python":sys.version,"executable":sys.executable,"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"elapsed_seconds":time.perf_counter()-start,"gpu_peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,
              "seeds":args.seeds,"epochs":args.epochs,"batch_size":args.batch_size,"parameters":sum(p.numel() for p in model.parameters()),"split_seed":314159,"assumed_mus":MUS.tolist(),"assumed_normal_ceilings_N":CEILINGS,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"data_source_revision":"136db55485b6501605b1d2648ce0fe44740e302e"}
    (args.out/"real_force_run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=="__main__":main()
