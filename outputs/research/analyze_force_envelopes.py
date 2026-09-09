"""Postprocess completed runs: plots, grouped bootstrap, alignment sensitivity."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SEEDS=[17,29,43]
METHODS=["point","frame_constraint","trajectory_coordinate_box","trajectory_constraint"]
NAMES=["Point","Frame CP","Trajectory box","Trajectory constraint"]
COLORS=["#9B9FA8","#D1A357","#6F7FBB","#177F73"]
SPLITS=["test_id","test_batch6","test_flat","test_sharp"]
SPLIT_LABELS=["Sphere ID","Sphere batch 6","Flat shape OOD*","Sharp shape OOD*"]
SPLIT_CODES={"test_id":3,"test_batch6":4,"test_flat":5,"test_sharp":6}

def load_csv(name):return list(csv.DictReader((ROOT/name).open(encoding="utf-8")))
def values(rows,metric,**filters):
    return np.array([float(r[metric]) for r in rows if all(str(r[k])==str(v) for k,v in filters.items()) and r[metric]!=""])
def mean(rows,metric,**filters):
    a=values(rows,metric,**filters);return float(a.mean()) if len(a) else np.nan
def write_csv(name,rows):
    with (ROOT/name).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

pred_rows=load_csv("real_force_predictor_metrics.csv")
env_rows=load_csv("real_force_envelope_metrics.csv")
sel_rows=load_csv("real_force_selective_metrics.csv")
cal_rows=load_csv("real_force_calibration.csv")
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":200})
fig,ax=plt.subplots(figsize=(9,4.2));x=np.arange(4)
for i,(method,label,color) in enumerate([("cnn","Small CNN","#177F73"),("train_mean","Training mean","#9B9FA8")]):
    data=[values(pred_rows,"mae_xyz_mean_N",split=s,method=method) for s in SPLITS]
    ax.bar(x+(i-.5)*.34,[a.mean() for a in data],.34,yerr=[a.std(ddof=1) for a in data],label=label,color=color,capsize=3)
ax.set_xticks(x,SPLIT_LABELS);ax.set_ylabel("Mean absolute force error (N)")
ax.set_title("Real public tactile recordings: good sphere fit, poor shape transfer")
ax.legend(frameon=False);ax.set_ylim(0,.72)
fig.text(.5,.01,"3 training seeds; bars show seed mean ± SD. * Exploratory official index mapping; no test retuning.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(ROOT/"real_force_generalization.png");fig.savefig(ROOT/"real_force_generalization.pdf");plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(10.5,4.4))
x=np.arange(4)
for ax,key,title in zip(axs,["frame_joint_constraint_coverage","trajectory_joint_constraint_coverage"],["All constraints covered at one frame","All frames covered in one trajectory"]):
    a=[values(env_rows,key,split="test_id",method=m,alpha="0.1") for m in METHODS]
    ax.bar(x,[100*v.mean() for v in a],color=COLORS)
    ax.axhline(90,color="#B14541",ls="--",lw=1,label="Nominal 90%")
    for j,v in enumerate(a):ax.text(j,100*v.mean()+1.2,f"{100*v.mean():.1f}",ha="center")
    ax.set_xticks(x,NAMES,rotation=20,ha="right");ax.set_ylim(0,107);ax.set_ylabel("Empirical coverage (%)");ax.set_title(title)
fig.suptitle("Real-data force envelopes: frame coverage is not trajectory coverage",y=1.0,fontsize=12)
fig.text(.5,.01,"Sphere ID: 116 held-out trajectories, 9,003 frames. Nominal alpha = 0.10; same predictor for every method.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(ROOT/"real_force_coverage.png");fig.savefig(ROOT/"real_force_coverage.pdf");plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(10.5,4.4))
for method,name,color in zip(METHODS,NAMES,COLORS):
    coverage=[];violation=[]
    for mu in [.3,.5,.8]:
        f=dict(split="test_id",method=method,alpha="0.1",assumed_mu=str(mu),assumed_normal_ceiling_N="2.0")
        coverage.append(100*mean(sel_rows,"certification_coverage",**f))
        violation.append(100*mean(sel_rows,"violation_given_certification",**f))
    axs[0].plot([.3,.5,.8],coverage,marker="o",label=name,color=color)
    axs[1].plot([.3,.5,.8],violation,marker="o",label=name,color=color)
for ax in axs:
    ax.set_xlabel("Assumed friction coefficient (analysis parameter)");ax.set_xticks([.3,.5,.8]);ax.grid(alpha=.15)
axs[0].set_ylabel("Frames certified (%)");axs[1].set_ylabel("Proxy violation / certified frames (%)")
axs[0].set_title("Certification coverage");axs[1].set_title("Conditional proxy-constraint violation")
axs[0].legend(frameon=False,fontsize=9)
fig.suptitle("Offline proxy decisions: abstention is not a successful robot action",fontsize=12)
fig.text(.5,.01,"Sphere ID; alpha = .10; assumed Fn ceiling 2 N. Undefined rates for zero certifications are omitted.",ha="center",fontsize=9)
fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(ROOT/"real_force_selection.png");fig.savefig(ROOT/"real_force_selection.pdf");plt.close(fig)

# Cluster bootstrap keeps all frames and all seed predictions of each selected
# trajectory together. It resamples held-out trajectories, not training seeds.
predictions=[np.load(ROOT/f"real_force_predictions_seed{s}.npz") for s in SEEDS]
rng=np.random.default_rng(20260908);bootstrap=[];alignment=[]
mus=np.array([.3,.5,.8])
def g(y):return np.linalg.norm(y[:,:2],axis=1)[:,None]-y[:,2,None]*mus[None,:]
def bound(p,method,q):
    if method=="trajectory_coordinate_box":
        return np.linalg.norm(abs(p[:,:2])+q,axis=1)[:,None]-(p[:,2,None]-q)*mus,p[:,2]+q
    return g(p)+q,p[:,2]+q
def result(key,split,method,estimate,samples):
    return dict(metric=key,split=split,method=method,estimate=float(estimate),ci95_low=float(np.nanquantile(samples,.025)),ci95_high=float(np.nanquantile(samples,.975)),resamples=2000,unit="trajectory cluster shared across model seeds")
for split in SPLITS:
    ix=predictions[0]["split_code"]==SPLIT_CODES[split]
    y=predictions[0]["label_N"][ix];groups=predictions[0]["trajectory_id"][ix]
    ids=np.unique(groups);m=len(ids);group_ix=[np.flatnonzero(groups==i) for i in ids]
    counts=np.array([len(a) for a in group_ix]);draws=rng.multinomial(m,np.full(m,1/m),size=2000)
    err=np.stack([np.array([np.abs(p["prediction_N"][ix][a]-y[a]).mean(1).sum() for a in group_ix]) for p in predictions])
    e=err.mean(0);bs=(draws@e)/(draws@counts)
    bootstrap.append(result("mae_xyz_mean_N",split,"cnn",e.sum()/counts.sum(),bs))
    for method in METHODS:
        coverage=[];accepted=[];invalid=[]
        for seed,pred in zip(SEEDS,predictions):
            p=pred["prediction_N"][ix]
            q=float(next(r["q_N"] for r in cal_rows if r["seed"]==str(seed) and r["method"]==method and r["alpha"]=="0.1"))
            upper,normal=bound(p,method,q)
            covered=(g(y)<=upper+1e-8).all(1)&(y[:,2]<=normal+1e-8)
            coverage.append([covered[a].all() for a in group_ix])
            accept=(upper[:,2]<=0)&(normal<=2.0)
            bad=accept&((g(y)[:,2]>0)|(y[:,2]>2.0))
            accepted.append([accept[a].sum() for a in group_ix]);invalid.append([bad[a].sum() for a in group_ix])
        coverage=np.array(coverage).mean(0);bs=(draws@coverage)/m
        bootstrap.append(result("trajectory_joint_coverage_alpha10",split,method,coverage.mean(),bs))
        acc=np.array(accepted,dtype=float);bad=np.array(invalid,dtype=float)
        bs=(draws@acc.mean(0))/(draws@counts)
        bootstrap.append(result("certification_fraction_mu08_ceiling2",split,method,acc.mean(0).sum()/counts.sum(),bs))
        if acc.sum()>0:
            # Mean conditional rate over model seeds, not treating seeds as new
            # independent held-out trajectories. Undefined models are omitted.
            denom=draws@acc.T;num=draws@bad.T
            rates=np.divide(num,denom,out=np.full_like(num,np.nan),where=denom>0)
            bs=np.nanmean(rates,axis=1)
            estimate=np.nanmean(np.divide(bad.sum(1),acc.sum(1),out=np.full(len(SEEDS),np.nan),where=acc.sum(1)>0))
            bootstrap.append(result("proxy_violation_given_certificate_mu08_ceiling2",split,method,estimate,bs))
    # Prespecified ±5-frame pairing sensitivity on the same center positions.
    for seed,pred in zip(SEEDS,predictions):
        p=pred["prediction_N"][ix]
        for offset in [-5,0,5]:
            per=[]
            for a in group_ix:
                if len(a)>10:
                    center=np.arange(5,len(a)-5)
                    per.append(abs(p[a[center+offset]]-y[a[center]]))
            e=np.concatenate(per)
            alignment.append({"seed":seed,"split":split,"prediction_frame_offset":offset,"paired_center_frames":len(e),"mae_xyz_mean_N":float(e.mean()),"selected":False})
write_csv("real_force_bootstrap_intervals.csv",bootstrap)
write_csv("real_force_alignment_sensitivity.csv",alignment)

summary=[]
for r in env_rows:
    if r["alpha"]!="0.1":continue
    key=(r["split"],r["method"])
    if any((s["split"],s["method"])==key for s in summary):continue
    summary.append({"split":key[0],"method":key[1],**{metric:mean(env_rows,metric,split=key[0],method=key[1],alpha="0.1") for metric in ["frame_joint_constraint_coverage","trajectory_joint_constraint_coverage","mean_g_upper_margin_N"]}})
write_csv("real_force_summary_alpha10.csv",summary)
(ROOT/"real_force_analysis_audit.json").write_text(json.dumps({"split_trajectory_overlap":False,"training_seeds":SEEDS,"labels_are_measured_force":True,"friction_and_normal_ceiling_are_assumed":True,"slip_label_not_used_as_independent_ground_truth":True,"zero_certification_violation_rate":"undefined, never treated as zero","bootstrap":"2000 trajectory-cluster resamples, shared indices across 3 seed models","alignment_offsets_reported_without_selection":[-5,0,5]},indent=2),encoding="utf-8")
print("Real-force plots, bootstrap intervals, sensitivity, and summary created.")
