"""Convert the released Meta GelSight trajectory schema, not the old raw loader.
Run using Python with NumPy and Pillow. Force labels in this release are already N.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import pickle
from pathlib import Path
import time
import numpy as np
from PIL import Image

REVISION="136db55485b6501605b1d2648ce0fe44740e302e"

def decode_image(buf):
    im=Image.open(io.BytesIO(buf)).convert("RGB") if isinstance(buf,bytes) else Image.fromarray(buf).convert("RGB")
    if im.height < im.width:
        im=im.transpose(Image.Transpose.ROTATE_270)
    # Official loader crops non-4:3 portrait images; preserve its mapping.
    h,w=im.height,im.width
    if h/w != 4/3:
        h2=int(h/(4/3))
        im=im.crop((0,int((h-h2)/2),w,int((h+h2)/2)))
    return np.asarray(im.resize((48,64),Image.Resampling.BILINEAR),dtype=np.uint8).transpose(2,0,1)


def aligned_pairs(trajectory):
    """Same sample mapping as official vision_based_forces_slip_probes.py.
    forces[j] pairs with indexes[j]; extra trailing indexes have no force label.
    This does not apply /1000 or reverse the already-positive normal force.
    """
    n=min(len(trajectory["indexes"]),len(trajectory["forces"]))
    return np.asarray(trajectory["indexes"][:n],dtype=np.int64),np.asarray(trajectory["forces"][:n],dtype=np.float32)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--audit",type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); labels=[];audit=[];total=0;trajectory_names=[]
    for path in sorted(args.data.glob("*/*/dataset_slip_forces.pkl")):
        d=pickle.load(path.open("rb"));assert set(d)=={"in_contact","trajectories"}
        trajectories=d["trajectories"]
        n=sum(len(aligned_pairs(t)[0]) for t in trajectories.values())
        labels.append((path,trajectories,n));total+=n
    images=np.lib.format.open_memmap(args.out/"images.npy",mode="w+",dtype=np.uint8,shape=(total,3,64,48))
    forces=np.lib.format.open_memmap(args.out/"forces.npy",mode="w+",dtype=np.float32,shape=(total,3))
    traj_ids=np.lib.format.open_memmap(args.out/"trajectory_ids.npy",mode="w+",dtype=np.int32,shape=(total,))
    source_idxs=np.lib.format.open_memmap(args.out/"source_frame_indexes.npy",mode="w+",dtype=np.int32,shape=(total,))
    ptr=0
    for path,trajectories,n in labels:
        batch=path.parent.relative_to(args.data).as_posix();frames=[]
        for image_path in sorted(path.parent.glob("dataset_gelsight_*.pkl")):
            frames.extend(pickle.load(image_path.open("rb")))
        indices=[];batch_forces=[];batch_ids=[];length_differences=[]
        for key,t in trajectories.items():
            frame_ids, f=aligned_pairs(t)
            assert len(frame_ids)>0 and frame_ids.max()<len(frames)
            trajectory_id=len(trajectory_names)
            trajectory_names.append(f"{batch}/trajectory_{key:04}")
            indices.extend(frame_ids.tolist());batch_forces.append(f)
            batch_ids.extend([trajectory_id]*len(frame_ids))
            length_differences.append(len(t["indexes"])-len(t["forces"]))
        assert len(indices)==len(set(indices)), f"Duplicate frame IDs in {batch}"
        with ThreadPoolExecutor(max_workers=8) as pool:
            converted=np.stack(list(pool.map(decode_image,(frames[i] for i in indices))))
        f=np.concatenate(batch_forces)
        assert np.isfinite(f).all()
        images[ptr:ptr+n]=converted;forces[ptr:ptr+n]=f
        traj_ids[ptr:ptr+n]=batch_ids;source_idxs[ptr:ptr+n]=indices
        audit.append({"batch":batch,"source_frames":len(frames),"retained_frames":n,
                      "trajectories":len(trajectories),"index_minus_label_lengths":sorted(set(length_differences)),
                      "force_min_N":f.min(0).tolist(),"force_max_N":f.max(0).tolist(),
                      "negative_normal_labels":int((f[:,2]<0).sum()),"alignment":"force[j] <-> indexes[j]",
                      "label_file_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        ptr+=n
        print(f"Converted {batch}: {n} frames, elapsed {time.perf_counter()-start:.1f}s",flush=True)
    for arr in [images,forces,traj_ids,source_idxs]:arr.flush()
    (args.out/"trajectory_names.json").write_text(json.dumps(trajectory_names,indent=2),encoding="utf-8")
    result={"dataset":"facebook/gelsight-force-estimation","revision":REVISION,
            "source_url":"https://huggingface.co/datasets/facebook/gelsight-force-estimation",
            "label_schema":"released trajectories: forces already Newtons and positive normal; no rescaling/sign flip",
            "axes":["Fx","Fy","Fn"],"total_frames":total,"total_trajectories":len(trajectory_names),
            "image_shape_CHW":[3,64,48],"flat_sharp_caveat":"five trailing indexes lack labels; follow official sample-index pairing; shape OOD exploratory",
            "batches":audit,"elapsed_seconds":time.perf_counter()-start}
    args.audit.parent.mkdir(parents=True,exist_ok=True)
    args.audit.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["total_frames","total_trajectories","elapsed_seconds"]}),flush=True)


if __name__=="__main__":main()
