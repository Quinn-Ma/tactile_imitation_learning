import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bimanual_imitation.algorithms.configs import ALG, get_param_config
from bimanual_imitation.algorithms.core.shared import util
from bimanual_imitation.constants import BIMANUAL_IMITATION_BASE_DIR, RESULTS_DIR
from bimanual_imitation.utils import get_enum_value
from irl_data.constants import EXPERT_TRAJS_DIR


def get_export_dir(date_id, sub_dir: str) -> Path:
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True)

    if date_id is None:
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}$")
        subdirs = [
            d for d in RESULTS_DIR.iterdir() if d.is_dir() and bool(date_pattern.match(d.name))
        ]
        if len(subdirs) > 0:
            sorted_subdirs = sorted(subdirs, key=lambda x: x.name, reverse=True)
            export_dir = sorted_subdirs[0] / sub_dir
            print("Using most recent date_id:", export_dir)
            if not export_dir.exists():
                export_dir.mkdir(parents=True)
        else:
            date_id = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            export_dir = RESULTS_DIR / date_id / sub_dir
            export_dir.mkdir(parents=True)
    else:
        export_dir = RESULTS_DIR / str(date_id) / sub_dir
        if not export_dir.exists():
            export_dir.mkdir(parents=True)

    return export_dir


def _run_commands(cmd_list: list, outputfiles: list):
    for cmd, outfile in zip(cmd_list, outputfiles):
        outfile.parent.mkdir(parents=True, exist_ok=True)
        print(f"Running: {cmd}")
        with open(outfile, "w") as f:
            result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"Command failed (exit {result.returncode}): {cmd}", file=sys.stderr)


def export_h5(
    args,
    h5_filename,
    prev_sub_dir: str,
    act_obs_dim=(18, 36),
    act_obs_pred_horizons=(4, 4, 8),
    unique_num_trajs=(50, 100, 200),
    export_metadata=True,
    export_rollouts=True,
):
    from irl_data import proto_logger
    from irl_data.trajectory import TrajBatch
    from irl_environments.constants import IRL_ENVIRONMENTS_BASE_DIR
    from irl_environments.core.utils import (
        ActionGroup,
        ObservationGroup,
        get_action_group_dim,
        get_observation_group_dim,
    )

    def _get_rename_map(env_name):
        env_yaml = IRL_ENVIRONMENTS_BASE_DIR / f"param/{env_name}.yaml"
        assert env_yaml.exists()
        with open(env_yaml, "r") as f:
            env_config = yaml.safe_load(f)

        obs_idx = 0
        act_idx = 0
        rename_map = {}
        for device_name, device_config in env_config["devices"].items():
            dev_space = env_config["spaces"][device_config["space"]]
            for obs_str in dev_space["observation_space"]:
                obs_enum = get_enum_value(obs_str, ObservationGroup)
                obs_dim = get_observation_group_dim(obs_enum)
                for d in range(obs_dim):
                    label = f"{device_name} {obs_str}" + (f" {d+1}" if obs_dim > 1 else "")
                    rename_map[f"obs_{obs_idx}"] = label
                    obs_idx += 1
            for act_str in dev_space["action_space"]:
                act_enum = get_enum_value(act_str, ActionGroup)
                act_dim = get_action_group_dim(act_enum)
                for d in range(act_dim):
                    label = f"{device_name} {act_str}" + (f" {d+1}" if act_dim > 1 else "")
                    rename_map[f"act_{act_idx}"] = label
                    act_idx += 1
        return rename_map

    prev_export_dir = get_export_dir(args.date_id, prev_sub_dir)
    dir_pattern = r"^alg=(\w+),env=(\w+),num_trajs=(\d{3}),run=(\d{3}),tag=(\d{2})$"
    dfs, mdfs, envs = [], [], []

    for dir_name in prev_export_dir.iterdir():
        if not dir_name.is_dir():
            continue
        dir_match = re.match(dir_pattern, dir_name.name)
        rollout_files = list(dir_name.glob("rollouts_snapshot_*.h5"))
        policy_log = dir_name / "policy_log.h5"
        if not (dir_match and rollout_files and policy_log.exists()):
            continue

        tag_group = int(dir_match.group(5))
        if tag_group != args.tag:
            continue

        alg = get_enum_value(str(dir_match.group(1)), ALG)
        env_group = str(dir_match.group(2))
        if env_group not in envs:
            envs.append(env_group)
        num_trajs_group = int(dir_match.group(3))
        run_group = int(dir_match.group(4))

        if export_rollouts:
            df = pd.read_hdf(rollout_files[0])
            if alg in (ALG.ACT, ALG.DIFFUSION):
                act_dim, obs_dim = act_obs_dim
                act_horizon, obs_horizon, pred_horizon = act_obs_pred_horizons
                start_act = (obs_horizon - 1) * act_dim
                act_keys = [f"act_{i}" for i in range(start_act, start_act + act_dim)]
                act_df = df[act_keys].rename(columns={k: f"act_{i}" for i, k in enumerate(act_keys)})
                df = pd.concat([df.drop(columns=list(df.filter(regex="act_*"))), act_df], axis=1)
                start_obs = (obs_horizon - 1) * obs_dim
                obs_keys = [f"obs_{i}" for i in range(start_obs, start_obs + obs_dim)]
                obs_df = df[obs_keys].rename(columns={k: f"obs_{i}" for i, k in enumerate(obs_keys)})
                df = pd.concat([df.drop(columns=list(df.filter(regex="obs_*"))), obs_df], axis=1)
            rename_map = _get_rename_map(env_group)
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        if export_metadata:
            with pd.HDFStore(policy_log, "r") as f:
                log_df = f["log"].set_index("iter")
            mdf = log_df[log_df["rollout_avgr"].notna() & log_df["rollout_avglen"].notna()].copy()
            mdf["iter_id"] = np.arange(mdf.shape[0])
            mdf.reset_index(inplace=True)

        add_keys = {"alg": alg.value, "num_trajs": num_trajs_group, "env": env_group, "seed": run_group}
        for key, val in add_keys.items():
            if export_rollouts:
                df[key] = val
            if export_metadata:
                mdf[key] = val
        if export_rollouts:
            dfs.append(df)
        if export_metadata:
            mdfs.append(mdf)

    if export_rollouts:
        util.warn("Exporting rollouts (do not exit)...")
        for env in envs:
            expert_proto = EXPERT_TRAJS_DIR / f"{env}.proto"
            ex_trajs = proto_logger.load_trajs(expert_proto)
            for nt in unique_num_trajs:
                add_keys = {"alg": ALG.EXPERT.value, "num_trajs": nt, "env": env, "seed": 0}
                ex_df = TrajBatch.FromTrajs(ex_trajs[:nt]).to_dataframe(**add_keys)
                rename_map = _get_rename_map(env)
                ex_df.rename(columns={k: v for k, v in rename_map.items() if k in ex_df.columns}, inplace=True)
                dfs.append(ex_df)

    def _save(in_dfs, path, pivot_cols=None):
        final_df = pd.concat(in_dfs)
        if pivot_cols is not None:
            remain = set(final_df.columns) - set(pivot_cols)
            final_df = final_df.pivot_table(index=pivot_cols, values=remain)
        final_df.to_hdf(path, key="irl_data", mode="w")
        print(f"Exported {path}")

    if export_rollouts:
        _save(dfs, h5_filename, pivot_cols=["alg", "env", "num_trajs", "seed", "rollout", "time"])
    if export_metadata:
        _save(mdfs, h5_filename.with_name("metadata_" + h5_filename.name),
              pivot_cols=["alg", "env", "num_trajs", "seed", "iter_id"])


def phase1_hp_search(args):
    util.header("=== Running Phase 1: HP Search ===")
    alg = get_enum_value(args.alg, ALG)
    export_dir = get_export_dir(args.date_id, "phase1_hp_search")
    tag = str(args.tag).zfill(2)

    target_script = BIMANUAL_IMITATION_BASE_DIR / f"algorithms/imitate_{alg.value}.py"
    cmds, outs = [], []
    for worker in range(args.hp_search_num_workers):
        cmd = (f"python3 -u {target_script} --mode hp_search --env_name {args.env_name} "
               f"--export_dir {export_dir} --tag {args.tag} "
               f"--hp_search_num_trials {args.hp_search_num_trials}")
        cmds.append(cmd)
        outs.append(export_dir / "output_logs" / f"alg={alg.value},tag={tag},worker={str(worker).zfill(3)}.log")

    _run_commands(cmds, outs)


def phase2_hp_search_analysis(args):
    import optuna
    from scipy.stats import sem

    util.header("=== Running Phase 2: HP Search Analysis ===")
    study_id = f"{args.alg}_{str(args.tag).zfill(2)}"
    prev_dir = get_export_dir(args.date_id, "phase1_hp_search")
    assert prev_dir.exists(), "Phase 1 study must already exist!"

    database_dir = prev_dir / "databases"
    study = optuna.load_study(study_name=None, storage=f"sqlite:///{database_dir}/{study_id}.db")

    cmap = {}
    for trial in study.trials:
        if trial.values:
            key = tuple(sorted(trial.params.items()))
            val = -1 * trial.values[0]
            cmap.setdefault(key, []).append(val)

    means = np.array([np.mean(v) for v in cmap.values()])
    std_errs = np.array([sem(v) for v in cmap.values()])
    bounds = [m if np.isnan(se) else m - se for m, se in zip(means, std_errs)]
    lens = [len(v) for v in cmap.values()]
    worsts = [min(v) for v in cmap.values()]

    best_idxs = np.argsort(bounds)[::-1][:30]
    print("Sorted Params:")
    for i, idx in enumerate(best_idxs):
        params = dict(list(cmap.keys())[idx])
        print(f"[{i}] trials={lens[idx]} avg={means[idx]:.4f} worst={worsts[idx]:.4f} "
              f"lower={bounds[idx]:.4f} params={params}")


def phase3_train(args):
    util.header("=== Running Phase 3: Train ===")
    alg = get_enum_value(args.alg, ALG)
    tag = str(args.tag).zfill(2)
    export_dir = get_export_dir(args.date_id, "phase3_train")

    if args.spec is None:
        spec_file = BIMANUAL_IMITATION_BASE_DIR / f"param/{alg.value}.yaml"
    else:
        spec_file = Path(args.spec)
    assert spec_file.exists(), f"Spec file not found: {spec_file}"

    with open(spec_file, "r") as f:
        spec = yaml.safe_load(f)

    target_script = BIMANUAL_IMITATION_BASE_DIR / f"algorithms/imitate_{alg.value}.py"
    cmds, outs = [], []
    for task in spec["tasks"]:
        for num_trajs in spec["training"]["dataset_num_trajs"]:
            for run in range(spec["training"]["runs"]):
                strid = (f"alg={alg.value},env={task['env']},"
                         f"num_trajs={str(num_trajs).zfill(3)},"
                         f"run={str(run).zfill(3)},tag={tag}")
                run_dir = export_dir / strid
                cfg_class = get_param_config(alg)
                cfg = cfg_class(**spec["params"])

                cmd = (f"python3 -u {target_script} --mode train_policy "
                       f"--env_name {task['env']} --export_dir {run_dir} "
                       f"--data_subsamp_freq {task['data_subsamp_freq']} "
                       f"--limit_trajs {num_trajs} ")
                for name, value in vars(cfg).items():
                    cmd += f"--{name} {value} "
                for name, value in spec["options"].items():
                    cmd += f"--{name} {value} "

                cmds.append(cmd)
                outs.append(run_dir / "output.log")

    _run_commands(cmds, outs)


def phase4_train_analysis(args):
    util.header("=== Running Phase 4: Train Analysis ===")
    tag = str(args.tag).zfill(2)
    export_dir = get_export_dir(args.date_id, "phase4_train_analysis")
    h5_filename = export_dir / f"train_analysis_{tag}.h5"
    export_h5(
        args,
        h5_filename,
        prev_sub_dir="phase3_train",
        act_obs_dim=(18, 36),
        act_obs_pred_horizons=(4, 4, 8),
        unique_num_trajs=(50, 100, 200),
        export_metadata=True,
        export_rollouts=True,
    )


if __name__ == "__main__":
    phases = {
        "1_hp_search": phase1_hp_search,
        "2_hp_search_analysis": phase2_hp_search_analysis,
        "3_train": phase3_train,
        "4_train_analysis": phase4_train_analysis,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str)
    parser.add_argument("--spec", type=str)
    parser.add_argument("--phase", choices=sorted(phases.keys()), required=True)
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--hp_search_num_workers", type=int, default=4)
    parser.add_argument("--hp_search_num_trials", type=int, default=10)
    parser.add_argument("--date_id", type=str)
    parser.add_argument("--tag", type=int, default=0)
    args = parser.parse_args()

    phases[args.phase](args)
