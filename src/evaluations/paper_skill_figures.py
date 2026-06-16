#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.interpolate import interp1d

from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset


METHOD_LABELS = {
    "susd": "SUSD",
    "metra": "METRA",
    "dads": "DADS",
    "lsd": "LSD",
    "diayn": "DIAYN",
    "dads_poe": "DADS-PoE",
}
DEFAULT_METHODS = ["susd", "metra", "dads", "lsd", "diayn", "dads_poe"]
DEFAULT_ENVS = ["ant", "half_cheetah", "kitchen"]

KITCHEN_TASKS = ["bottom burner", "top burner", "light switch", "slide cabinet", "hinge cabinet", "microwave", "kettle"]
KITCHEN_CUSTOM_ORDER = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    28, 29, 30, 49, 50, 51,
    31, 52,
    32, 33, 34, 35, 36, 37, 38, 53, 54, 55, 56, 57, 58,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper figures for skill traces, state coverage, and zero-shot goal distance."
    )
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS, choices=DEFAULT_ENVS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--checkpoint-root", default="final_models")
    parser.add_argument("--checkpoint-epoch", default="latest")
    parser.add_argument("--output-root", default="results/paper_skill_figures")
    parser.add_argument("--stages", nargs="+", default=["traces", "coverage", "downstream", "plots"])
    parser.add_argument("--num-skills", type=int, default=16)
    parser.add_argument("--trace-horizon", type=int, default=200)
    parser.add_argument("--coverage-steps", type=int, default=100000)
    parser.add_argument("--downstream-steps", type=int, default=20000)
    parser.add_argument("--skill-period", type=int, default=200)
    parser.add_argument("--downstream-horizon", type=int, default=200)
    parser.add_argument("--kitchen-goal", default="kettle", choices=KITCHEN_TASKS)
    parser.add_argument("--coverage-bin-size", type=float, default=0.25)
    parser.set_defaults(skip_missing=True, deterministic=True)
    parser.add_argument("--skip-missing", dest="skip_missing", action="store_true")
    parser.add_argument("--no-skip-missing", dest="skip_missing", action="store_false")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    return parser.parse_args()


def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def make_base_env(env_name, *, goal=False, seed=0, render_hw=100, kitchen_goal="kettle"):
    if env_name == "ant":
        if goal:
            from downstream_tasks.ant_multi_goals import AntMultiGoalsEnv

            env = AntMultiGoalsEnv(render_hw=render_hw)
        else:
            from envs.mujoco.ant_env import AntEnv

            env = AntEnv(render_hw=render_hw)
    elif env_name == "half_cheetah":
        if goal:
            from downstream_tasks.half_cheetah_multi_goals import HalfCheetahGoal

            env = HalfCheetahGoal(render_hw=render_hw)
        else:
            from envs.mujoco.half_cheetah_env import HalfCheetahEnv

            env = HalfCheetahEnv(render_hw=render_hw, fixed_initial_state=True)
    elif env_name == "kitchen":
        from gymnasium.wrappers import TimeLimit
        from gymnasium_robotics.envs.franka_kitchen import KitchenEnv

        tasks = KITCHEN_TASKS if not goal else [kitchen_goal]
        env = KitchenEnv(tasks_to_complete=tasks, terminate_on_tasks_completed=goal, render_mode="rgb_array")
        env = TimeLimit(env, max_episode_steps=200)
    else:
        raise ValueError(f"Unsupported env: {env_name}")

    if hasattr(env, "seed"):
        env.seed(seed)
    return env


def make_env(env_name, *, goal=False, seed=0, render_hw=100, kitchen_goal="kettle"):
    base_env = make_base_env(env_name, goal=goal, seed=seed, render_hw=render_hw, kitchen_goal=kitchen_goal)
    if env_name == "kitchen":
        return base_env
    mean, std = get_normalizer_preset(f"{env_name}_preset")
    return consistent_normalize(base_env, normalize_obs=True, mean=mean, std=std)


def kitchen_obs_vector(obs):
    if obs is None:
        raise ValueError("Kitchen observation is not initialized.")
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        obs = obs["observation"]
    return np.asarray(obs, dtype=np.float32)[KITCHEN_CUSTOM_ORDER]


def reset_env(env, seed=None):
    try:
        result = env.reset(seed=seed)
    except TypeError:
        result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    if not hasattr(env, "_normalize_obs"):
        env._last_obs = obs
        if isinstance(obs, dict):
            return kitchen_obs_vector(obs)
    return obs


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, reward, done, info = result
    if not hasattr(env, "_normalize_obs"):
        env._last_obs = obs
        if isinstance(obs, dict):
            obs = kitchen_obs_vector(obs)
    return obs, reward, done, info


def raw_obs(env):
    return np.asarray(getattr(env, "_cur_obs", None), dtype=np.float32)


def raw_xy(env, env_name):
    if env_name == "kitchen":
        obs = kitchen_obs_vector(getattr(env, "_last_obs", None))
        return np.asarray([obs[0], obs[1]], dtype=np.float64)
    base = unwrap_env(env)
    if env_name == "ant":
        return np.asarray(base.sim.data.qpos[:2], dtype=np.float64)
    return np.asarray([base.sim.data.qpos[0], 0.0], dtype=np.float64)


def kitchen_goal_slice(goal_name):
    return {
        "microwave": (31, 32),
        "bottom burner": (35, 37),
        "top burner": (37, 39),
        "light switch": (39, 41),
        "slide cabinet": (41, 42),
        "hinge cabinet": (42, 44),
        "kettle": (44, 51),
    }[goal_name]


def goal_distance(env, env_name, kitchen_goal="kettle"):
    if env_name == "kitchen":
        obs = kitchen_obs_vector(getattr(env, "_last_obs", None))
        base = unwrap_env(env)
        lo, hi = kitchen_goal_slice(kitchen_goal)
        goal = np.asarray(base.goal[kitchen_goal], dtype=np.float32).reshape(-1)
        return float(np.linalg.norm(obs[lo:hi] - goal))
    base = unwrap_env(env)
    goal = base.current_goal
    if env_name == "ant":
        return float(np.linalg.norm(np.asarray(base.sim.data.qpos[:2]) - np.asarray(goal)))
    return float(abs(base.sim.data.qpos[0] - float(goal)))


def checkpoint_candidates(root, env_name, method, seed, epoch, kind):
    method_dir = Path(root) / env_name / method.upper()
    if epoch == "latest":
        patterns = [
            method_dir / f"seed_{seed}" / f"{kind}*.pt",
            method_dir / f"{kind}*.pt",
        ]
    else:
        patterns = [
            method_dir / f"seed_{seed}" / f"{kind}{epoch}.pt",
            method_dir / f"{kind}{epoch}.pt",
        ]

    candidates = []
    for pattern in patterns:
        candidates.extend(pattern.parent.glob(pattern.name))
    return sorted(candidates, key=lambda p: (extract_epoch(p, kind), str(p)))


def extract_epoch(path, kind):
    stem = path.stem
    suffix = stem.replace(kind, "", 1)
    return int(suffix) if suffix.isdigit() else -1


def resolve_checkpoint(root, env_name, method, seed, epoch, kind):
    candidates = checkpoint_candidates(root, env_name, method, seed, epoch, kind)
    return candidates[-1] if candidates else None


def load_checkpoint(path, key, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    obj = checkpoint[key] if isinstance(checkpoint, dict) and key in checkpoint else checkpoint
    return obj.to(device).eval()


def policy_input_dim(policy):
    module = getattr(policy, "_module", None)
    for obj in [module, getattr(module, "module", None)]:
        if obj is None:
            continue
        for attr in ["_input_dim", "input_dim"]:
            value = getattr(obj, attr, None)
            if value is not None:
                return int(value)
    raise ValueError("Could not infer policy input dimension from checkpoint.")


def infer_skill_dim(policy, obs_dim):
    skill_dim = policy_input_dim(policy) - int(obs_dim)
    if skill_dim <= 0:
        raise ValueError(f"Invalid skill_dim={skill_dim}")
    return skill_dim


def unit_skill(skill_idx, num_skills, skill_dim):
    angle = 2.0 * np.pi * skill_idx / num_skills
    base = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
    reps = int(np.ceil(skill_dim / 2))
    z = np.tile(base, reps)[:skill_dim]
    return z / (np.linalg.norm(z) + 1e-12)


def random_skill(rng, skill_dim):
    z = rng.normal(size=skill_dim).astype(np.float32)
    return z / (np.linalg.norm(z) + 1e-12)


def policy_action(policy, obs, skill, device, deterministic):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    skill_t = torch.as_tensor(skill, dtype=torch.float32, device=device).unsqueeze(0)
    policy_input = torch.cat([obs_t, skill_t], dim=-1)
    with torch.no_grad():
        if deterministic and hasattr(policy, "get_mode_actions"):
            action, _ = policy.get_mode_actions(policy_input)
        else:
            action, _ = policy.get_actions(policy_input)
    return action[0]


def encoder_mean(encoder, obs_t):
    value = encoder(obs_t)
    return value if torch.is_tensor(value) else value.mean


def goal_skill(env, env_name, encoder, obs, device, kitchen_goal="kettle"):
    if env_name == "kitchen":
        current = np.asarray(obs, dtype=np.float32).copy()
        goal = current.copy()
        lo, hi = kitchen_goal_slice(kitchen_goal)
        goal_value = np.asarray(unwrap_env(env).goal[kitchen_goal], dtype=np.float32).reshape(-1)
        goal[lo:hi] = goal_value
        with torch.no_grad():
            phi_s = encoder_mean(encoder, torch.as_tensor(current, dtype=torch.float32, device=device).unsqueeze(0))
            phi_g = encoder_mean(encoder, torch.as_tensor(goal, dtype=torch.float32, device=device).unsqueeze(0))
        z = phi_g - phi_s
        z = z / (torch.norm(z, dim=-1, keepdim=True) + 1e-12)
        return z.squeeze(0).detach().cpu().numpy().astype(np.float32)

    current_raw = raw_obs(env).copy()
    goal_raw = current_raw.copy()
    base = unwrap_env(env)
    if env_name == "ant":
        goal_raw[0] = base.current_goal[0]
        goal_raw[1] = base.current_goal[1]
    else:
        goal_raw[0] = base.current_goal

    current_norm = env._apply_normalize_obs(current_raw)
    goal_norm = env._apply_normalize_obs(goal_raw)
    with torch.no_grad():
        phi_s = encoder_mean(encoder, torch.as_tensor(current_norm, dtype=torch.float32, device=device).unsqueeze(0))
        phi_g = encoder_mean(encoder, torch.as_tensor(goal_norm, dtype=torch.float32, device=device).unsqueeze(0))
    z = phi_g - phi_s
    z = z / (torch.norm(z, dim=-1, keepdim=True) + 1e-12)
    return z.squeeze(0).detach().cpu().numpy().astype(np.float32)


def load_pair(args, env_name, method, seed, need_encoder=False):
    option_path = resolve_checkpoint(args.checkpoint_root, env_name, method, seed, args.checkpoint_epoch, "option_policy")
    encoder_path = resolve_checkpoint(args.checkpoint_root, env_name, method, seed, args.checkpoint_epoch, "traj_encoder")
    if option_path is None or (need_encoder and encoder_path is None):
        missing = "option_policy" if option_path is None else "traj_encoder"
        msg = f"Missing {missing} checkpoint for env={env_name} method={method} seed={seed}"
        if args.skip_missing:
            print(f"[skip] {msg}")
            return None, None, None, None
        raise FileNotFoundError(msg)

    policy = load_checkpoint(option_path, "policy", args.device)
    encoder = load_checkpoint(encoder_path, "traj_encoder", args.device) if encoder_path is not None else None
    return policy, encoder, option_path, encoder_path


def evaluate_traces(args, env_name, method, seed, out_dir):
    policy, _, option_path, _ = load_pair(args, env_name, method, seed, need_encoder=False)
    if policy is None:
        return None
    env = make_env(env_name, goal=False, seed=seed)
    obs = reset_env(env, seed=seed)
    skill_dim = infer_skill_dim(policy, len(obs))
    rows = []

    for skill_idx in range(args.num_skills):
        obs = reset_env(env, seed=seed)
        start_xy = raw_xy(env, env_name).copy()
        z = unit_skill(skill_idx, args.num_skills, skill_dim)
        for t in range(args.trace_horizon + 1):
            xy = raw_xy(env, env_name) - start_xy
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "skill_idx": skill_idx,
                    "time": t,
                    "x": xy[0],
                    "y": xy[1],
                    "checkpoint": str(option_path),
                }
            )
            if t == args.trace_horizon:
                break
            action = policy_action(policy, obs, z, args.device, args.deterministic)
            obs, _, done, _ = step_env(env, action)
            if done:
                obs = reset_env(env, seed=seed)

    path = out_dir / f"traces_{method}_seed{seed}.csv"
    write_csv(path, rows)
    return path


def evaluate_coverage(args, env_name, method, seed, out_dir):
    policy, _, option_path, _ = load_pair(args, env_name, method, seed, need_encoder=False)
    if policy is None:
        return None
    env = make_env(env_name, goal=False, seed=seed)
    obs = reset_env(env, seed=seed)
    skill_dim = infer_skill_dim(policy, len(obs))
    rng = np.random.default_rng(seed)
    z = random_skill(rng, skill_dim)
    unique = set()
    rows = []

    for step in range(1, args.coverage_steps + 1):
        if step == 1 or (step - 1) % args.skill_period == 0:
            obs = reset_env(env, seed=seed)
            z = random_skill(rng, skill_dim)
        action = policy_action(policy, obs, z, args.device, args.deterministic)
        obs, _, done, info = step_env(env, action)
        xy = raw_xy(env, env_name)
        if env_name == "kitchen":
            unique.update(info.get("episode_task_completions", []))
        elif env_name == "ant":
            key = tuple(np.round(xy / args.coverage_bin_size).astype(int))
            unique.add(key)
        else:
            key = int(round(xy[0] / args.coverage_bin_size))
            unique.add(key)
        rows.append(
            {
                "method": method,
                "seed": seed,
                "time": step,
                "total_state_coverage": len(unique),
                "checkpoint": str(option_path),
            }
        )
        if done:
            obs = reset_env(env, seed=seed)

    path = out_dir / f"coverage_{method}_seed{seed}.csv"
    write_csv(path, rows)
    return path


def evaluate_downstream(args, env_name, method, seed, out_dir):
    policy, encoder, option_path, encoder_path = load_pair(args, env_name, method, seed, need_encoder=True)
    if policy is None:
        return None
    env = make_env(env_name, goal=True, seed=seed, render_hw=100, kitchen_goal=args.kitchen_goal)
    obs = reset_env(env, seed=seed)
    rows = []
    episode_t = 0

    for step in range(1, args.downstream_steps + 1):
        if episode_t >= args.downstream_horizon:
            obs = reset_env(env, seed=seed)
            episode_t = 0
        z = goal_skill(env, env_name, encoder, obs, args.device, args.kitchen_goal)
        action = policy_action(policy, obs, z, args.device, args.deterministic)
        obs, _, done, _ = step_env(env, action)
        episode_t += 1
        rows.append(
            {
                "method": method,
                "seed": seed,
                "time": step,
                "negative_goal_distance": -goal_distance(env, env_name, args.kitchen_goal),
                "checkpoint": str(option_path),
                "traj_encoder": str(encoder_path),
            }
        )
        if done:
            obs = reset_env(env, seed=seed)
            episode_t = 0

    path = out_dir / f"downstream_{method}_seed{seed}.csv"
    write_csv(path, rows)
    return path


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {path}")


def load_stage_csvs(out_dir, prefix):
    frames = []
    for path in sorted(out_dir.glob(f"{prefix}_*_seed*.csv")):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_traces(env_name, out_dir, methods):
    df = load_stage_csvs(out_dir, "traces")
    if df.empty:
        print(f"[skip] no trace CSVs in {out_dir}")
        return
    method_order = [m for m in methods if m in set(df["method"])]
    fig, axes = plt.subplots(1, len(method_order), figsize=(3.4 * len(method_order), 3.2), squeeze=False)
    for ax, method in zip(axes[0], method_order):
        sub = df[(df["method"] == method) & (df["seed"] == df[df["method"] == method]["seed"].min())]
        colors = plt.cm.hsv(np.linspace(0, 1, sub["skill_idx"].nunique() + 1))
        for color, (_, skill_df) in zip(colors, sub.groupby("skill_idx")):
            ax.plot(skill_df["x"], skill_df["y"], color=color, linewidth=1.2, alpha=0.9)
            ax.scatter(skill_df["x"].iloc[-1], skill_df["y"].iloc[-1], color=color, s=8)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.25)
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.25)
        ax.set_title(METHOD_LABELS.get(method, method.upper()))
        ax.set_xlabel("x")
        ax.set_aspect("equal" if env_name == "ant" else "auto", adjustable="box")
        ax.grid(True, alpha=0.2)
    axes[0][0].set_ylabel("y" if env_name == "ant" else "skill trace offset")
    fig.tight_layout()
    path = out_dir / f"{env_name}_xy_skill_traces.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[saved] {path}")


def plot_timeseries(out_dir, prefix, value_col, ylabel, title, methods, max_duration=None):
    df = load_stage_csvs(out_dir, prefix)
    if df.empty:
        print(f"[skip] no {prefix} CSVs in {out_dir}")
        return
    if max_duration is None:
        max_duration = int(df["time"].max())
    common_times = np.arange(1, max_duration + 1)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for method in methods:
        method_df = df[df["method"] == method]
        if method_df.empty:
            continue
        seed_values = []
        for _, seed_df in method_df.groupby("seed"):
            seed_df = seed_df.sort_values("time")
            f = interp1d(
                seed_df["time"],
                seed_df[value_col],
                kind="previous",
                bounds_error=False,
                fill_value=(seed_df[value_col].iloc[0], seed_df[value_col].iloc[-1]),
            )
            seed_values.append(f(common_times))
        seed_values = np.asarray(seed_values)
        mean = seed_values.mean(axis=0)
        if len(seed_values) > 1:
            sem = stats.sem(seed_values, axis=0)
            margin = sem * stats.t.ppf(0.975, len(seed_values) - 1)
        else:
            margin = np.zeros_like(mean)
        ax.plot(common_times, mean, label=METHOD_LABELS.get(method, method.upper()), linewidth=1.8)
        ax.fill_between(common_times, mean - margin, mean + margin, alpha=0.15)

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / f"{prefix}_{value_col}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[saved] {path}")


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    np.random.seed(0)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    for env_name in args.envs:
        out_dir = Path(args.output_root) / env_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[env] {env_name} -> {out_dir}")

        for method in args.methods:
            method = method.lower()
            for seed in args.seeds:
                if "traces" in args.stages:
                    evaluate_traces(args, env_name, method, seed, out_dir)
                if "coverage" in args.stages:
                    evaluate_coverage(args, env_name, method, seed, out_dir)
                if "downstream" in args.stages:
                    evaluate_downstream(args, env_name, method, seed, out_dir)

        if "plots" in args.stages:
            plot_traces(env_name, out_dir, [m.lower() for m in args.methods])
            plot_timeseries(
                out_dir,
                "coverage",
                "total_state_coverage",
                "Total State Coverage",
                f"{env_name}: State Coverage",
                [m.lower() for m in args.methods],
                max_duration=args.coverage_steps,
            )
            plot_timeseries(
                out_dir,
                "downstream",
                "negative_goal_distance",
                "Negative Goal Distance",
                f"{env_name}: Zero-Shot Goal Reaching",
                [m.lower() for m in args.methods],
                max_duration=args.downstream_steps,
            )


if __name__ == "__main__":
    main()
