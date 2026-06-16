#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from envs.mujoco.ant_env import AntEnv
from garagei.envs.consistent_normalized_env import consistent_normalize
from iod.utils import get_normalizer_preset


DEFAULT_HEADINGS = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
DEFAULT_METHODS = ["susd", "metra", "csd", "lsd", "diayn"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ant heading counterfactual diagnostic for skill-conditioned policies."
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--checkpoint-root", default="final_models/ant")
    parser.add_argument("--checkpoint-epoch", default="latest")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Override checkpoint as METHOD=path/to/option_policy.pt. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", default="results/ant_heading_counterfactual")
    parser.add_argument("--num-skills", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--num-eval-rollouts", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--normalize-obs", choices=["preset", "off"], default="preset")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--render-hw", type=int, default=100)
    parser.add_argument("--skip-missing", action="store_true", default=True)
    return parser.parse_args()


def load_policy(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    policy = checkpoint["policy"] if isinstance(checkpoint, dict) and "policy" in checkpoint else checkpoint
    return policy.to(device).eval()


def extract_epoch(path, kind):
    suffix = path.stem.replace(kind, "", 1)
    return int(suffix) if suffix.isdigit() else -1


def checkpoint_candidates(root, method, seed, epoch, kind="option_policy"):
    method_dir = Path(root) / method.upper()
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


def resolve_checkpoint(root, method, seed, epoch):
    candidates = checkpoint_candidates(root, method, seed, epoch)
    return candidates[-1] if candidates else None


def checkpoint_overrides(items):
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--checkpoint must be METHOD=PATH, got: {item}")
        method, path = item.split("=", 1)
        overrides[method.lower()] = Path(path)
    return overrides


def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def make_env(seed, normalize_obs, render_hw):
    base_env = AntEnv(render_hw=render_hw)
    base_env.seed(seed)
    if normalize_obs == "preset":
        mean, std = get_normalizer_preset("ant_preset")
        return consistent_normalize(base_env, normalize_obs=True, mean=mean, std=std)
    return consistent_normalize(base_env, normalize_obs=False)


def yaw_quat(theta):
    return np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)])


def normalize_observation_if_needed(env, obs):
    if hasattr(env, "_normalize_obs") and env._normalize_obs:
        obs = env._apply_normalize_obs(obs)
    return obs


def reset_ant_with_heading(env, theta):
    env.reset()
    base_env = unwrap_env(env)
    qpos = base_env.sim.data.qpos.copy()
    qvel = base_env.sim.data.qvel.copy()
    qpos[3:7] = yaw_quat(theta)
    qvel[3:6] = 0.0
    base_env.set_state(qpos, qvel)
    obs = base_env._get_obs()
    if hasattr(env, "_cur_obs"):
        env._cur_obs = obs
    return normalize_observation_if_needed(env, obs)


def root_xy(env):
    base_env = unwrap_env(env)
    return np.array(base_env.sim.data.qpos[:2], dtype=np.float64)


def infer_skill_dim(policy, obs_dim):
    module = getattr(policy, "_module", None)
    input_dim = getattr(module, "_input_dim", None)
    if input_dim is None:
        input_dim = getattr(module, "input_dim", None)
    if input_dim is None and hasattr(module, "module"):
        input_dim = getattr(module.module, "_input_dim", None)
        if input_dim is None:
            input_dim = getattr(module.module, "input_dim", None)
    if input_dim is None:
        raise ValueError("Could not infer policy input dimension. Pass a compatible checkpoint.")
    skill_dim = int(input_dim) - int(obs_dim)
    if skill_dim <= 0:
        raise ValueError(f"Invalid inferred skill_dim={skill_dim} from input_dim={input_dim}, obs_dim={obs_dim}")
    return skill_dim


def base_skills(num_skills):
    angles = 2.0 * np.pi * np.arange(num_skills) / num_skills
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def expand_skill(skill_2d, skill_dim):
    if skill_dim == 2:
        return skill_2d.astype(np.float32)
    repeats = int(np.ceil(skill_dim / 2))
    return np.tile(skill_2d, repeats)[:skill_dim].astype(np.float32)


def policy_action(policy, obs, skill, device, stochastic):
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    skill_tensor = torch.as_tensor(skill, dtype=torch.float32, device=device).unsqueeze(0)
    policy_input = torch.cat([obs_tensor, skill_tensor], dim=-1)
    with torch.no_grad():
        if stochastic:
            action, _ = policy.get_actions(policy_input)
        elif hasattr(policy, "get_mode_actions"):
            action, _ = policy.get_mode_actions(policy_input)
        else:
            action, _ = policy.get_actions(policy_input)
    return action[0]


def rotate_minus_theta(points, theta):
    rot = np.array(
        [
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta), np.cos(-theta)],
        ]
    )
    return points @ rot.T


def rollout(env, policy, skill, theta, horizon, device, stochastic):
    obs = reset_ant_with_heading(env, theta)
    p0 = root_xy(env)
    world = [np.zeros(2, dtype=np.float64)]
    done = False
    for _ in range(horizon):
        action = policy_action(policy, obs, skill, device, stochastic)
        obs, _, done, _ = env.step(action)
        world.append(root_xy(env) - p0)
        if done:
            break
    while len(world) < horizon + 1:
        world.append(world[-1].copy())
    tau_world = np.asarray(world)
    tau_body = rotate_minus_theta(tau_world, theta)
    return tau_world, tau_body


def trace_cov(points):
    if points.shape[0] <= 1:
        return 0.0
    return float(np.trace(np.cov(points.T, bias=True)))


def compute_metrics(records, num_skills, headings):
    rows = []
    for skill_idx in range(num_skills):
        skill_records = [r for r in records if r["skill_idx"] == skill_idx]
        world_by_heading = []
        body_by_heading = []
        for theta in headings:
            theta_records = [r for r in skill_records if np.isclose(r["theta"], theta)]
            world_by_heading.append(np.mean([r["tau_world"] for r in theta_records], axis=0))
            body_by_heading.append(np.mean([r["tau_body"] for r in theta_records], axis=0))
        world_by_heading = np.asarray(world_by_heading)
        body_by_heading = np.asarray(body_by_heading)
        world_var = np.mean([trace_cov(world_by_heading[:, t, :]) for t in range(world_by_heading.shape[1])])
        body_var = np.mean([trace_cov(body_by_heading[:, t, :]) for t in range(body_by_heading.shape[1])])
        rows.append(
            {
                "skill_idx": skill_idx,
                "body_var": body_var,
                "world_var": world_var,
                "locality_score": world_var / (body_var + 1e-8),
            }
        )
    return rows


def save_rollouts_npz(path, records):
    skill_dims = np.asarray([len(r["skill"]) for r in records])
    max_skill_dim = int(skill_dims.max()) if len(skill_dims) else 0
    skills = np.full((len(records), max_skill_dim), np.nan, dtype=np.float32)
    for idx, record in enumerate(records):
        skill = np.asarray(record["skill"], dtype=np.float32)
        skills[idx, : len(skill)] = skill

    np.savez_compressed(
        path,
        method=np.asarray([r["method"] for r in records]),
        seed=np.asarray([r["seed"] for r in records]),
        rollout_idx=np.asarray([r["rollout_idx"] for r in records]),
        skill_idx=np.asarray([r["skill_idx"] for r in records]),
        theta=np.asarray([r["theta"] for r in records]),
        skill=skills,
        skill_dim=skill_dims,
        tau_world=np.asarray([r["tau_world"] for r in records]),
        tau_body=np.asarray([r["tau_body"] for r in records]),
    )


def save_metrics_csv(path, rows):
    fieldnames = ["method", "seed", "skill_idx", "body_var", "world_var", "locality_score"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(path, rows):
    fieldnames = ["method", "seed", "body_var", "world_var", "locality_score"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_method(records, method, out_path, num_skills, headings):
    selected = [idx for idx in [0, num_skills // 4, num_skills // 2, 3 * num_skills // 4] if idx < num_skills]
    fig, axes = plt.subplots(len(selected), 2, figsize=(9, 3.0 * len(selected)), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(headings)))
    for row_idx, skill_idx in enumerate(selected):
        for col_idx, frame_key in enumerate(["tau_world", "tau_body"]):
            ax = axes[row_idx, col_idx]
            for color, theta in zip(colors, headings):
                theta_records = [
                    r for r in records if r["skill_idx"] == skill_idx and np.isclose(r["theta"], theta)
                ]
                if not theta_records:
                    continue
                traj = np.mean([r[frame_key] for r in theta_records], axis=0)
                ax.plot(traj[:, 0], traj[:, 1], color=color, label=f"{theta / np.pi:.1f}pi")
                ax.scatter(traj[0, 0], traj[0, 1], color=color, s=12)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.set_title(f"skill {skill_idx} / {'world' if col_idx == 0 else 'body'}")
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="best", fontsize=8)
    fig.suptitle(method.upper())
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def summarize(metric_rows):
    by_method_seed = {}
    for row in metric_rows:
        key = (row["method"], row["seed"])
        by_method_seed.setdefault(key, []).append(row)

    summary = []
    for (method, seed), rows in sorted(by_method_seed.items()):
        summary.append(
            {
                "method": method,
                "seed": seed,
                "body_var": float(np.mean([r["body_var"] for r in rows])),
                "world_var": float(np.mean([r["world_var"] for r in rows])),
                "locality_score": float(np.mean([r["locality_score"] for r in rows])),
            }
        )
    return summary


def print_summary(rows):
    if not rows:
        print("No methods were evaluated. Provide checkpoints with --checkpoint METHOD=PATH or --checkpoint-root.")
        return
    print("\nAnt heading counterfactual summary")
    print("method seed body_var world_var locality_score")
    for row in rows:
        print(
            f"{row['method']:>6} {row['seed']:>4} "
            f"{row['body_var']:.6f} {row['world_var']:.6f} {row['locality_score']:.6f}"
        )


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = checkpoint_overrides(args.checkpoint)
    headings = DEFAULT_HEADINGS
    skills_2d = base_skills(args.num_skills)

    all_metric_rows = []
    all_records = []

    for method in args.methods:
        method = method.lower()
        for seed in args.seeds:
            checkpoint_path = overrides.get(method, resolve_checkpoint(args.checkpoint_root, method, seed, args.checkpoint_epoch))
            if checkpoint_path is None or not checkpoint_path.exists():
                message = (
                    f"[skip] {method} seed={seed}: checkpoint not found under "
                    f"{Path(args.checkpoint_root) / method.upper()} for epoch={args.checkpoint_epoch}"
                )
                if args.skip_missing:
                    print(message)
                    continue
                raise FileNotFoundError(message)

            print(f"[load] {method} seed={seed}: {checkpoint_path}")
            policy = load_policy(checkpoint_path, device)

            env = make_env(seed, args.normalize_obs, args.render_hw)
            obs_dim = int(env.observation_space.shape[0])
            skill_dim = infer_skill_dim(policy, obs_dim)
            skills = np.asarray([expand_skill(skill, skill_dim) for skill in skills_2d])
            records = []
            for skill_idx, skill in enumerate(skills):
                for theta in headings:
                    for rollout_idx in range(args.num_eval_rollouts):
                        tau_world, tau_body = rollout(
                            env=env,
                            policy=policy,
                            skill=skill,
                            theta=theta,
                            horizon=args.horizon,
                            device=device,
                            stochastic=args.stochastic,
                        )
                        records.append(
                            {
                                "method": method,
                                "seed": seed,
                                "rollout_idx": rollout_idx,
                                "skill_idx": skill_idx,
                                "theta": theta,
                                "skill": skill,
                                "tau_world": tau_world,
                                "tau_body": tau_body,
                            }
                        )
            metric_rows = compute_metrics(records, args.num_skills, headings)
            for row in metric_rows:
                row["method"] = method
                row["seed"] = seed
            all_metric_rows.extend(metric_rows)
            all_records.extend(records)
            plot_method(
                records=records,
                method=f"{method}_seed{seed}",
                out_path=output_dir / f"{method}_seed{seed}_trajectories.png",
                num_skills=args.num_skills,
                headings=headings,
            )
            print(f"[done] {method} seed={seed} skill_dim={skill_dim}")

    save_metrics_csv(output_dir / "metrics_by_skill.csv", all_metric_rows)
    if all_records:
        save_rollouts_npz(output_dir / "rollouts.npz", all_records)
    summary_rows = summarize(all_metric_rows)
    save_summary_csv(output_dir / "summary.csv", summary_rows)
    print_summary(summary_rows)
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
