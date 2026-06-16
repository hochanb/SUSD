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

import dowel_wrapper
assert dowel_wrapper is not None

from gym import spaces
import gym
from scipy import stats
from scipy.interpolate import interp1d
import copy
from types import SimpleNamespace

from garage.torch.distributions import TanhNormal
from garagei.torch.modules.gaussian_mlp_module_ex import GaussianMLPTwoHeadedModuleEx
from garagei.torch.modules.parameter_module import ParameterModule
from garagei.torch.policies.policy_ex import PolicyEx
from garagei.torch.q_functions.continuous_mlp_q_function_ex import ContinuousMLPQFunctionEx
from iod import sac_utils

from downstream_tasks.ant_maze_goal import AntMazeGoalEnv
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate meta-policies on Ant Maze using pretrained skills.")
    parser.add_argument("--mode", choices=["train", "plot", "all"], default="all")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--checkpoint-root", default="final_models/ant")
    parser.add_argument("--checkpoint-epoch", default="latest")
    parser.add_argument("--output-dir", default="results/ant_maze_meta_policy")
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--skill-steps", type=int, default=10)
    parser.add_argument("--maze-horizon", type=int, default=400)
    parser.add_argument("--goal-x", type=float, default=6.0)
    parser.add_argument("--goal-y", type=float, default=6.0)
    parser.add_argument("--goal-radius", type=float, default=0.75)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--min-buffer-size", type=int, default=1000)
    parser.add_argument("--gradient-steps", type=int, default=50)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target-coef", type=float, default=1.0)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[512, 512])
    parser.add_argument("--init-alpha", type=float, default=1.0)
    parser.add_argument("--unit-skill", action="store_true", default=True)
    parser.add_argument("--no-unit-skill", dest="unit_skill", action="store_false")
    parser.add_argument("--stochastic-low-policy", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(skip_missing=True)
    parser.add_argument("--skip-missing", dest="skip_missing", action="store_true")
    parser.add_argument("--no-skip-missing", dest="skip_missing", action="store_false")
    return parser.parse_args()


def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def extract_epoch(path, kind):
    suffix = path.stem.replace(kind, "", 1)
    return int(suffix) if suffix.isdigit() else -1


def checkpoint_candidates(root, method, seed, epoch, kind):
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


def resolve_checkpoint(root, method, seed, epoch, kind):
    candidates = checkpoint_candidates(root, method, seed, epoch, kind)
    return candidates[-1] if candidates else None


def load_policy(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    policy = checkpoint["policy"] if isinstance(checkpoint, dict) and "policy" in checkpoint else checkpoint
    return policy.to(device).eval()


def policy_input_dim(policy):
    module = getattr(policy, "_module", None)
    for obj in [module, getattr(module, "module", None)]:
        if obj is None:
            continue
        for attr in ["_input_dim", "input_dim"]:
            value = getattr(obj, attr, None)
            if value is not None:
                return int(value)
    raise ValueError("Could not infer option policy input dimension.")


def infer_skill_dim(policy, obs_dim):
    skill_dim = policy_input_dim(policy) - int(obs_dim)
    if skill_dim <= 0:
        raise ValueError(f"Invalid skill_dim={skill_dim}")
    return skill_dim


def load_method_policy(args, method, seed):
    option_path = resolve_checkpoint(args.checkpoint_root, method, seed, args.checkpoint_epoch, "option_policy")
    if option_path is None:
        msg = f"Missing option_policy checkpoint for method={method} seed={seed}"
        if args.skip_missing:
            print(f"[skip] {msg}")
            return None, None
        raise FileNotFoundError(msg)
    return load_policy(option_path, args.device), option_path


def make_ant_maze_env(args, seed):
    base_env = AntMazeGoalEnv(
        goal=(args.goal_x, args.goal_y),
        goal_radius=args.goal_radius,
        max_path_length=args.maze_horizon,
        render_hw=100,
    )
    if hasattr(base_env, "seed"):
        base_env.seed(seed)
    mean, std = get_normalizer_preset("ant_preset")
    return consistent_normalize(base_env, normalize_obs=True, mean=mean, std=std)


class AntMazeMetaPolicyEnv(gym.Env):
    def __init__(self, env, option_policy, method, skill_steps, device, unit_skill=True, stochastic_low_policy=False):
        super().__init__()
        self.env = env
        self.option_policy = option_policy.to(device).eval()
        self.method = method
        self.skill_steps = int(skill_steps)
        self.device = device
        self.unit_skill = bool(unit_skill)
        self.stochastic_low_policy = bool(stochastic_low_policy)
        self.current_obs = None
        obs_dim = int(np.prod(env.observation_space.shape))
        self.skill_dim = infer_skill_dim(option_policy, obs_dim)
        self.is_poe_control = method == "dads_poe" and hasattr(option_policy._module, "forward_mode_with_weights")
        if self.is_poe_control:
            self.control_dim = int(option_policy._module.num_heads)
            self.control_name = "w"
        else:
            self.control_dim = self.skill_dim
            self.control_name = "z"
        obs_low = np.full(obs_dim + 2, -np.inf, dtype=np.float32)
        obs_high = np.full(obs_dim + 2, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.control_dim,), dtype=np.float32)

    def _meta_obs(self):
        base = unwrap_env(self.env)
        raw_xy = base.sim.data.qpos[:2]
        goal_delta = (base.goal - raw_xy) / 10.0
        return np.concatenate([np.asarray(self.current_obs, dtype=np.float32), goal_delta.astype(np.float32)])

    def reset(self):
        self.current_obs = self.env.reset()
        return self._meta_obs()

    def _low_action_with_z(self, control):
        z = np.asarray(control, dtype=np.float32)
        if self.unit_skill:
            z = z / (np.linalg.norm(z) + 1e-8)
        obs_t = torch.as_tensor(self.current_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        z_t = torch.as_tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
        policy_input = torch.cat([obs_t, z_t], dim=-1)
        with torch.no_grad():
            if self.stochastic_low_policy:
                action, _ = self.option_policy.get_actions(policy_input)
            elif hasattr(self.option_policy, "get_mode_actions"):
                action, _ = self.option_policy.get_mode_actions(policy_input)
            else:
                action, _ = self.option_policy.get_actions(policy_input)
        return action[0]

    def _low_action_with_w(self, control):
        logits = torch.as_tensor(control, dtype=torch.float32, device=self.device).unsqueeze(0)
        weights = torch.softmax(logits, dim=-1)
        obs_t = torch.as_tensor(self.current_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.stochastic_low_policy:
                dist = self.option_policy._module.forward_with_weights(obs_t, weights)
                action = dist.sample().detach().cpu().numpy()
            else:
                action = self.option_policy._module.forward_mode_with_weights(obs_t, weights).detach().cpu().numpy()
        return action[0]

    def step(self, control):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self.skill_steps):
            if self.is_poe_control:
                action = self._low_action_with_w(control)
            else:
                action = self._low_action_with_z(control)
            self.current_obs, reward, done, info = self.env.step(action)
            total_reward += float(reward)
            if done:
                break
        info = dict(info)
        info["control_name"] = self.control_name
        return self._meta_obs(), total_reward, done, info


def make_meta_env(args, method, seed, option_policy):
    env = make_ant_maze_env(args, seed)
    return AntMazeMetaPolicyEnv(
        env=env,
        option_policy=option_policy,
        method=method,
        skill_steps=args.skill_steps,
        device=args.device,
        unit_skill=args.unit_skill,
        stochastic_low_policy=args.stochastic_low_policy,
    )


class TransitionReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.storage = []
        self.next_idx = 0

    def add(self, obs, action, reward, next_obs, done):
        transition = (
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.asarray([reward], dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray([done], dtype=np.float32),
        )
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.next_idx] = transition
        self.next_idx = (self.next_idx + 1) % self.capacity

    def __len__(self):
        return len(self.storage)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, len(self.storage), size=batch_size)
        obs, actions, rewards, next_obs, dones = zip(*(self.storage[i] for i in idx))
        return {
            "obs": torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device),
            "actions": torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(np.asarray(rewards).reshape(-1), dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(np.asarray(next_obs), dtype=torch.float32, device=device),
            "dones": torch.as_tensor(np.asarray(dones).reshape(-1), dtype=torch.float32, device=device),
        }


class RepoSACTrainer:
    def __init__(self, obs_dim, action_dim, args):
        self.device = torch.device(args.device)
        self.discount = args.discount
        self.tau = args.tau
        self._target_entropy = -float(action_dim) / 2.0 * args.target_coef
        module = GaussianMLPTwoHeadedModuleEx(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_sizes=tuple(args.hidden_sizes),
            hidden_nonlinearity=torch.relu,
            normal_distribution_cls=TanhNormal,
            init_std=1.0,
            max_std=np.exp(2.0),
        )
        self.policy = PolicyEx(name="meta_policy", module=module).to(self.device)
        self.qf1 = ContinuousMLPQFunctionEx(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=tuple(args.hidden_sizes),
            hidden_nonlinearity=torch.relu,
        ).to(self.device)
        self.qf2 = ContinuousMLPQFunctionEx(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=tuple(args.hidden_sizes),
            hidden_nonlinearity=torch.relu,
        ).to(self.device)
        self.target_qf1 = copy.deepcopy(self.qf1).to(self.device)
        self.target_qf2 = copy.deepcopy(self.qf2).to(self.device)
        self.log_alpha = ParameterModule(torch.tensor([np.log(args.init_alpha)], dtype=torch.float32)).to(self.device)
        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=args.learning_rate)
        self.optim_qf = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.learning_rate)
        self.optim_alpha = torch.optim.Adam(self.log_alpha.parameters(), lr=args.learning_rate)

        low = -np.ones(action_dim, dtype=np.float32)
        high = np.ones(action_dim, dtype=np.float32)
        self._env_spec = SimpleNamespace(action_space=SimpleNamespace(low=low, high=high, shape=(action_dim,)))

    def predict(self, obs, deterministic=True):
        obs_batch = np.asarray(obs, dtype=np.float32)[None]
        if deterministic and hasattr(self.policy, "get_mode_actions"):
            action, _ = self.policy.get_mode_actions(obs_batch)
        else:
            action, _ = self.policy.get_actions(obs_batch)
        return action[0]

    def update(self, batch):
        tensors = {}
        sac_utils.update_loss_qf(
            self,
            tensors,
            batch,
            obs=batch["obs"],
            actions=batch["actions"],
            next_obs=batch["next_obs"],
            dones=batch["dones"],
            rewards=batch["rewards"],
            policy=self.policy,
        )
        self.optim_qf.zero_grad()
        (tensors["LossQf1"] + tensors["LossQf2"]).backward()
        self.optim_qf.step()

        sac_utils.update_loss_sacp(self, tensors, batch, obs=batch["obs"], policy=self.policy)
        self.optim_policy.zero_grad()
        tensors["LossSacp"].backward()
        self.optim_policy.step()

        sac_utils.update_loss_alpha(self, tensors, batch)
        self.optim_alpha.zero_grad()
        tensors["LossAlpha"].backward()
        self.optim_alpha.step()

        sac_utils.update_targets(self)
        return {k: float(v.detach().cpu()) for k, v in tensors.items() if torch.is_tensor(v) and v.numel() == 1}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy,
                "qf1": self.qf1,
                "qf2": self.qf2,
                "target_qf1": self.target_qf1,
                "target_qf2": self.target_qf2,
                "log_alpha": self.log_alpha,
            },
            path,
        )

    @classmethod
    def load(cls, path, obs_dim, action_dim, args):
        trainer = cls(obs_dim, action_dim, args)
        try:
            checkpoint = torch.load(path, map_location=args.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=args.device)
        trainer.policy = checkpoint["policy"].to(args.device).eval()
        trainer.qf1 = checkpoint["qf1"].to(args.device)
        trainer.qf2 = checkpoint["qf2"].to(args.device)
        trainer.target_qf1 = checkpoint["target_qf1"].to(args.device)
        trainer.target_qf2 = checkpoint["target_qf2"].to(args.device)
        trainer.log_alpha = checkpoint["log_alpha"].to(args.device)
        return trainer


def evaluate_model(model, args, method, seed, option_policy):
    rewards = []
    distances = []
    successes = []
    for ep in range(args.n_eval_episodes):
        env = make_meta_env(args, method, seed + 1000 + ep, option_policy)
        obs = env.reset()
        done = False
        total_reward = 0.0
        info = {}
        while not done:
            action = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
        rewards.append(total_reward)
        distances.append(float(info.get("distance_to_goal", np.nan)))
        successes.append(float(info.get("success", False)))
    return {
        "total_reward": float(np.mean(rewards)),
        "distance_to_goal": float(np.nanmean(distances)),
        "success_rate": float(np.mean(successes)),
    }


def append_metrics(metrics_path, row):
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "seed", "timestep", "total_reward", "distance_to_goal", "success_rate", "control"]
    exists = metrics_path.exists()
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_one(args, method, seed):
    option_policy, option_path = load_method_policy(args, method, seed)
    if option_policy is None:
        return
    run_dir = Path(args.output_dir) / method / f"seed_{seed}"
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    env = make_meta_env(args, method, seed, option_policy)
    obs = env.reset()
    trainer = RepoSACTrainer(obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], args=args)
    replay = TransitionReplayBuffer(args.buffer_size)
    control = "w" if method == "dads_poe" else "z"

    print(f"[train] method={method} seed={seed} control={control} checkpoint={option_path}")

    def record_eval(timestep):
        metrics = evaluate_model(trainer, args, method, seed, option_policy)
        row = {
            "method": method,
            "seed": seed,
            "timestep": int(timestep),
            "total_reward": metrics["total_reward"],
            "distance_to_goal": metrics["distance_to_goal"],
            "success_rate": metrics["success_rate"],
            "control": control,
        }
        append_metrics(metrics_path, row)
        print(
            f"[eval] method={method} seed={seed} step={timestep} "
            f"reward={row['total_reward']:.2f} dist={row['distance_to_goal']:.2f} success={row['success_rate']:.2f}",
            flush=True,
        )

    record_eval(0)
    for timestep in range(1, args.total_timesteps + 1):
        action = trainer.predict(obs, deterministic=False)
        next_obs, reward, done, _ = env.step(action)
        replay.add(obs, action, reward, next_obs, float(done))
        obs = next_obs if not done else env.reset()

        if len(replay) >= args.min_buffer_size and timestep % args.train_freq == 0:
            for _ in range(args.gradient_steps):
                batch = replay.sample(args.batch_size, trainer.device)
                trainer.update(batch)

        if timestep % args.eval_freq == 0:
            record_eval(timestep)
            trainer.save(model_dir / f"sac_ant_maze_meta_{timestep}_steps.pt")

    trainer.save(model_dir / "sac_ant_maze_meta_final.pt")

def load_metrics(output_dir):
    frames = []
    for path in sorted(Path(output_dir).glob("*/seed_*/metrics.csv")):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_metric(df, methods, metric, ylabel, out_path):
    if df.empty:
        print(f"[skip] no metrics for {metric}")
        return
    max_step = int(df["timestep"].max())
    common_steps = np.arange(0, max_step + 1)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for method in methods:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        seed_values = []
        for _, seed_df in sub.groupby("seed"):
            seed_df = seed_df.sort_values("timestep")
            f = interp1d(
                seed_df["timestep"],
                seed_df[metric],
                kind="previous",
                bounds_error=False,
                fill_value=(seed_df[metric].iloc[0], seed_df[metric].iloc[-1]),
            )
            seed_values.append(f(common_steps))
        seed_values = np.asarray(seed_values)
        mean = seed_values.mean(axis=0)
        if len(seed_values) > 1:
            sem = stats.sem(seed_values, axis=0)
            margin = sem * stats.t.ppf(0.975, len(seed_values) - 1)
        else:
            margin = np.zeros_like(mean)
        ax.plot(common_steps, mean, label=METHOD_LABELS.get(method, method.upper()), linewidth=1.8)
        ax.fill_between(common_steps, mean - margin, mean + margin, alpha=0.15)
    ax.set_xlabel("Meta-policy training timesteps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_all(args):
    df = load_metrics(args.output_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.lower() for m in args.methods]
    plot_metric(df, methods, "total_reward", "Total Reward", out_dir / "ant_maze_meta_total_reward.png")
    plot_metric(df, methods, "distance_to_goal", "Distance to Goal", out_dir / "ant_maze_meta_distance_to_goal.png")
    plot_metric(df, methods, "success_rate", "Success Rate", out_dir / "ant_maze_meta_success_rate.png")


def main():
    args = parse_args()
    methods = [m.lower() for m in args.methods]
    if args.mode in ["train", "all"]:
        for method in methods:
            for seed in args.seeds:
                train_one(args, method, seed)
    if args.mode in ["plot", "all"]:
        plot_all(args)


if __name__ == "__main__":
    main()
