"""
Train a baseline RL agent (PPO/SAC) on Gymnasium CarRacing-v3.

Design goals:
- config-driven (YAML + CLI overrides)
- modular env/model/callback creation
- reproducible (seeded, saves VecNormalize stats)
- clean paths + logging
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable

import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    VecTransposeImage,
    VecEnv,
)

# ✅ IMPORTANT: update import to match your package layout
# Put make_wrapped_carracing in: src/carrl/envs/make_env.py (recommended)
from carrl.envs.make_env import make_wrapped_carracing  # type: ignore


ALGOS = {"PPO": PPO, "SAC": SAC}


@dataclass
class TrainConfig:
    # algo / run
    algo: str = "PPO"
    total_timesteps: int = 1_000_000
    seed: int = 0

    # io
    save_path: str = "models/baseline.zip"
    log_dir: str = "logs"

    # env
    n_envs: int = 4
    vec_norm: bool = True
    grayscale: bool = True
    resize: Optional[int] = 84
    frame_stack: int = 4
    clip_action: bool = True
    render: bool = False

    # eval/checkpoint
    eval_freq: int = 50_000
    eval_episodes: int = 5
    checkpoint_freq: int = 50_000  # decouple from eval if you want


# ---------------------------
# Config utilities
# ---------------------------

def load_yaml(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(cfg: TrainConfig, overrides: Dict[str, Any]) -> TrainConfig:
    data = asdict(cfg)
    for k, v in overrides.items():
        if k in data:
            data[k] = v
    return TrainConfig(**data)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)

    p.add_argument("--algo", type=str, choices=list(ALGOS.keys()), default="PPO")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--save-path", type=str, default="models/baseline.zip")
    p.add_argument("--log-dir", type=str, default="logs")

    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--vec-norm", action="store_true")
    p.add_argument("--no-vec-norm", action="store_true")

    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--no-grayscale", action="store_true")

    p.add_argument("--resize", type=int, default=84)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--no-clip-action", action="store_true")
    p.add_argument("--render", action="store_true")

    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig(
        algo=args.algo,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        save_path=args.save_path,
        log_dir=args.log_dir,
        n_envs=args.n_envs,
        vec_norm=not args.no_vec_norm,
        grayscale=not args.no_grayscale,
        resize=args.resize,
        frame_stack=args.frame_stack,
        clip_action=not args.no_clip_action,
        render=args.render,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
    )
    if args.config:
        cfg = merge_config(cfg, load_yaml(args.config))
    return cfg


# ---------------------------
# Env creation
# ---------------------------

def make_env_fn(cfg: TrainConfig, rank: int, *, is_eval: bool) -> Callable[[], Any]:
    """
    Returns a thunk that creates a single environment instance.
    rank is used to vary seed across vectorized envs.
    """
    def _init():
        env = make_wrapped_carracing(
            seed=cfg.seed + rank,
            render_mode=("human" if (cfg.render and not is_eval and rank == 0) else None),
            grayscale=cfg.grayscale,
            resize=cfg.resize,
            frame_stack=cfg.frame_stack,
            clip_action=cfg.clip_action,
        )
        # Monitor is good for episode stats
        return Monitor(env)
    return _init


def build_vec_env(cfg: TrainConfig) -> VecEnv:
    vec_cls = SubprocVecEnv if cfg.n_envs > 1 else DummyVecEnv
    env = vec_cls([make_env_fn(cfg, i, is_eval=False) for i in range(cfg.n_envs)])
    env = VecTransposeImage(env)  # (H,W,C) -> (C,H,W) for CNN policies
    if cfg.vec_norm:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def build_eval_env(cfg: TrainConfig) -> VecEnv:
    # eval typically should be single-env for clean rollouts
    env = DummyVecEnv([make_env_fn(cfg, 999, is_eval=True)])
    env = VecTransposeImage(env)
    if cfg.vec_norm:
        # IMPORTANT:
        # For eval, we usually keep obs normalization ON (same stats),
        # and reward normalization OFF for unbiased reward reporting.
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return env


def sync_vecnormalize(train_env: VecEnv, eval_env: VecEnv) -> None:
    """
    Ensures eval env uses the same obs normalization statistics as training env.
    SB3 stores stats inside VecNormalize wrapper.
    """
    if isinstance(train_env, VecNormalize) and isinstance(eval_env, VecNormalize):
        eval_env.obs_rms = train_env.obs_rms
        # Do NOT sync return_rms if norm_reward=False (keeps reporting real rewards)


# ---------------------------
# Model + callbacks
# ---------------------------

def build_model(cfg: TrainConfig, env: VecEnv):
    algo = cfg.algo.upper()
    algo_cls = ALGOS[algo]
    # You can later inject hyperparams from config here
    return algo_cls(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        seed=cfg.seed,
        tensorboard_log=str(Path(cfg.log_dir)),
    )


def build_callbacks(cfg: TrainConfig, eval_env: VecEnv, log_dir: Path) -> list:
    best_dir = log_dir / "best_model"
    ckpt_dir = log_dir / "checkpoints"
    best_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(log_dir),
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(cfg.checkpoint_freq, 10_000),
        save_path=str(ckpt_dir),
        name_prefix="agent",
    )
    return [eval_cb, ckpt_cb]


def save_artifacts(cfg: TrainConfig, model, train_env: VecEnv) -> None:
    save_path = Path(cfg.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(save_path))

    # Save VecNormalize stats so eval/record scripts can restore normalization.
    if isinstance(train_env, VecNormalize):
        vec_path = save_path.parent / "vecnormalize.pkl"
        train_env.save(str(vec_path))


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_vec_env(cfg)
    eval_env = build_eval_env(cfg)
    sync_vecnormalize(train_env, eval_env)

    model = build_model(cfg, train_env)
    callbacks = build_callbacks(cfg, eval_env, log_dir)

    model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks)
    save_artifacts(cfg, model, train_env)

    # Close cleanly (important with SubprocVecEnv)
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()