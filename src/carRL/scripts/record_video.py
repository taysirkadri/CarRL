"""
Record a video of a trained RL agent on Gymnasium CarRacing-v3.
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC
from carRL.envs import make_wrapped_carracing


def _video_folder_from_arg(video_path: str) -> Path:
    p = Path(video_path)
    if p.suffix.lower() == ".mp4":
        return p.parent if p.parent != Path("") else Path(".")
    return p


def main(model_path: str | None = None, video_path: str = "videos/agent.mp4", seed: int = 0):
    if model_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-path", type=str, required=True)
        parser.add_argument("--video-path", type=str, default="videos/agent.mp4")
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args()
        model_path, video_path, seed = args.model_path, args.video_path, args.seed

    folder = _video_folder_from_arg(video_path)
    existing = {p.resolve() for p in folder.glob("*.mp4")} if folder.exists() else set()

    env = make_wrapped_carracing(
        seed=seed,
        record_video_path=video_path,
        grayscale=True,
        resize=84,
        frame_stack=4,
    )
    algo_cls = SAC if "sac" in model_path.lower() else PPO
    model = algo_cls.load(model_path)
    try:
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    finally:
        env.close()

    generated = []
    if folder.exists():
        generated = sorted(
            (p.resolve() for p in folder.glob("*.mp4") if p.resolve() not in existing),
            key=lambda p: p.stat().st_mtime,
        )
    if generated:
        print(f"Video saved under {generated[-1]}")
    else:
        print(f"Video saved under {folder.resolve()}")


if __name__ == "__main__":
    main()
