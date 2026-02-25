"""
Record a video of a trained RL agent on Gymnasium CarRacing-v3.
"""

import argparse
from stable_baselines3 import PPO
from src.carRL.envs import make_wrapped_carracing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--video-path", type=str, default="videos/agent.mp4")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_wrapped_carracing(
        seed=args.seed,
        record_video_path=args.video_path,
        grayscale=True,
        resize=84,
        frame_stack=4,
    )
    model = PPO.load(args.model_path)
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    print(f"Video saved under {args.video_path}")


if __name__ == "__main__":
    main()
