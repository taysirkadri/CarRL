"""
Evaluate a trained RL agent on Gymnasium CarRacing-v3.
"""

import argparse
import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage

from carRL.envs import make_wrapped_carracing


def main(model_path: str | None = None, episodes: int = 10, seed: int = 0):
    if model_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-path", type=str, required=True)
        parser.add_argument("--episodes", type=int, default=10)
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args()
        model_path, episodes, seed = args.model_path, args.episodes, args.seed

    env = DummyVecEnv(
        [
            lambda seed=seed: make_wrapped_carracing(
                seed=seed,
                grayscale=True,
                resize=84,
                frame_stack=4,
            )
        ]
    )
    env = VecTransposeImage(env)
    vec_norm_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    algo_cls = SAC if "sac" in model_path.lower() else PPO
    model = algo_cls.load(model_path)
    rewards = []
    try:
        for ep in range(episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, _ = env.step(action)
                ep_reward += float(reward[0])
                done = bool(dones[0])
            rewards.append(ep_reward)
            print(f"Episode {ep+1}: Reward = {ep_reward:.2f}")
        print(f"Mean reward over {episodes} episodes: {sum(rewards)/len(rewards):.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
