"""
Evaluate a trained RL agent on Gymnasium CarRacing-v3.
"""

import argparse
import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage

from src.carRL.envs import make_wrapped_carracing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = DummyVecEnv(
        [
            lambda: make_wrapped_carracing(
                seed=args.seed,
                grayscale=True,
                resize=84,
                frame_stack=4,
            )
        ]
    )
    env = VecTransposeImage(env)
    vec_norm_path = os.path.join(os.path.dirname(args.model_path), "vecnormalize.pkl")
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    algo_cls = SAC if "sac" in args.model_path.lower() else PPO
    model = algo_cls.load(args.model_path)
    rewards = []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward)
        rewards.append(ep_reward)
        print(f"Episode {ep+1}: Reward = {ep_reward:.2f}")
    print(f"Mean reward over {args.episodes} episodes: {sum(rewards)/len(rewards):.2f}")


if __name__ == "__main__":
    main()
