"""
Basic unit-style sanity checks for wrappers.
"""

from src.carRL.envs import make_wrapped_carracing


def test_env_reset_step():
    env = make_wrapped_carracing(seed=42)
    obs, info = env.reset()
    assert obs is not None, "Observation should not be None"
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float), "Reward should be float"
    assert obs.shape[-1] == 3, "Observation should be RGB image"
    env.close()


if __name__ == "__main__":
    test_env_reset_step()
    print("Sanity check passed.")
