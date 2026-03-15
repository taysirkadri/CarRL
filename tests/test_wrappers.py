"""
Sanity checks for environment wrappers.
"""

import gymnasium as gym

from carRL.envs import make_wrapped_carracing


def _has_wrapper(env: gym.Env, wrapper_cls: type[gym.Wrapper]) -> bool:
    current = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, wrapper_cls):
            return True
        current = current.env
    return False


def test_env_defaults():
    """Default config: grayscale + resize + frame_stack."""
    env = make_wrapped_carracing(seed=42)
    obs, info = env.reset()
    assert obs is not None
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    env.close()


def test_env_rgb_no_stack():
    """Raw RGB with no frame stacking."""
    env = make_wrapped_carracing(seed=42, grayscale=False, frame_stack=0, resize=None)
    obs, info = env.reset()
    assert obs.shape == (96, 96, 3), f"Expected (96, 96, 3), got {obs.shape}"
    env.close()


def test_clip_action_enabled_by_default():
    env = make_wrapped_carracing(seed=42, frame_stack=0)
    assert _has_wrapper(env, gym.wrappers.ClipAction)
    env.close()


def test_clip_action_can_be_disabled():
    env = make_wrapped_carracing(
        seed=42,
        grayscale=False,
        resize=None,
        frame_stack=0,
        clip_action=False,
    )
    assert not _has_wrapper(env, gym.wrappers.ClipAction)
    env.close()


if __name__ == "__main__":
    test_env_defaults()
    test_env_rgb_no_stack()
    test_clip_action_enabled_by_default()
    test_clip_action_can_be_disabled()
    print("All sanity checks passed.")
