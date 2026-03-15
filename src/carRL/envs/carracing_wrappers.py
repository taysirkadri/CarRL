import os
import gymnasium as gym
from typing import Optional, Tuple


class _FrameStackToHWC(gym.ObservationWrapper):
    """Reshape FrameStackObservation output for SB3 compatibility.

    Converts (stack, H, W, C) → (H, W, stack*C) so the result is a
    standard 3-D HWC image that VecTransposeImage can handle.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        old = env.observation_space
        n_stack, h, w = old.shape[0], old.shape[1], old.shape[2]
        c = old.shape[3] if len(old.shape) == 4 else 1
        self._squeeze = len(old.shape) == 4
        self._n = n_stack
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, n_stack * c), dtype=old.dtype
        )

    def observation(self, obs):
        if self._squeeze:
            n, h, w, c = obs.shape
            return obs.transpose(1, 2, 0, 3).reshape(h, w, n * c)
        return obs.transpose(1, 2, 0)


def _resolve_video_path(record_video_path: str) -> Tuple[str, str]:
    if record_video_path.lower().endswith(".mp4"):
        folder = os.path.dirname(record_video_path)
        name_prefix = os.path.splitext(os.path.basename(record_video_path))[0]
    else:
        folder = record_video_path
        name_prefix = "agent"
    return folder or ".", name_prefix


def make_wrapped_carracing(
    seed: Optional[int] = None,
    record_video_path: Optional[str] = None,
    render_mode: Optional[str] = None,
    grayscale: bool = True,
    resize: Optional[int] = 84,
    frame_stack: int = 4,
    clip_action: bool = True,
) -> gym.Env:
    resolved_render_mode = render_mode or "rgb_array"
    if record_video_path:
        resolved_render_mode = "rgb_array"
    env = gym.make("CarRacing-v3", render_mode=resolved_render_mode)
    if seed is not None:
        env.reset(seed=seed)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if clip_action:
        env = gym.wrappers.ClipAction(env)
    if resize is not None:
        env = gym.wrappers.ResizeObservation(env, (resize, resize))
    if grayscale:
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
    if frame_stack and frame_stack > 1:
        env = gym.wrappers.FrameStackObservation(env, frame_stack)
        env = _FrameStackToHWC(env)
    # TODO: Add wrappers for noise, delay, domain randomization
    if record_video_path:
        from gymnasium.wrappers import RecordVideo

        video_folder, name_prefix = _resolve_video_path(record_video_path)
        os.makedirs(video_folder, exist_ok=True)

        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=lambda x: True,
        )
    return env
