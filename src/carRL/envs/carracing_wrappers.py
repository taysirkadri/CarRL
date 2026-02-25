import os
import gymnasium as gym
from typing import Optional, Tuple


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
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
    if frame_stack and frame_stack > 1:
        env = gym.wrappers.FrameStack(env, frame_stack)
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
