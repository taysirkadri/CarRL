"""
carRL - Premium LinkedIn visuals.
Dark cinematic style with real training metrics.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

from carRL.envs import make_wrapped_carracing

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS = PROJECT_ROOT / "assets"
LOGS_ROOT = PROJECT_ROOT / "logs"
ASSETS.mkdir(exist_ok=True)

# Brand system
BG = "#05080F"
BG2 = "#0B1120"
SURFACE = "#111927"
NEON_BLUE = "#00D4FF"
NEON_CYAN = "#00FFD1"
ELECTRIC = "#7B61FF"
HOT_PINK = "#FF3CAC"
FIRE = "#FF6B35"
GOLD = "#FFD60A"
WHITE = "#F0F6FC"
MUTED = "#6E7681"
SILVER = "#C9D1D9"


def _glow_text(target, x, y, text, fontsize, color, **kwargs):
    """Draw text with soft glow."""
    target.text(
        x,
        y,
        text,
        fontsize=fontsize,
        color=color,
        path_effects=[
            pe.withStroke(linewidth=8, foreground=color, alpha=0.15),
            pe.withStroke(linewidth=4, foreground=color, alpha=0.25),
            pe.Normal(),
        ],
        **kwargs,
    )


def _pill_badge(ax, x, y, text, color, fontsize=9, width=0.09):
    """Draw a rounded badge chip."""
    rect = FancyBboxPatch(
        (x - width / 2, y - 0.025),
        width,
        0.05,
        boxstyle="round,pad=0.012",
        facecolor=color,
        alpha=0.15,
        edgecolor=color,
        linewidth=1.2,
        transform=ax.transAxes,
        zorder=5,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
        transform=ax.transAxes,
        zorder=6,
    )


def _load_real_eval_series(logs_root: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    """Load real eval metrics from the latest evaluations.npz file."""
    candidates = sorted(
        logs_root.rglob("evaluations.npz"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("No evaluations.npz found under logs/.")

    source = candidates[-1]
    with np.load(source) as data:
        if "timesteps" not in data.files or "results" not in data.files:
            raise ValueError(f"Missing expected arrays in {source}.")
        timesteps = np.asarray(data["timesteps"], dtype=float).reshape(-1)
        results = np.asarray(data["results"], dtype=float)

    if timesteps.size == 0 or results.size == 0:
        raise ValueError(f"No evaluation data found in {source}.")

    rewards = results.reshape(-1) if results.ndim == 1 else results.mean(axis=1)

    if rewards.size != timesteps.size:
        n = min(rewards.size, timesteps.size)
        timesteps = timesteps[:n]
        rewards = rewards[:n]

    return timesteps, rewards, source


def _relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _capture_frame(seed: int = 42) -> np.ndarray:
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    action = np.array([0.4, 0.7, 0.0], dtype=np.float32)
    try:
        env.reset(seed=seed)
        for _ in range(85):
            env.step(action)
        return env.render()
    finally:
        env.close()


def _capture_processed_obs(
    *,
    grayscale: bool,
    resize: int,
    frame_stack: int,
    seed: int = 42,
) -> np.ndarray:
    env = make_wrapped_carracing(
        seed=seed,
        grayscale=grayscale,
        resize=resize,
        frame_stack=frame_stack,
    )
    action = np.array([0.4, 0.7, 0.0], dtype=np.float32)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    try:
        env.reset()
        for _ in range(85):
            env.step(action)
        obs, *_ = env.step(noop)
        return obs
    finally:
        env.close()


def slide_1_hero() -> None:
    frame = _capture_frame(seed=42)

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.imshow(frame, extent=[0, 1, 0, 1], aspect="auto", alpha=0.28, zorder=1)

    y_grid, x_grid = np.mgrid[0:200, 0:400]
    center_x, center_y = 200, 100
    dist = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / 220
    vignette = np.zeros((200, 400, 4))
    vignette[..., 3] = np.clip(dist**1.6, 0, 0.88)
    ax.imshow(vignette, extent=[0, 1, 0, 1], aspect="auto", zorder=2)

    _glow_text(
        ax,
        0.50,
        0.62,
        "carRL",
        fontsize=76,
        color=NEON_BLUE,
        ha="center",
        va="center",
        fontweight="heavy",
        fontfamily="sans-serif",
        zorder=10,
    )

    ax.text(
        0.50,
        0.445,
        "SELF-DRIVING CAR  x  REINFORCEMENT LEARNING",
        fontsize=14.5,
        color=WHITE,
        ha="center",
        va="center",
        fontweight="bold",
        fontfamily="sans-serif",
        zorder=10,
        alpha=0.92,
    )

    ax.plot([0.28, 0.72], [0.385, 0.385], color=NEON_BLUE, linewidth=1.2, alpha=0.45, zorder=10)

    ax.text(
        0.50,
        0.31,
        "Pixel Input  -  PPO and SAC  -  End-to-End Pipeline",
        fontsize=11,
        color=MUTED,
        ha="center",
        va="center",
        fontfamily="sans-serif",
        zorder=10,
    )

    badges = [
        ("Python", NEON_CYAN, 0.32),
        ("Gymnasium", FIRE, 0.43),
        ("SB3", ELECTRIC, 0.54),
        ("PyTorch", HOT_PINK, 0.65),
    ]
    for label, color, x_pos in badges:
        _pill_badge(ax, x_pos, 0.15, label, color, fontsize=8.5)

    fig.savefig(
        ASSETS / "linkedin_hero.png",
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        pad_inches=0,
    )
    plt.close(fig)
    print("  [ok] Slide 1 - Hero banner")


def slide_2_preprocessing() -> None:
    frame_raw = _capture_frame(seed=42)
    obs_resized = _capture_processed_obs(grayscale=False, resize=84, frame_stack=0)
    obs_gray = _capture_processed_obs(grayscale=True, resize=84, frame_stack=0)
    obs_stacked = _capture_processed_obs(grayscale=True, resize=84, frame_stack=4)

    fig = plt.figure(figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)

    _glow_text(
        fig,
        0.50,
        0.95,
        "Observation Pipeline",
        fontsize=22,
        color=NEON_CYAN,
        ha="center",
        va="top",
        fontweight="bold",
        transform=fig.transFigure,
    )
    fig.text(
        0.50,
        0.89,
        "Raw pixels to agent-ready tensor in 3 steps",
        fontsize=11,
        color=MUTED,
        ha="center",
        va="top",
        transform=fig.transFigure,
    )

    gs = gridspec.GridSpec(1, 4, wspace=0.14, left=0.04, right=0.96, bottom=0.08, top=0.78)
    panels = [
        (frame_raw, "RAW RGB", "96 x 96 x 3", None, FIRE),
        (obs_resized, "RESIZED", "84 x 84 x 3", None, NEON_BLUE),
        (obs_gray[:, :, 0], "GRAYSCALE", "84 x 84 x 1", "gray", ELECTRIC),
        (obs_stacked[:, :, 0], "FRAME STACK", "84 x 84 x 4", "gray", NEON_CYAN),
    ]

    for idx, (img, title, dims, cmap, color) in enumerate(panels):
        ax = fig.add_subplot(gs[idx])
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5)
            spine.set_alpha(0.35)
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.imshow(img, cmap=cmap)

        _glow_text(
            ax,
            0.50,
            1.13,
            title,
            fontsize=12,
            color=color,
            ha="center",
            va="center",
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.50,
            -0.09,
            dims,
            fontsize=9,
            color=MUTED,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontfamily="monospace",
        )

        if idx < 3:
            fig.text(
                gs[idx].get_position(fig).x1 + 0.014,
                0.44,
                ">",
                fontsize=22,
                color=color,
                ha="center",
                va="center",
                fontweight="bold",
                alpha=0.65,
            )

    fig.savefig(
        ASSETS / "linkedin_preprocessing.png",
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        pad_inches=0.08,
    )
    plt.close(fig)
    print("  [ok] Slide 2 - Preprocessing pipeline")


def slide_3_architecture() -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _glow_text(
        ax,
        0.50,
        0.93,
        "Training Architecture",
        fontsize=22,
        color=ELECTRIC,
        ha="center",
        va="center",
        fontweight="bold",
    )

    stages = [
        ("CarRacing-v3\nEnvironment", FIRE),
        ("Preprocess\nGrayscale - Resize\nStack x4", NEON_BLUE),
        ("VecEnv x4\n+ Reward\nNormalization", NEON_CYAN),
        ("PPO / SAC\nCNN Policy\n(PyTorch)", ELECTRIC),
        ("Trained\nAgent", GOLD),
    ]

    n_steps = len(stages)
    box_w, box_h = 0.14, 0.50
    total_w = n_steps * box_w + (n_steps - 1) * 0.035
    start_x = (1 - total_w) / 2

    for idx, (label, color) in enumerate(stages):
        x_pos = start_x + idx * (box_w + 0.035)
        y_pos = 0.16

        glow = FancyBboxPatch(
            (x_pos - 0.01, y_pos - 0.01),
            box_w + 0.02,
            box_h + 0.02,
            boxstyle="round,pad=0.025",
            facecolor=color,
            alpha=0.05,
            edgecolor="none",
            zorder=1,
        )
        ax.add_patch(glow)

        rect = FancyBboxPatch(
            (x_pos, y_pos),
            box_w,
            box_h,
            boxstyle="round,pad=0.018",
            facecolor=BG2,
            alpha=0.95,
            edgecolor=color,
            linewidth=1.8,
            zorder=2,
        )
        ax.add_patch(rect)

        badge = FancyBboxPatch(
            (x_pos + box_w / 2 - 0.016, y_pos + box_h - 0.02),
            0.032,
            0.04,
            boxstyle="round,pad=0.006",
            facecolor=color,
            alpha=0.25,
            edgecolor=color,
            linewidth=1,
            zorder=3,
        )
        ax.add_patch(badge)
        ax.text(
            x_pos + box_w / 2,
            y_pos + box_h,
            str(idx + 1),
            fontsize=9,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            zorder=4,
        )

        ax.text(
            x_pos + box_w / 2,
            y_pos + box_h * 0.42,
            label,
            fontsize=9.5,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            fontfamily="sans-serif",
            zorder=3,
            linespacing=1.5,
        )

        if idx < n_steps - 1:
            ax.annotate(
                "",
                xy=(x_pos + box_w + 0.031, y_pos + box_h / 2),
                xytext=(x_pos + box_w + 0.004, y_pos + box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.8, mutation_scale=16),
                zorder=4,
            )

    fig.savefig(
        ASSETS / "linkedin_pipeline.png",
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        pad_inches=0.05,
    )
    plt.close(fig)
    print("  [ok] Slide 3 - Architecture diagram")


def slide_4_results() -> None:
    timesteps, rewards, source = _load_real_eval_series(LOGS_ROOT)
    source_rel = _relative_path(source)
    algo = "SAC" if "sac" in source_rel.lower() else "PPO"

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor(BG)

    _glow_text(
        fig,
        0.50,
        0.97,
        "Training Results (Real Logged Data)",
        fontsize=22,
        color=GOLD,
        ha="center",
        va="top",
        fontweight="bold",
        transform=fig.transFigure,
    )

    gs = gridspec.GridSpec(
        1,
        2,
        width_ratios=[2.3, 1],
        wspace=0.06,
        left=0.07,
        right=0.95,
        bottom=0.12,
        top=0.86,
    )

    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(BG2)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.5)
    ax.tick_params(colors=MUTED, labelsize=9)

    if timesteps.size > 1:
        cmap_fill = LinearSegmentedColormap.from_list("fill", [BG2, NEON_BLUE])
        y_norm = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-9)
        for idx in range(timesteps.size - 1):
            ax.fill_between(
                timesteps[idx : idx + 2],
                rewards[idx : idx + 2],
                rewards.min() - 5,
                color=cmap_fill(float(y_norm[idx]) * 0.5),
                alpha=0.35,
                zorder=2,
            )

    ax.plot(
        timesteps,
        rewards,
        color=NEON_BLUE,
        linewidth=2.5,
        zorder=3,
        marker="o",
        markersize=5,
        path_effects=[pe.withStroke(linewidth=5, foreground=NEON_BLUE, alpha=0.2)],
    )

    best_idx = int(np.argmax(rewards))
    ax.scatter(
        [timesteps[best_idx]],
        [rewards[best_idx]],
        color=GOLD,
        s=100,
        zorder=5,
        edgecolors=WHITE,
        linewidths=1.5,
    )

    x_offset = max(float(timesteps[-1]) * 0.2, 3000.0)
    ax.annotate(
        f"Best: {rewards[best_idx]:.2f}",
        xy=(timesteps[best_idx], rewards[best_idx]),
        xytext=(max(0.0, timesteps[best_idx] - x_offset), rewards[best_idx] + 10.0),
        fontsize=9,
        fontweight="bold",
        color=GOLD,
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2),
        zorder=6,
    )

    ax.axhline(0, color=MUTED, linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_xlabel("Training Steps", color=SILVER, fontsize=11, labelpad=8)
    ax.set_ylabel("Eval Reward", color=SILVER, fontsize=11, labelpad=8)
    ax.set_xlim(0, max(1.0, float(timesteps[-1]) * 1.05))
    ax.set_title(
        f"{algo} Eval Reward Curve - CarRacing-v3",
        fontsize=13,
        fontweight="bold",
        color=WHITE,
        pad=12,
    )

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG)
    ax2.axis("off")

    panel = FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.96,
        boxstyle="round,pad=0.04",
        facecolor=SURFACE,
        edgecolor=MUTED,
        linewidth=0.8,
        alpha=0.95,
        transform=ax2.transAxes,
        zorder=1,
    )
    ax2.add_patch(panel)

    metrics = [
        ("ALGORITHM", algo, NEON_BLUE),
        ("EVAL POINTS", str(timesteps.size), NEON_CYAN),
        ("LAST STEP", f"{int(timesteps[-1]):,}", NEON_CYAN),
        ("LATEST REWARD", f"{rewards[-1]:.2f}", ELECTRIC),
        ("BEST REWARD", f"{rewards.max():.2f}", GOLD),
        ("DATA FILE", source.name, SILVER),
    ]

    ax2.text(
        0.50,
        0.94,
        "- Metrics -",
        fontsize=12,
        fontweight="bold",
        color=SILVER,
        ha="center",
        va="top",
        transform=ax2.transAxes,
        zorder=5,
    )

    for idx, (key, value, color) in enumerate(metrics):
        y_pos = 0.83 - idx * 0.125
        ax2.text(
            0.10,
            y_pos,
            key,
            fontsize=8.5,
            color=MUTED,
            fontweight="bold",
            va="center",
            transform=ax2.transAxes,
            zorder=5,
        )
        ax2.text(
            0.92,
            y_pos,
            value,
            fontsize=10,
            fontweight="bold",
            color=color,
            ha="right",
            va="center",
            transform=ax2.transAxes,
            zorder=5,
            path_effects=[pe.withStroke(linewidth=3, foreground=color, alpha=0.1)],
        )

        if idx < len(metrics) - 1:
            ax2.plot(
                [0.08, 0.92],
                [y_pos - 0.055, y_pos - 0.055],
                color=MUTED,
                linewidth=0.3,
                alpha=0.4,
                transform=ax2.transAxes,
                zorder=5,
            )

    fig.text(
        0.50,
        0.04,
        f"Real metrics source: {source_rel}",
        color=MUTED,
        fontsize=9,
        ha="center",
        va="center",
    )

    fig.savefig(
        ASSETS / "linkedin_results.png",
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        pad_inches=0.08,
    )
    plt.close(fig)
    print("  [ok] Slide 4 - Results dashboard")


def main() -> None:
    print("\n  carRL - Generating LinkedIn visuals\n")
    slide_1_hero()
    slide_2_preprocessing()
    slide_3_architecture()
    slide_4_results()
    print(f"\n  Done -> {ASSETS}\n")


if __name__ == "__main__":
    main()
