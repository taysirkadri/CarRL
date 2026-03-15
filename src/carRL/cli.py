import argparse
import sys

from .scripts.train import main as train_main
from .scripts.eval import main as eval_main
from .scripts.record_video import main as record_video_main


def main():
    parser = argparse.ArgumentParser(description="carRL — Self-Driving Car RL Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_p = subparsers.add_parser("train", help="Train an RL agent")
    train_p.add_argument(
        "--config", type=str, default="configs/baseline.yaml", help="Path to YAML config"
    )

    # Eval
    eval_p = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_p.add_argument("--model-path", type=str, required=True, help="Path to saved model")
    eval_p.add_argument("--episodes", type=int, default=10)
    eval_p.add_argument("--seed", type=int, default=0)

    # Record video
    vid_p = subparsers.add_parser("record-video", help="Record agent video")
    vid_p.add_argument("--model-path", type=str, required=True, help="Path to saved model")
    vid_p.add_argument("--video-path", type=str, default="videos/agent.mp4")
    vid_p.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.command == "train":
        train_main(args.config)
    elif args.command == "eval":
        eval_main(args.model_path, args.episodes, args.seed)
    elif args.command == "record-video":
        record_video_main(args.model_path, args.video_path, args.seed)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
