import argparse
from pathlib import Path

import kagglehub


DATASET_MAP = {
    "fintabnet": {
        "kaggle_repo": "lewisdo/fintabnet",
        "default_out_subdir": "data/fintabnet",
    },
    "pubtables-1m-structure": {
        "kaggle_repo": "bsmock/pubtables-1m-structure",
        "default_out_subdir": "data/pubtables-1m-structure",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download selected dataset via kagglehub."
    )
    parser.add_argument(
        "--type",
        choices=sorted(DATASET_MAP.keys()),
        help="Dataset type to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to put the downloaded dataset (default: ./data/<dataset>).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output directory already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DATASET_MAP[args.type]

    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg["default_out_subdir"])
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # kagglehub itself chooses the final directory; we use out_dir as the base output_dir.
    if out_dir.exists() and not args.force:
        print(f"[skip] Output directory already exists: {out_dir}")
        print("Use --force to re-download.")
        return

    try:
        # Download latest version.
        # kagglehub returns the resolved local path of the dataset.
        resolved_path = kagglehub.dataset_download(
            cfg["kaggle_repo"],
            output_dir=str(out_dir),
        )
    except Exception as e:  # pragma: no cover (depends on network / kagglehub behavior)
        raise SystemExit(f"Download failed for {args.type}: {e}") from e

    print("Downloaded dataset:", args.type)
    print("Path to dataset files:", resolved_path)


if __name__ == "__main__":
    main()