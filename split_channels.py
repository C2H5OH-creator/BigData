import argparse
from pathlib import Path

import pandas as pd

CHANNEL_OFFSETS = {
    "R": 0,
    "G": 1,
    "B": 2,
    "NIR": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SAT-4 CSV copies with selected spectral channels."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to source X_*.csv file",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        required=True,
        choices=["R", "G", "B", "NIR"],
        help="Channels to keep, for example: R or R G B",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. If omitted, filename is generated automatically.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10000,
        help="Number of rows per processing chunk",
    )
    return parser.parse_args()


def build_selected_columns(channels: list[str], total_columns: int = 3136) -> list[int]:
    offsets = sorted(CHANNEL_OFFSETS[ch] for ch in channels)
    return [i for i in range(total_columns) if i % 4 in offsets]


def build_output_path(
    input_path: Path, channels: list[str], output_path: Path | None
) -> Path:
    if output_path is not None:
        return output_path
    suffix = "_".join(channels)
    return input_path.with_name(f"{input_path.stem}_{suffix}{input_path.suffix}")


def main() -> None:
    args = parse_args()
    channels = list(dict.fromkeys(args.channels))
    selected_columns = build_selected_columns(channels)
    output_path = build_output_path(args.input, channels, args.output)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = False
    first_chunk = True

    for chunk in pd.read_csv(args.input, header=None, chunksize=args.chunksize):
        filtered = chunk.iloc[:, selected_columns]
        filtered.to_csv(
            output_path,
            index=False,
            header=write_header,
            mode="w" if first_chunk else "a",
        )
        first_chunk = False

    print(
        f"Saved {len(selected_columns)} columns for channels {channels} to: {output_path}"
    )


if __name__ == "__main__":
    main()
