"""Command-line interface for ``aerialgeometry``.

Drop-in replacement for the original research script ``infer.py``: given a
folder of images it writes ``predictions.txt`` (CSV) with columns
``image_path,height,distance,angle``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Optional, Sequence, Tuple

from . import __version__
from .predictor import Prediction, Predictor


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aerialgeometry",
        description=(
            "Estimate drone height, distance and viewing angle from aerial images "
            "(arXiv:2601.21405)."
        ),
    )
    p.add_argument(
        "input",
        help="Folder of images (scanned recursively) or a single image file.",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output CSV file. For a folder input the default is "
            "<input>/predictions.txt; for a single image the default is stdout."
        ),
    )
    p.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to the trained checkpoint. Defaults to cached weights; "
            "see README for AERIALGEOMETRY_MODEL / AERIALGEOMETRY_MODEL_URL."
        ),
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. 'cpu' or 'cuda:0' (default: auto).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images to process per batch (default: 8).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def write_results(results: List[Tuple[str, Prediction]], output_path: str) -> None:
    """Write ``(path, prediction)`` pairs to a CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["image_path", "height", "distance", "angle"])
        for img_path, pred in results:
            writer.writerow([img_path, f"{pred.height:.4f}", f"{pred.distance:.4f}", f"{pred.angle:.1f}"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    predictor = Predictor(model_path=args.model_path, device=args.device)

    input_path = os.path.abspath(args.input)
    if os.path.isdir(input_path):
        results = predictor.predict_folder(input_path, batch_size=args.batch_size)
        output = args.output or os.path.join(input_path, "predictions.txt")
        write_results(results, output)
        print(f"Saved {len(results)} predictions to {output}")
        return 0
    if os.path.isfile(input_path):
        pred = predictor.predict(input_path)
        if args.output:
            write_results([(input_path, pred)], args.output)
            print(f"Saved 1 prediction to {args.output}")
        else:
            print(f"{input_path}\t{pred.height:.4f}\t{pred.distance:.4f}\t{pred.angle:.1f}")
        return 0

    print(f"error: input not found: {args.input}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
