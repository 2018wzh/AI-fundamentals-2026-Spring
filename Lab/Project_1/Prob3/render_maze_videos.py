"""Batch render the maze search manim scenes.

Usage:
    python Lab/Project_1/Prob3/render_maze_videos.py

Optional:
    python Lab/Project_1/Prob3/render_maze_videos.py --quality low
    python Lab/Project_1/Prob3/render_maze_videos.py --open
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCENES = [
    "BFSMazeSearchScene",
    "DFSMazeSearchScene",
    "DijkstraMazeSearchScene",
    "AStarMazeSearchScene",
]

QUALITY_FLAGS = {
    "low": "-ql",
    "medium": "-qm",
    "high": "-qh",
    "production": "-qk",
}


def build_command(script_path: Path, scene: str, quality: str, preview: bool) -> list[str]:
    cmd = [sys.executable, "-m", "manim", QUALITY_FLAGS[quality], str(script_path), scene]
    if preview:
        cmd.append("-p")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch render maze search scenes with manim.")
    parser.add_argument(
        "--quality",
        choices=sorted(QUALITY_FLAGS),
        default="high",
        help="Render quality preset.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open each rendered video after completion.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).with_name("maze_visualization.py")
    if not script_path.exists():
        print(f"Missing scene file: {script_path}", file=sys.stderr)
        return 1

    failures: list[tuple[str, int]] = []
    for scene in SCENES:
        print(f"[render] {scene} ...")
        cmd = build_command(script_path, scene, args.quality, args.open)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append((scene, result.returncode))
            print(f"[fail] {scene} exited with code {result.returncode}", file=sys.stderr)
        else:
            print(f"[ok] {scene}")

    if failures:
        print("\nCompleted with failures:", file=sys.stderr)
        for scene, code in failures:
            print(f"  - {scene}: exit {code}", file=sys.stderr)
        return 1

    print("\nAll scenes rendered successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
