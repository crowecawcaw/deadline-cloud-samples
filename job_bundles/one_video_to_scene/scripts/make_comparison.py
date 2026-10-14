"""Encode render/compare.mp4: input frames, clay render, and textured render side by side."""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import stages  # noqa: E402


def main():
    analyze = stages.input_dir("analyze")
    render = stages.output_dir("render")
    fps = json.loads((analyze / "cameras.json").read_text()).get("fps", 24)
    inputs = [
        (analyze / "frames" / "%05d.jpg", 0),
        (render / "clay" / "%04d.png", 1),
        (render / "textured" / "%04d.png", 1),
    ]
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for pattern, start in inputs:
        cmd += ["-framerate", str(fps), "-start_number", str(start), "-i", str(pattern)]
    cmd += [
        "-filter_complex",
        "[0:v][1:v][2:v]hstack=inputs=3,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(render / "compare.mp4"),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"openjd_status: Wrote {render / 'compare.mp4'}")


if __name__ == "__main__":
    main()
