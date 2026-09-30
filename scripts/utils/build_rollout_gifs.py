"""Build labelled rollout GIFs from the recordings written by
``scripts/utils/record_rollout_episodes.py``.

Every camera row in each recording is a rendered frame, captioned with the run's
batch size and its logged training eval success score. For each run a standalone
GIF is written, and a combined GIF tiles the runs into a grid (default 2x2) so
they play side by side. The combined grid can show a subset of cameras with
``--combined-cameras`` (by default it shows the main ``agentview`` camera so the
grid stays readable). All frames are streamed one at a time, so memory stays
bounded even for long recordings.

Examples:
    uv run python scripts/utils/build_rollout_gifs.py \\
        --recordings-dir assets/rollout-two-camera/_recordings \\
        --out-dir assets/rollout-two-camera --scale 1 --frame-stride 12 --fps 8

    # keep every camera in the combined grid
    uv run python scripts/utils/build_rollout_gifs.py \\
        --recordings-dir assets/rollout-two-camera/_recordings \\
        --out-dir assets/rollout-two-camera --combined-cameras agentview robot0_eye_in_hand
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RECORDINGS_FILENAME = "recordings.json"
DEFAULT_COMBINED_CAMERAS = ["agentview"]
DEFAULT_COMBINED_NAME = "two_camera_rollout_grid.gif"
LABEL_MARGIN = 6
FONT_FAMILY = "DejaVu Sans"
MIN_FONT_SIZE = 9
BACKGROUND_COLOR = (16, 16, 16)
TEXT_COLOR = (255, 255, 255)
LABEL_FONT_SIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        required=True,
        help="directory containing recordings.json and the per-run mp4 files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory for the generated GIFs",
    )
    parser.add_argument("--scale", type=int, default=2, help="integer upscale factor")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="keep every Nth recorded frame (controls GIF length/size)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="output GIF fps (default: the recording fps)",
    )
    parser.add_argument("--rows", type=int, default=2, help="combined grid rows")
    parser.add_argument("--cols", type=int, default=2, help="combined grid columns")
    parser.add_argument("--gap", type=int, default=4, help="pixels between grid cells")
    parser.add_argument(
        "--combined-cameras",
        nargs="+",
        default=list(DEFAULT_COMBINED_CAMERAS),
        help="cameras shown in the combined grid (default: agentview only)",
    )
    parser.add_argument(
        "--combined-name",
        default=DEFAULT_COMBINED_NAME,
        help="output filename for the combined GIF",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        default=None,
        help="restrict to these run names (repeatable)",
    )
    parser.add_argument(
        "--no-per-run",
        action="store_true",
        help="skip the per-run GIFs",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="only write the per-run GIFs",
    )
    return parser.parse_args()


def font_path() -> str:
    from matplotlib import font_manager

    return font_manager.findfont(FONT_FAMILY)


def fit_font(
    *,
    lines: list[str],
    max_width: int,
    max_size: int) -> ImageFont.FreeTypeFont:
    path = font_path()
    size = max_size
    while size > MIN_FONT_SIZE:
        font = ImageFont.truetype(path, size=size)
        if all(font.getlength(line) <= max_width for line in lines):
            return font
        size -= 1
    return ImageFont.truetype(path, size=MIN_FONT_SIZE)


def label_lines_for(*, record: dict[str, Any]) -> list[str]:
    label = record.get("label")
    if isinstance(label, list) and label:
        return [str(line) for line in label]
    batch = record.get("batch_size")
    header = f"bs={batch}" if batch is not None else record["run_name"][:24]
    rate = record.get("train_eval_success_rate")
    if rate is None:
        rate = record["success_rate"]
    return [header, f"eval {rate:.2f}"]


def label_geometry(
    *,
    fit_lines: list[str],
    line_count: int,
    panel_width: int,
    scale: int) -> tuple[ImageFont.FreeTypeFont, int]:
    margin = LABEL_MARGIN * scale
    font = fit_font(
        lines=fit_lines,
        max_width=panel_width - 2 * margin,
        max_size=LABEL_FONT_SIZE * scale,
    )
    line_gap = max(2, font.size // 5)
    label_height = line_count * font.size + (line_count - 1) * line_gap + 2 * margin
    return font, label_height


def upscale_frame(*, frame: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return np.asarray(frame, dtype=np.uint8)
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    size = (image.width * scale, image.height * scale)
    return np.asarray(image.resize(size, Image.BICUBIC), dtype=np.uint8)


def make_panel(
    *,
    frame: np.ndarray,
    lines: list[str],
    scale: int,
    font: ImageFont.FreeTypeFont,
    label_height: int) -> np.ndarray:
    image = Image.fromarray(upscale_frame(frame=frame, scale=scale))
    canvas = Image.new(
        "RGB",
        (image.width, image.height + label_height),
        color=BACKGROUND_COLOR,
    )
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    margin = LABEL_MARGIN * scale
    line_gap = max(2, font.size // 5)
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill=TEXT_COLOR, font=font)
        y += font.size + line_gap
    return np.asarray(canvas, dtype=np.uint8)


def recording_dimensions(*, record: dict[str, Any]) -> tuple[int, int]:
    reader = imageio.get_reader(str(record["_recording_path"]))
    try:
        width, height = reader.get_meta_data()["size"]
    finally:
        reader.close()
    return width, height


def camera_indices(*, camera_names: list[str], selected: list[str]) -> list[int]:
    unknown = [name for name in selected if name not in camera_names]
    if unknown:
        raise SystemExit(
            f"unknown combined cameras {unknown}; available: {camera_names}"
        )
    return [camera_names.index(name) for name in selected]


def select_cameras_in_frame(
    *,
    frame: np.ndarray,
    camera_count: int,
    indices: list[int]) -> np.ndarray:
    if len(indices) == camera_count:
        return frame
    camera_width = frame.shape[1] // camera_count
    parts = [frame[:, index * camera_width : (index + 1) * camera_width] for index in indices]
    return np.concatenate(parts, axis=1)


def write_single_gif(
    *,
    record: dict[str, Any],
    out_path: Path,
    scale: int,
    frame_stride: int,
    font: ImageFont.FreeTypeFont,
    label_height: int,
    writer_kwargs: dict[str, Any]) -> int:
    reader = imageio.get_reader(str(record["_recording_path"]))
    lines = label_lines_for(record=record)
    writer = imageio.get_writer(str(out_path), mode="I", **writer_kwargs)
    kept = 0
    try:
        for index, frame in enumerate(reader):
            if index % frame_stride != 0:
                continue
            panel = make_panel(
                frame=frame,
                lines=lines,
                scale=scale,
                font=font,
                label_height=label_height,
            )
            writer.append_data(panel)
            kept += 1
    finally:
        writer.close()
        reader.close()
    return kept


def grid_frame_count(*, record: dict[str, Any], frame_stride: int) -> int:
    reader = imageio.get_reader(str(record["_recording_path"]))
    try:
        total = reader.count_frames()
    finally:
        reader.close()
    return math.ceil(total / frame_stride)


def compose_grid(
    *,
    panels: list[np.ndarray | None],
    rows: int,
    cols: int,
    panel_width: int,
    panel_height: int,
    gap: int) -> np.ndarray:
    width = cols * panel_width + (cols + 1) * gap
    height = rows * panel_height + (rows + 1) * gap
    canvas = np.full((height, width, 3), BACKGROUND_COLOR, dtype=np.uint8)
    for position, panel in enumerate(panels):
        if panel is None:
            continue
        row, column = divmod(position, cols)
        x = gap + column * (panel_width + gap)
        y = gap + row * (panel_height + gap)
        canvas[y : y + panel_height, x : x + panel_width] = panel
    return canvas


def write_combined_gif(
    *,
    records: list[dict[str, Any]],
    out_path: Path,
    scale: int,
    frame_stride: int,
    rows: int,
    cols: int,
    gap: int,
    font: ImageFont.FreeTypeFont,
    panel_width: int,
    panel_height: int,
    label_height: int,
    selected_cameras: list[str],
    writer_kwargs: dict[str, Any]) -> int:
    readers = [imageio.get_reader(str(record["_recording_path"])) for record in records]
    labels = [label_lines_for(record=record) for record in records]
    selections = [
        camera_indices(camera_names=list(record["camera_names"]), selected=selected_cameras)
        for record in records
    ]
    counts = [grid_frame_count(record=record, frame_stride=frame_stride) for record in records]
    total_frames = max(counts)
    last_panels: list[np.ndarray | None] = [None] * len(records)

    writer = imageio.get_writer(str(out_path), mode="I", **writer_kwargs)
    try:
        for frame_index in range(total_frames):
            panels: list[np.ndarray | None] = []
            for cell, reader in enumerate(readers):
                if frame_index < counts[cell]:
                    raw = reader.get_data(frame_index * frame_stride)
                    raw = select_cameras_in_frame(
                        frame=raw,
                        camera_count=len(records[cell]["camera_names"]),
                        indices=selections[cell],
                    )
                    panel = make_panel(
                        frame=raw,
                        lines=labels[cell],
                        scale=scale,
                        font=font,
                        label_height=label_height,
                    )
                    last_panels[cell] = panel
                panels.append(last_panels[cell])
            grid = compose_grid(
                panels=panels,
                rows=rows,
                cols=cols,
                panel_width=panel_width,
                panel_height=panel_height,
                gap=gap,
            )
            writer.append_data(grid)
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    return total_frames


def load_records(*, recordings_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((recordings_dir / RECORDINGS_FILENAME).read_text())
    records = payload["recordings"]
    for record in records:
        record["_recording_path"] = recordings_dir / record["recording"]
    records.sort(key=lambda record: (record.get("batch_size") or 0))
    return records


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(recordings_dir=args.recordings_dir)
    if args.run_name is not None:
        wanted = set(args.run_name)
        records = [record for record in records if record["run_name"] in wanted]
    if not records:
        raise SystemExit("no recordings found")

    fps = args.fps if args.fps is not None else int(records[0]["fps"])
    writer_kwargs = {"fps": fps, "loop": 0}

    raw_width, raw_height = recording_dimensions(record=records[0])
    line_count = max(len(label_lines_for(record=record)) for record in records)

    panel_width = raw_width * args.scale
    fit_lines = [line for record in records for line in label_lines_for(record=record)]
    font, label_height = label_geometry(
        fit_lines=fit_lines,
        line_count=line_count,
        panel_width=panel_width,
        scale=args.scale,
    )
    panel_height = raw_height * args.scale + label_height

    for record in records:
        if args.no_per_run:
            break
        out_path = args.out_dir / f"bs{record['batch_size']}_two_camera_rollout.gif"
        frames = write_single_gif(
            record=record,
            out_path=out_path,
            scale=args.scale,
            frame_stride=args.frame_stride,
            font=font,
            label_height=label_height,
            writer_kwargs=writer_kwargs,
        )
        print(f"wrote {out_path} ({frames} frames)")

    if not args.no_combined:
        first_camera_names = list(records[0]["camera_names"])
        first_indices = camera_indices(
            camera_names=first_camera_names,
            selected=args.combined_cameras,
        )
        per_camera_width = raw_width // len(first_camera_names)
        combined_panel_width = per_camera_width * len(first_indices) * args.scale
        combined_font, combined_label_height = label_geometry(
            fit_lines=fit_lines,
            line_count=line_count,
            panel_width=combined_panel_width,
            scale=args.scale,
        )
        combined_h = raw_height * args.scale + combined_label_height

        combined_path = args.out_dir / args.combined_name
        frames = write_combined_gif(
            records=records,
            out_path=combined_path,
            scale=args.scale,
            frame_stride=args.frame_stride,
            rows=args.rows,
            cols=args.cols,
            gap=args.gap,
            font=combined_font,
            panel_width=combined_panel_width,
            panel_height=combined_h,
            label_height=combined_label_height,
            selected_cameras=args.combined_cameras,
            writer_kwargs=writer_kwargs,
        )
        print(f"wrote {combined_path} ({frames} frames)")


if __name__ == "__main__":
    main()
