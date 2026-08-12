"""Record a fixed-depth DIGIT/TAXIM indentation sequence.

The object is lowered by 0.1 mm before each frame, producing 30 PNG images
over a total travel of 3 mm.  The images are then encoded as a timestamped
MP4 with ffmpeg.
"""

import argparse
from datetime import datetime
import math
from pathlib import Path
import shutil
import subprocess

import cv2
import mujoco as mj

from TaximSensor import TaximSensor
from norm2tex.normals import BUMP_DIRECTION


CURRENT_DIR = Path(__file__).resolve().parent
SCENE_FILE = CURRENT_DIR / "xml" / "touch_playground_rot.xml"
NORM2TEX_SCENE_FILE = CURRENT_DIR / "normal_xml" / "touch_playground_rot.xml"
NORM2TEX_NORMAL_MAP = (
    CURRENT_DIR
    / "normal_xml"
    / "assets"
    / "target_objs"
    / "golf_small_Fabric_normal.png"
)

# 0.1 mm = 0.0001 m.  Thirty steps therefore reach exactly 3 mm.
DEPTH_INCREMENT_M = 0.0001
MAX_DEPTH_M = 0.003
FRAME_COUNT = 30
DEFAULT_FPS = 30


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record a 30-frame, 3 mm TAXIM indentation sequence."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "recordings",
        help="Directory for the timestamped MP4 and PNG frame directory.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Output video frame rate (default: {DEFAULT_FPS}).",
    )
    parser.add_argument(
        "--raster-backend",
        choices=("cpu", "cuda"),
        default="cuda",
        help="TAXIM raster backend (CUDA falls back to CPU if unavailable).",
    )
    parser.add_argument(
        "--norm2tex",
        action="store_true",
        help=(
            "Render the UV-mapped golf ball with its Fabric normal-map texture "
            "instead of the default untextured object."
        ),
    )
    return parser.parse_args()


def encode_video(frame_dir: Path, video_path: Path, fps: int):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. The PNG frames were saved, but the "
            "video could not be created."
        )

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            "1",
            "-i",
            str(frame_dir / "frame_%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
    )


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")
    if not math.isclose(
        FRAME_COUNT * DEPTH_INCREMENT_M, MAX_DEPTH_M, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("Frame count and depth constants do not equal MAX_DEPTH_M")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.resolve()
    frame_dir = output_dir / f"{timestamp}_frames"
    video_path = output_dir / f"{timestamp}.mp4"
    frame_dir.mkdir(parents=True, exist_ok=False)

    scene_file = NORM2TEX_SCENE_FILE if args.norm2tex else SCENE_FILE
    model = mj.MjModel.from_xml_path(str(scene_file))
    data = mj.MjData(model)
    model.opt.timestep = 0.001
    mj.mj_forward(model, data)

    sensor = TaximSensor(
        sensor_type="digit",
        bg_file=None,
        bg_index=0,
        resize=None,
        gelmap_file="/home/sbien/Documents/Development/V2T/TactoSampler/taxim_files/gelmap_alt.npy",
        preprocess_bg=True,
        texture_bump_scale_mm=0.2 if args.norm2tex else 0.05,
        raster_backend=args.raster_backend,
    )
    geom_options = {}
    if args.norm2tex:
        geom_options = {
            "normal_map_path": str(NORM2TEX_NORMAL_MAP),
            "texture_map_direction": BUMP_DIRECTION.AWAY_FROM_SENSOR,
        }
    sensor.add_geom_mujoco(
        "can_geom", model=model, data=data, mesh_name="can_mesh", **geom_options
    )
    sensor.add_camera_mujoco("left_tacto_pad", model, data)
    sensor.set_sensor_pad_geom("finger_1_left_0")

    vertical_qpos = model.joint("xx").qposadr
    initial_height = float(data.qpos[vertical_qpos])

    for frame_number in range(1, FRAME_COUNT + 1):
        depth_m = frame_number * DEPTH_INCREMENT_M
        data.qpos[vertical_qpos] = initial_height - depth_m
        data.qvel[:] = 0.0
        mj.mj_forward(model, data)

        taxim_rgb, _, _ = sensor.render_taxim(
            model,
            data,
            shadow=True,
            get_depth=False,
            img_noise_sigma=0,
            pcn_add_noise=False,
            visualize=False,
            cycle_bg=False,
        )
        frame_path = frame_dir / f"frame_{frame_number:03d}.png"
        if not cv2.imwrite(
            str(frame_path), cv2.cvtColor(taxim_rgb, cv2.COLOR_RGB2BGR)
        ):
            raise RuntimeError(f"Could not save frame to {frame_path}")
        print(
            f"Saved frame {frame_number:02d}/{FRAME_COUNT} "
            f"at {depth_m * 1000:.1f} mm"
        )

    encode_video(frame_dir, video_path, args.fps)
    print(f"Saved frames to: {frame_dir}")
    print(f"Saved video to:  {video_path}")


if __name__ == "__main__":
    main()
