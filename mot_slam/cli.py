import argparse
from itertools import islice
from pathlib import Path
import sys
import cv2
from .geometry import LocalizationError
from .io import Sequence, write_results
from .pipeline import Config, StereoIMUSLAM


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stereo + IMU local mapping from synchronized KITTI Raw")
    parser.add_argument("sequence", type=Path, help=".../2011_09_26_drive_0005_sync")
    parser.add_argument("--calib-dir", type=Path, help="KITTI date directory containing three calibration files")
    parser.add_argument("--left", default="image_00")
    parser.add_argument("--right", default="image_01")
    parser.add_argument("--camera-json", type=Path, help="Custom fx,fy,cx,cy,baseline; requires --orientations")
    parser.add_argument("--orientations", type=Path, help="CSV: frame,timestamp,qx,qy,qz,qw; camera-to-world")
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--feature", choices=["sift", "surf"], default="sift")
    parser.add_argument("--stereo-interval", type=int, default=25)
    parser.add_argument("--motion-threshold", type=int, default=30)
    parser.add_argument("--disable-moving-removal", action="store_true", help="Ablation; RANSAC remains enabled")
    parser.add_argument("--save-masks", action="store_true")
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    results, names = [], []
    try:
        sequence = Sequence(args.sequence, args.left, args.right, args.calib_dir, args.camera_json, args.orientations)
        slam = StereoIMUSLAM(sequence.camera, Config(feature=args.feature, stereo_interval=args.stereo_interval,
                            motion_threshold=args.motion_threshold, remove_moving=not args.disable_moving_removal))
        frames = iter(sequence) if args.max_frames is None else islice(sequence, args.max_frames)
        failure = None
        try:
            for frame in frames:
                result = slam.process(frame.left, frame.right, frame.R_wc, frame.timestamp)
                results.append(result)
                names.append(frame.name)
                if args.save_masks:
                    masks = args.output/"masks"
                    masks.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(masks/frame.name), result.motion_mask):
                        raise OSError("Failed to save motion mask")
                # Images need not accumulate with the trajectory.
                result.motion_mask = None
                if len(results) % 25 == 0:
                    print(f"{len(results)} frames | {result.landmarks} landmarks | {result.reprojection_error:.2f} px")
        except (LocalizationError, ValueError, OSError) as exc:
            failure = str(exc)
        write_results(args.output, results, slam.map_points(), names)
        print(f"Saved {len(results)} valid poses and sparse map to {args.output}")
        if failure:
            print(f"Stopped: {failure}. Output contains only the valid trajectory prefix.", file=sys.stderr)
            return 2
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
