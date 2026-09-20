"""Synchronized KITTI Raw input and portable trajectory/cloud output."""
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import Camera, rotation_matrix


def calibration_fields(path):
    fields = {}
    for line in Path(path).read_text().splitlines():
        key, _, value = line.partition(":")
        try:
            fields[key] = np.array([float(x) for x in value.split()])
        except ValueError:
            continue  # KITTI's calib_time is metadata.
    return fields


@dataclass
class Frame:
    timestamp: float
    left: np.ndarray
    right: np.ndarray
    R_wc: np.ndarray
    name: str


class Sequence:
    def __init__(self, root, left="image_00", right="image_01", calib_dir=None,
                 camera_json=None, orientations=None):
        root = Path(root)
        self.left_files = sorted((root/left/"data").glob("*.png"))
        self.right_dir = root/right/"data"
        if not self.left_files:
            raise ValueError(f"No PNG frames in {root/left/'data'}")
        if camera_json is not None:
            if orientations is None:
                raise ValueError("A custom camera requires --orientations")
            values = json.loads(Path(camera_json).read_text())
            self.camera = Camera(**values)
        else:
            calib_dir = Path(calib_dir) if calib_dir else root.parent
            cc = calibration_fields(calib_dir/"calib_cam_to_cam.txt")
            li, ri = left[-2:], right[-2:]
            p = cc[f"P_rect_{li}"].reshape(3, 4)
            q = cc[f"P_rect_{ri}"].reshape(3, 4)
            if not np.allclose(p[:, :3], q[:, :3], atol=1e-5):
                raise ValueError("Rectified stereo cameras must share intrinsics")
            if abs(p[1, 3]-q[1, 3]) > 1e-4:
                raise ValueError("Expected horizontal rectified stereo")
            baseline = p[0, 3]/p[0, 0] - q[0, 3]/q[0, 0]
            self.camera = Camera(p[0, 0], p[1, 1], p[0, 2], p[1, 2], baseline)
            if orientations is None:
                cv = calibration_fields(calib_dir/"calib_velo_to_cam.txt")
                vi = calibration_fields(calib_dir/"calib_imu_to_velo.txt")
                # All P_rect cameras project from the rectified reference camera.
                R_camera_imu = cc["R_rect_00"].reshape(3, 3) @ cv["R"].reshape(3, 3) @ vi["R"].reshape(3, 3)
                self.R_imu_camera = rotation_matrix(R_camera_imu).T
        self.orientations = None
        if orientations is not None:
            with Path(orientations).open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.orientations = {}
            last = -np.inf
            for row in rows:
                timestamp = float(row["timestamp"])
                quat = np.array([float(row[k]) for k in ["qx", "qy", "qz", "qw"]])
                if not np.isfinite(timestamp) or timestamp <= last or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-9:
                    raise ValueError("Invalid timestamp or camera quaternion in orientations CSV")
                if row["frame"] in self.orientations:
                    raise ValueError("Duplicate frame in orientations CSV")
                self.orientations[row["frame"]] = (timestamp, Rotation.from_quat(quat).as_matrix())
                last = timestamp
        else:
            self.oxts = root/"oxts"/"data"
            lines = (root/left/"timestamps.txt").read_text().splitlines()
            self.times = np.array([np.datetime64(s.strip(), "ns") for s in lines])
            if len(self.times) != len(self.left_files) or np.isnat(self.times).any() or np.any(np.diff(self.times) <= np.timedelta64(0, "ns")):
                raise ValueError("KITTI timestamps must match frames and strictly increase")
        for path in self.left_files:
            if not (self.right_dir/path.name).is_file():
                raise ValueError(f"Missing right image: {path.name}")
            if self.orientations is not None and path.name not in self.orientations:
                raise ValueError(f"Missing orientation for {path.name}")
            if self.orientations is None and not (self.oxts/(path.stem+".txt")).is_file():
                raise ValueError(f"Missing OXTS packet for {path.name}; use a synced KITTI Raw drive")

    def __iter__(self):
        for i, path in enumerate(self.left_files):
            left = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            right = cv2.imread(str(self.right_dir/path.name), cv2.IMREAD_GRAYSCALE)
            if left is None or right is None:
                raise ValueError(f"Cannot decode stereo frame {path.name}")
            if self.orientations is not None:
                timestamp, R_wc = self.orientations[path.name]
            else:
                packet = np.loadtxt(self.oxts/(path.stem+".txt"))
                if packet.size < 6 or not np.isfinite(packet[3:6]).all():
                    raise ValueError(f"Invalid OXTS orientation: {path.name}")
                # KITTI roll/pitch/yaw: R_world_imu = Rz(yaw) Ry(pitch) Rx(roll).
                R_wc = Rotation.from_euler("xyz", packet[3:6]).as_matrix() @ self.R_imu_camera
                timestamp = float((self.times[i]-self.times[0])/np.timedelta64(1, "s"))
            yield Frame(timestamp, left, right, R_wc, path.name)


def write_results(directory, results, points, names):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"trajectory.tum").open("w") as stream:
        stream.write("# timestamp tx ty tz qx qy qz qw; camera-to-world, meters\n")
        for result in results:
            q = Rotation.from_matrix(result.R_wc).as_quat()
            stream.write(" ".join(f"{x:.9f}" for x in [result.timestamp, *result.position, *q])+"\n")
    # KITTI-compatible camera poses relative to the initial camera frame.
    with (directory/"poses.txt").open("w") as stream:
        if results:
            initial_inverse = np.linalg.inv(results[0].pose)
            for result in results:
                relative = initial_inverse @ result.pose
                stream.write(" ".join(f"{x:.9f}" for x in relative[:3].ravel())+"\n")
    with (directory/"diagnostics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "timestamp", "landmarks", "inliers", "median_reprojection_px", "stereo_refresh", "gamma"])
        for name, r in zip(names, results):
            writer.writerow([name, r.timestamp, r.landmarks, r.inliers, r.reprojection_error, r.refreshed, r.gamma])
    with (directory/"map.ply").open("w") as stream:
        stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(stream, points, fmt="%.6f")
