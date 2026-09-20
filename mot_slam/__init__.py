"""Paper-inspired stereo/IMU localization and local mapping, without object tracking."""
from .geometry import Camera, LocalizationError
from .pipeline import Config, StereoIMUSLAM

__all__ = ["Camera", "Config", "LocalizationError", "StereoIMUSLAM"]
