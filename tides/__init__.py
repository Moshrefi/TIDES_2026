"""TIDES: Time-Derivative Event Simulation via Deformable Reconstruction (re-implementation)."""
from .camera import Intrinsics, PoseTrajectory
from .renderer import TangentRenderer
from .simulator import TidesConfig, TidesSimulator

__all__ = ["Intrinsics", "PoseTrajectory", "TangentRenderer", "TidesConfig", "TidesSimulator"]
__version__ = "0.1.0"
