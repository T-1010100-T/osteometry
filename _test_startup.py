# -*- coding: utf-8 -*-
import sys, os, traceback, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.chdir(project_root)

print("Project root:", project_root, flush=True)

from src.utils.mediapipe_config import ensure_mediapipe_env
ensure_mediapipe_env()
print("mediapipe env OK", flush=True)

print("Importing modules...", flush=True)
from src.core.camera_controller import CameraController, CameraConfig
print("  camera_controller OK", flush=True)
from src.core.measurement_engine import MeasurementEngine
print("  measurement_engine OK", flush=True)
from src.core.stability_detector import StabilityDetector
print("  stability_detector OK", flush=True)
from src.core.depth_config import load_depth_config
print("  depth_config OK", flush=True)
from src.core import HolisticEstimator
print("  holistic_estimator OK", flush=True)
from src.core.keypoint_stabilizer import KeypointStabilizer
print("  keypoint_stabilizer OK", flush=True)
from src.core.stabilizer_config import get_stabilizer_preset
print("  stabilizer_config OK", flush=True)

print("", flush=True)
print("=" * 60, flush=True)
print("Step 1: load_depth_config", flush=True)
print("=" * 60, flush=True)
depth_config = load_depth_config()
print(f"depth_config OK", flush=True)

print("", flush=True)
print("=" * 60, flush=True)
print("Step 2: CameraController + start", flush=True)
print("=" * 60, flush=True)
camera = CameraController(CameraConfig(width=640, height=480, fps=30, enable_imu=False, depth_config=depth_config))
print("CameraController created", flush=True)
camera_state = camera.start()
print(f"camera_state OK: mode={camera_state.mode} transformer={'YES' if camera_state.transformer else 'NO'}", flush=True)

print("", flush=True)
print("=" * 60, flush=True)
print("Step 3: HolisticEstimator", flush=True)
print("=" * 60, flush=True)
estimator = HolisticEstimator(model_complexity=1)
print(f"HolisticEstimator OK", flush=True)
estimator.close()

print("", flush=True)
print("=" * 60, flush=True)
print("Step 4: KeypointStabilizer", flush=True)
print("=" * 60, flush=True)
sc = get_stabilizer_preset('measurement')
sc.history_size = 10
ks = KeypointStabilizer(config=sc)
print(f"KeypointStabilizer OK", flush=True)

print("", flush=True)
print("=" * 60, flush=True)
print("Step 5: MeasurementEngine", flush=True)
print("=" * 60, flush=True)
me = MeasurementEngine(transformer=camera_state.transformer, keypoint_stabilizer=ks)
print(f"MeasurementEngine OK", flush=True)

print("", flush=True)
print("=" * 60, flush=True)
print("Step 6: StabilityDetector", flush=True)
print("=" * 60, flush=True)
sd = StabilityDetector(window_size=15, body_threshold=0.015, hand_threshold=0.008)
print(f"StabilityDetector OK", flush=True)

print("", flush=True)
print("=" * 60, flush=True)
print("ALL DONE!", flush=True)
print("=" * 60, flush=True)

camera.stop()