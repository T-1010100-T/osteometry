# -*- coding: utf-8 -*-
import sys, os, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.chdir(project_root)
print(f"CWD: {os.getcwd()}", flush=True)

from src.utils.mediapipe_config import ensure_mediapipe_env
ensure_mediapipe_env()
print(f"MediaPipe env configured, cache: {os.environ.get('MEDIAPIPE_CACHE_DIR')}", flush=True)

print("About to import app...", flush=True)
import app
print("App imported successfully!", flush=True)

# Now call start_camera
with app.app.test_request_context():
    try:
        result = app.start_camera()
        print(f"start_camera result: {result}", flush=True)
    except Exception as e:
        import traceback
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc()

print("DONE", flush=True)