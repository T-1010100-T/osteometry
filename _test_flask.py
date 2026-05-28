# -*- coding: utf-8 -*-
import sys, os, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.chdir(project_root)

import threading
import time
import urllib.request
import json

from src.utils.mediapipe_config import ensure_mediapipe_env
ensure_mediapipe_env()

import app as flask_app

server_thread = threading.Thread(target=lambda: flask_app.socketio.run(
    flask_app.app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True
), daemon=True)
server_thread.start()

time.sleep(3)
print("Server should be up, calling /api/start...", flush=True)

try:
    req = urllib.request.Request(
        'http://127.0.0.1:5000/api/start',
        data=b'',
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())
        print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}", flush=True)
except Exception as e:
    import traceback
    print(f"Request failed: {e}", flush=True)
    traceback.print_exc()

flask_app.state.is_running = False
print("Done", flush=True)