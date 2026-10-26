import os
import sys
import time
import signal
import socket
import threading
import subprocess

# Ensure UTF-8 output on Windows
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(ROOT_DIR, ".venv", "Scripts", "python.exe")
PYTHON_EXE = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a local TCP port is already open/listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def kill_process_on_port(port: int) -> None:
    """Finds and kills any dangling process listening on the specified port on Windows."""
    if os.name != 'nt':
        return
    try:
        cmd = f'netstat -ano | findstr :{port}'
        output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
        for line in output.strip().splitlines():
            parts = line.split()
            if len(parts) >= 5 and "LISTENING" in line:
                pid = parts[-1]
                if pid.isdigit() and int(pid) > 0 and int(pid) != os.getpid():
                    print(f"[CLEANUP] Freeing port {port} by terminating dangling PID {pid}...")
                    subprocess.run(f"taskkill /F /T /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(0.5)
    except Exception:
        pass


def stream_output(pipe, prefix: str, color_code: str):
    """Streams lines from process stdout/stderr with a tagged prefix."""
    try:
        for line in iter(pipe.readline, ''):
            if not line:
                break
            stripped = line.rstrip()
            if stripped:
                print(f"{color_code}{prefix}\033[0m {stripped}", flush=True)
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def main():
    print("=" * 70)
    print("  SatyaDrishti Unified Launcher — Starting Full Stack")
    print("=" * 70, flush=True)

    # 1. Clean up stale ports
    print("[1/3] Checking ports (5000 for backend, 8080 for frontend)...")
    kill_process_on_port(5000)
    kill_process_on_port(8080)

    # 2. Start Flask Backend
    print("[2/3] Launching Flask Backend API on port 5000...")
    backend_env = os.environ.copy()
    backend_env["PYTHONUNBUFFERED"] = "1"
    backend_env["PYTHONPATH"] = ROOT_DIR

    backend_cmd = [PYTHON_EXE, os.path.join(ROOT_DIR, "backend", "run.py")]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=ROOT_DIR,
        env=backend_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    t_backend = threading.Thread(
        target=stream_output,
        args=(backend_proc.stdout, "[BACKEND] ", "\033[36m"),
        daemon=True
    )
    t_backend.start()

    # Wait for backend to be ready on port 5000
    print("      Waiting for backend to listen on http://127.0.0.1:5000...")
    for _ in range(30):
        if is_port_in_use(5000):
            print("      ✓ Backend is online on http://127.0.0.1:5000")
            break
        if backend_proc.poll() is not None:
            print("[ERROR] Backend process exited unexpectedly! Check logs above.")
            sys.exit(1)
        time.sleep(0.5)

    # 3. Start Vite Frontend
    print("[3/3] Launching Vite Frontend on port 8080...")
    run_vite = os.path.join(ROOT_DIR, "run-vite.js")
    frontend_cmd = ["node", run_vite, "--port", "8080", "--host", "127.0.0.1"]

    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    t_frontend = threading.Thread(
        target=stream_output,
        args=(frontend_proc.stdout, "[FRONTEND]", "\033[32m"),
        daemon=True
    )
    t_frontend.start()

    # Wait for frontend to be ready on port 8080
    for _ in range(60):
        if is_port_in_use(8080):
            print("      ✓ Frontend is online on http://localhost:8080")
            break
        if frontend_proc.poll() is not None:
            print("[ERROR] Frontend process exited unexpectedly! Check logs above.")
            break
        time.sleep(0.5)

    print("\n" + "=" * 70)
    print("  🚀 SatyaDrishti Full-Stack Application is RUNNING!")
    print("  --------------------------------------------------")
    print("  🌐 Frontend UI:     http://localhost:8080")
    print("  ⚙️  Backend API:     http://127.0.0.1:5000")
    print("  🩺 API Health:      http://127.0.0.1:5000/health")
    print("  --------------------------------------------------")
    print("  Press Ctrl+C at any time to stop both servers cleanly.")
    print("=" * 70 + "\n", flush=True)

    def shutdown(signum=None, frame=None):
        print("\n[SHUTDOWN] Stopping all SatyaDrishti services cleanly...")
        try:
            frontend_proc.terminate()
            time.sleep(0.2)
            if frontend_proc.poll() is None:
                frontend_proc.kill()
        except Exception:
            pass

        try:
            backend_proc.terminate()
            time.sleep(0.2)
            if backend_proc.poll() is None:
                backend_proc.kill()
        except Exception:
            pass

        kill_process_on_port(5000)
        kill_process_on_port(8080)
        print("[SHUTDOWN] Both backend and frontend stopped successfully.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Monitor loop
    try:
        while True:
            b_code = backend_proc.poll()
            f_code = frontend_proc.poll()
            if b_code is not None:
                print(f"[WARN] Backend exited with code {b_code}.")
                shutdown()
                break
            if f_code is not None:
                print(f"[WARN] Frontend exited with code {f_code}.")
                shutdown()
                break
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
