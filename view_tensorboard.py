"""Open local TensorBoard logs, including projects with Unicode Windows paths."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    logs = Path(__file__).resolve().parent / "logs" / "fit"
    if not logs.is_dir():
        logs = Path(__file__).resolve().parent / "artifacts" / "tensorboard"
    if not logs.is_dir():
        raise SystemExit("No TensorBoard logs found: run generate_notebook.py first.")
    with tempfile.TemporaryDirectory(prefix="lab2_view_tensorboard_") as staging:
        readable_logs = logs
        if os.name == "nt" and not str(logs).isascii():
            readable_logs = Path(staging) / "logs"
            shutil.copytree(logs, readable_logs)
        print("TensorBoard: http://localhost:6006 (Ctrl+C to stop)", flush=True)
        subprocess.run([sys.executable, "-m", "tensorboard.main", "--logdir",
                        str(readable_logs), "--port", "6006"], check=True)


if __name__ == "__main__":
    main()
