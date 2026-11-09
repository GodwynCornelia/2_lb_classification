"""Open local TensorBoard logs, including projects with Unicode Windows paths."""
import os
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--additional", action="store_true", help="Show the additional 5-fold assignment")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    logs = root / "logs" / ("additional" if args.additional else "fit")
    snapshot = root / "additional" / "artifacts" / "tensorboard" if args.additional else root / "artifacts" / "tensorboard"
    if not logs.is_dir():
        logs = snapshot
    if not logs.is_dir():
        raise SystemExit("No TensorBoard logs found: run generate_notebook.py first.")
    with tempfile.TemporaryDirectory(prefix="lab2_view_tensorboard_") as staging:
        readable_logs = logs
        if os.name == "nt" and not str(logs).isascii():
            readable_logs = Path(staging) / "logs"
            shutil.copytree(logs, readable_logs)
        port = "6007" if args.additional else "6006"
        print(f"TensorBoard: http://localhost:{port} (Ctrl+C to stop)", flush=True)
        subprocess.run([sys.executable, "-m", "tensorboard.main", "--logdir",
                        str(readable_logs), "--port", port], check=True)


if __name__ == "__main__":
    main()
