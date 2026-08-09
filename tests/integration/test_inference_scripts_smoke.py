import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _run(script_module, *args):
    return subprocess.run(
        [sys.executable, "-m", script_module, *args],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )

def test_generate_script_runs_and_prints_image():
    r = _run("scripts.inference.generate", "--modality", "image", "--prompt", "cat", "--steps", "1")
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert "image" in r.stdout.lower() or "shape" in r.stdout.lower()

def test_generate_script_text_runs():
    r = _run("scripts.inference.generate", "--modality", "text", "--prompt", "hi", "--steps", "2")
    assert r.returncode == 0, f"stderr: {r.stderr}"
