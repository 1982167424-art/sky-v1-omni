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

def test_serve_script_engine_flag_enables_engine():
    """serve --engine 应构造带 SkyInferenceEngine 的 app（mock uvicorn.run 捕获 app 实例）"""
    import sky_v1
    if not sky_v1.API_AVAILABLE or not sky_v1.INFERENCE_AVAILABLE:
        import pytest
        pytest.skip("API or Inference module unavailable")
    import scripts.inference.serve as s
    import uvicorn

    captured = {}

    def _fake_run(app, host=None, port=None, **kwargs):
        captured["app"] = app

    # parse_args 应识别 --engine
    ns = s.parse_args(["--engine", "--port", "9999"])
    assert ns.engine is True
    assert ns.port == 9999

    # 驱动 main()（内部 parse_args 读 sys.argv，故临时 patch），并 mock uvicorn.run
    orig_argv = sys.argv
    orig_run = uvicorn.run
    sys.argv = ["serve", "--engine", "--port", "9999"]
    uvicorn.run = _fake_run
    try:
        rc = s.main()
    finally:
        sys.argv = orig_argv
        uvicorn.run = orig_run

    assert rc == 0
    app = captured.get("app")
    assert app is not None, "uvicorn.run 未被调用 / app 未构造"
    assert getattr(app.state, "engine", None) is not None, "Engine 模式未启用 SkyInferenceEngine"
