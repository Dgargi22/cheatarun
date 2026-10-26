"""
diagnose_mujoco_gl.py — find out which MuJoCo rendering backend actually
works on this machine, and why the others fail.

Run:  python diagnose_mujoco_gl.py

Each backend is tested in its own subprocess so there's no leftover
import state from a previous attempt polluting the result.
"""
import subprocess
import sys
import textwrap

BACKENDS = ["glfw", "egl", "osmesa"]

CHILD_TEMPLATE = """
import os, sys, traceback
os.environ["MUJOCO_GL"] = {backend!r}

try:
    from dm_control import _render
    print("RESOLVED_BACKEND =", getattr(_render, "BACKEND", "?"))
    print("RENDERER =", _render.Renderer)
    print("RESULT: SUCCESS")
except Exception as e:
    print("RESULT: FAILED")
    print("EXCEPTION TYPE:", type(e).__name__)
    print("EXCEPTION MSG:", e)
    print("--- full traceback ---")
    traceback.print_exc()
    sys.exit(1)
"""

def test_backend(name):
    code = CHILD_TEMPLATE.format(backend=name)
    print(f"\n{'='*70}\nTesting MUJOCO_GL={name}\n{'='*70}")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr.strip()[-2000:])  # tail of traceback, avoid flooding
    return proc.returncode == 0


def test_actual_render():
    """Once a backend resolves, actually try to render a frame — importing
    _render successfully does NOT guarantee rendering works."""
    code = textwrap.dedent("""
        import os
        os.environ.setdefault("MUJOCO_GL", "glfw")
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        env = suite.load(domain_name="cheetah", task_name="run")
        env = pixels.Wrapper(env, pixels_only=True,
                              render_kwargs={"height": 64, "width": 64, "camera_id": 0})
        ts = env.reset()
        img = ts.observation["pixels"]
        print("RENDER_OK shape=", img.shape, "dtype=", img.dtype)
    """)
    print(f"\n{'='*70}\nActually rendering a frame (current MUJOCO_GL env)\n{'='*70}")
    proc = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=60)
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr.strip()[-2000:])


if __name__ == "__main__":
    results = {}
    for b in BACKENDS:
        results[b] = test_backend(b)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for b, ok in results.items():
        print(f"  {b:8s}: {'OK (imports cleanly)' if ok else 'FAILED'}")

    working = [b for b, ok in results.items() if ok]
    if working:
        print(f"\nUse: MUJOCO_GL={working[0]}  (set this in your dmc_common.py / shell)")
        test_actual_render()
    else:
        print("\nNone of glfw/egl/osmesa import cleanly on this Python install.")
        print("See the tracebacks above for the real root cause of each failure.")