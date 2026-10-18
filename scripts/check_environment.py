"""Report whether this machine can run the bot. Usage: python scripts/check_environment.py"""
import importlib
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ok = True
    print("python", platform.python_version(), platform.platform())
    for mod in ("numpy", "PIL", "cv2", "requests", "onnxruntime", "onnx"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:12s} OK  {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"  {mod:12s} MISSING")
            ok = False
    try:
        import onnxruntime as ort
        print("  providers:", ort.get_available_providers())
    except ImportError:
        pass
    try:
        import resource  # noqa: F401
        with open("/proc/meminfo") as fh:
            print("  ", fh.readline().strip())
    except (ImportError, OSError):
        pass
    from app.config import ConfigError, load_settings
    try:
        s = load_settings(os.environ)
        print("configuration: OK (models:", ", ".join(m.path.name for m in s.models) + ")")
    except ConfigError as exc:
        print("configuration: INVALID -", exc)
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
