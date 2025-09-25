# verify_env.py
import importlib

def check(import_name, display_name=None):
    name = display_name or import_name
    try:
        m = importlib.import_module(import_name)
        ver = getattr(m, "__version__", None) or getattr(m, "__file__", "version unknown")
        print(f"{name} -> OK: {ver}")
        if import_name == "tensorflow":
            try:
                print("  tf built with cuda:", m.test.is_built_with_cuda())
                print("  GPUs:", m.config.list_physical_devices("GPU"))
            except Exception as e:
                print("  TF GPU check error:", e)
    except Exception as e:
        print(f"{name} -> FAILED: {e}")

checks = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("talib", "TA-Lib (python wrapper)"),
    ("tensorflow", "tensorflow (includes Keras)"),
    ("backtrader", "backtrader"),
    ("streamlit", "streamlit"),
    ("plotly", "plotly")
]

for imp, label in checks:
    check(imp, label)
