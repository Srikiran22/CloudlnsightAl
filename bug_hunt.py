import sys
from streamlit.testing.v1 import AppTest
from Utils.paths import ensure_project_directories

ensure_project_directories()

pages = [
    "App.py",
    "Pages/Upload.py",
    "Pages/Cleaning.py",
    "Pages/Compare.py",
    "Pages/EDA.py",
    "Pages/Visualization.py",
    "Pages/ML.py",
    "Pages/AI.py",
    "Pages/Report.py",
    "Pages/Dashboard.py",
]

failures = 0
for page in pages:
    try:
        at = AppTest.from_file(page, default_timeout=60)
        at.run()
        if at.exception:
            failures += 1
            print(f"[FAIL] {page}")
            for ex in at.exception:
                val = getattr(ex, "value", None) or getattr(ex, "message", None) or str(ex)
                print(f"       {type(ex).__name__}: {val}")
                stack = getattr(ex, "stack_trace", None)
                if stack:
                    print(f"       {stack}")
        else:
            print(f"[ OK ] {page}")
    except Exception as e:
        failures += 1
        print(f"[ERR ] {page}: {type(e).__name__}: {e}")

print(f"\n{failures} page(s) failing")
sys.exit(1 if failures else 0)
