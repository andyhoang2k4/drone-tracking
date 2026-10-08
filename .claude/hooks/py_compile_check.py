"""PostToolUse hook: kiem tra cu phap file .py vua sua (khong ghi file .pyc). Loi -> exit 2."""
import json
import sys

try:
    data = json.load(sys.stdin)
    path = (data.get("tool_input") or {}).get("file_path", "")
except Exception:
    sys.exit(0)

if path.endswith(".py"):
    try:
        with open(path, encoding="utf-8") as f:
            compile(f.read(), path, "exec")
    except SyntaxError as e:
        print(f"Loi cu phap trong {path}, dong {e.lineno}: {e.msg}", file=sys.stderr)
        sys.exit(2)
    except OSError:
        pass
sys.exit(0)
