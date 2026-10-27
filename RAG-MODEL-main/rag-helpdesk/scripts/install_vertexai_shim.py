"""Copy the langchain_community Vertex AI chat shim into the active venv."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root
        / "compatibility"
        / "langchain_community"
        / "chat_models"
        / "vertexai.py"
    )
    if not source.is_file():
        print(f"Missing shim source: {source}", file=sys.stderr)
        return 1

    import sysconfig

    site_packages = Path(sysconfig.get_paths()["purelib"])
    target_dir = site_packages / "langchain_community" / "chat_models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "vertexai.py"
    shutil.copy2(source, target)
    print(f"Installed shim: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
