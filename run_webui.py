import argparse
import os
from pathlib import Path

import uvicorn
import yaml


def load_config(path: Path) -> dict:
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/webui.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    webui_cfg = config.get("webui", {})
    host = webui_cfg.get("bind_host", "127.0.0.1")
    port = int(webui_cfg.get("port", 8000))

    os.environ["AICHECKER_CONFIG"] = str(args.config)

    uvicorn.run("webui.backend.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
