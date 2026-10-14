"""Download public model files; no paid inference service is used."""

import json
from pathlib import Path

from . import DEFAULT_MODEL, ROOT


def model_path() -> Path:
    manifest = ROOT / "runs/model.json"
    if manifest.exists():
        info = json.loads(manifest.read_text())
        path = Path(info["snapshot"])
        # Prefer this project's relocated cache. Historical training manifests keep
        # their original absolute paths; moving the project must not require editing
        # those experiment records or downloading the same model a second time.
        revision = info.get("revision", path.name)
        local = ROOT / ".cache/hub" / ("models--" + info.get("model", DEFAULT_MODEL).replace("/", "--")) / "snapshots" / revision
        if local.exists():
            return local.resolve()
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("先运行 python -m microduck_qwen.download 下载 Qwen 模型。")


def main():
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        DEFAULT_MODEL,
        cache_dir=ROOT / ".cache/hub",
        allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "README.md"],
    )
    manifest = ROOT / "runs/model.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "model": DEFAULT_MODEL, "snapshot": path,
        "revision": Path(path).name,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"Model ready: {path}", flush=True)


if __name__ == "__main__":
    main()
