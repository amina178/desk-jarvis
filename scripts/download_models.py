"""Download the MediaPipe .task models into models/mediapipe/."""
import urllib.request

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from deskjarvis import config as cfg


def main():
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in cfg.MODEL_URLS.items():
        target = cfg.MODELS_DIR / f"{name}.task"
        if target.exists():
            print(f"[skip] {target.name} already exists")
            continue
        print(f"[get ] {name} <- {url}")
        urllib.request.urlretrieve(url, target)
    print("Done.")


if __name__ == "__main__":
    main()
