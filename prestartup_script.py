"""ComfyUI-MoGe2 Prestartup Script."""

from pathlib import Path

from comfy_env import copy_files, setup_env

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy example images so the bundled workflows can find them via LoadImage.
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input", "**/*")
