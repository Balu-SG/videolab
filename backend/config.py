import os
from pathlib import Path

from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parent.parent
# Runs and uploads. On the hosted app this is a persistent volume.
OUTPUT_DIR = Path(os.environ.get("VIDEOLAB_DATA", REPO_DIR / "output"))

# The key lives outside the repo (../.env) so it can never be committed.
load_dotenv(REPO_DIR.parent / ".env")
load_dotenv(REPO_DIR / ".env")

META_BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"

# Guards for the public deployment, so one visitor can't spend the API credits.
MAX_UPLOAD_MB = int(os.environ.get("VIDEOLAB_MAX_UPLOAD_MB", "300"))
MAX_FRAMES = int(os.environ.get("VIDEOLAB_MAX_FRAMES", "3600"))  # 15 s at 240 fps, about $1.50 of SAM
MAX_RUNS_PER_DAY = int(os.environ.get("VIDEOLAB_MAX_RUNS_PER_DAY", "40"))
MAX_CHATS_PER_DAY = int(os.environ.get("VIDEOLAB_MAX_CHATS_PER_DAY", "600"))


def api_key() -> str:
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set (expected in ../.env or the environment)")
    return key
