import os
from pathlib import Path

from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_DIR / "output"

# The key lives outside the repo (../.env) so it can never be committed.
load_dotenv(REPO_DIR.parent / ".env")
load_dotenv(REPO_DIR / ".env")

META_BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"


def api_key() -> str:
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set (expected in ../.env or the environment)")
    return key
