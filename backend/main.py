from fastapi import FastAPI

app = FastAPI(title="VideoLab")


@app.get("/health")
def health() -> dict:
    return {"ok": True}
