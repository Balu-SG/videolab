"""Hosted VideoLab on Modal: HTTPS (needed for the phone camera) plus a volume for runs and uploads.

    cd frontend && npm run build && cd ..
    modal secret create videolab-meta MODEL_API_KEY=...   # once
    modal deploy modal_app.py
"""

import modal

volume = modal.Volume.from_name("videolab-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .env({"VIDEOLAB_DATA": "/data"})
    .add_local_dir("backend", "/root/backend", ignore=["__pycache__"])
    .add_local_dir("frontend/dist", "/root/frontend/dist")
)

app = modal.App("videolab", image=image)


# One container: measurement jobs and chat sessions live in its memory.
@app.function(
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("videolab-meta")],
    cpu=2,
    memory=2048,
    max_containers=1,
    scaledown_window=900,
    timeout=900,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    from backend import main

    main.on_run_saved = volume.commit
    return main.app
