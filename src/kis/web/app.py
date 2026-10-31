"""FastAPI app for browsing visual search results."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kis import db
from kis.config import Settings
from kis.search.engine import SearchEngine, format_timestamp

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = SearchEngine(
            settings.db_path,
            settings.faiss_path,
            settings.manifest_path,
            settings.embed_model,
            settings.embed_text_template,
            translate_query=settings.translate_query,
            translate_model=settings.translate_model,
        )
        yield

    app = FastAPI(title="KIS visual search", lifespan=lifespan)

    app.mount(
        "/media/keyframes",
        StaticFiles(directory=str(settings.keyframes_dir), check_dir=False),
        name="keyframes",
    )
    app.mount(
        "/media/thumbnails",
        StaticFiles(directory=str(settings.thumbnails_dir), check_dir=False),
        name="thumbnails",
    )

    @app.get("/")
    def index_page() -> FileResponse:
        page = STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=404, detail="UI file not found")
        return FileResponse(page, media_type="text/html")

    @app.get("/api/health")
    def health() -> dict:
        engine: SearchEngine = app.state.engine
        conn = db.connect(settings.db_path)
        try:
            payload = {
                "videos": db.count_rows(conn, "videos"),
                "keyframes": db.count_rows(conn, "keyframes"),
                "indexed": engine.store.ntotal,
                "dim": engine.store.dim,
                "model": engine.model_id,
                "partial": engine.store.partial,
            }
        finally:
            conn.close()
        return payload

    @app.get("/api/search")
    def search(
        q: str = Query("", description="Vietnamese or English query"),
        k: int | None = Query(None, ge=1, le=100),
    ) -> dict:
        engine: SearchEngine = app.state.engine
        return engine.search(q, k or settings.search_topk)

    @app.get("/api/keyframe/{keyframe_pk}")
    def keyframe(keyframe_pk: int) -> dict:
        conn = db.connect(settings.db_path)
        try:
            rows = db.get_keyframes_by_pks(conn, [keyframe_pk])
        finally:
            conn.close()
        row = rows.get(keyframe_pk)
        if row is None:
            raise HTTPException(status_code=404, detail="keyframe not found")
        return {
            **row,
            "t_label": format_timestamp(int(row["t_ms"])),
            "image_url": "/media/" + row["image_path"].replace("\\", "/").lstrip("/"),
            "thumb_url": "/media/" + row["thumb_path"].replace("\\", "/").lstrip("/"),
            "video_url": f"/api/video/{row['video_id']}#t={int(row['t_ms']) / 1000.0:.3f}",
        }

    @app.get("/api/video/{video_id}")
    def video(video_id: str) -> FileResponse:
        conn = db.connect(settings.db_path)
        try:
            row = db.get_video(conn, video_id)
        finally:
            conn.close()
        if row is None:
            raise HTTPException(status_code=404, detail="video not found")
        path = Path(row["video_path"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="video file missing on disk")
        return FileResponse(path, media_type="video/mp4")

    return app
