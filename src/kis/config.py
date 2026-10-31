"""Runtime settings for the keyframe pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KIS_", env_file=".env", extra="ignore")

    dataset_dir: Path = Path("Dataset/Video")
    index_dir: Path = Path("index")

    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"

    grid_s: float = 3.0
    max_per_shot: int = 6
    min_shot_s: float = 0.4
    long_shot_s: float = 10.0
    # Plan 3.3 can also degrade any video containing a shot > long_shot_s.
    # Off by default because hybrid already grids inside long shots; enabling
    # it on news footage sends every video to the 1 fps fallback.
    degrade_on_long_shot: bool = False
    min_gap_ms: int = 800
    # pHash may not drop a frame if the next candidate is farther than this
    # from the last kept keyframe (protects temporal coverage).
    phash_max_gap_ms: int = 2000
    # Global safety grid on top of shot picks. Needed to satisfy the plan's
    # coverage gate (a 3 s grid alone gives only ~2/3 coverage at +-1 s).
    safety_grid_s: float = 2.0

    keyframe_long_side: int = 384
    thumb_long_side: int = 224
    jpeg_quality: int = 85
    thumb_quality: int = 80

    shots_backend: str = "auto"
    scene_threshold: float = 0.30

    embed_model: str = "google/siglip2-base-patch16-384"
    embed_batch: int = 32
    embed_dtype: str = "float16"
    # SigLIP-2 text tower expects caption-like text; the prompt template is
    # applied at query time only (image shards are unaffected).
    embed_text_template: str = "This is a photo of {}."
    # SigLIP-2's multilingual text tower is weak on Vietnamese in practice.
    # An optional VI->EN MarianMT layer is available but off by default until
    # measured on a real query set (enable with KIS_TRANSLATE_QUERY=1).
    translate_query: bool = False
    translate_model: str = "Helsinki-NLP/opus-mt-vi-en"
    search_topk: int = 20
    web_host: str = "127.0.0.1"
    web_port: int = 8000

    @property
    def db_path(self) -> Path:
        return self.index_dir / "meta.db"

    @property
    def keyframes_dir(self) -> Path:
        return self.index_dir / "keyframes"

    @property
    def thumbnails_dir(self) -> Path:
        return self.index_dir / "thumbnails"

    @property
    def embeddings_dir(self) -> Path:
        return self.index_dir / "embeddings"

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"


def sampling_config(settings: Settings) -> dict:
    """Picklable subset passed to worker processes."""
    return {
        "grid_s": settings.grid_s,
        "max_per_shot": settings.max_per_shot,
        "min_shot_s": settings.min_shot_s,
        "long_shot_s": settings.long_shot_s,
        "degrade_on_long_shot": settings.degrade_on_long_shot,
        "min_gap_ms": settings.min_gap_ms,
        "phash_max_gap_ms": settings.phash_max_gap_ms,
        "safety_grid_s": settings.safety_grid_s,
        "keyframe_long_side": settings.keyframe_long_side,
        "thumb_long_side": settings.thumb_long_side,
        "jpeg_quality": settings.jpeg_quality,
        "thumb_quality": settings.thumb_quality,
    }
