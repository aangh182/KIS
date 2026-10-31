"""Command line interface for the keyframe pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from kis import db
from kis.config import Settings, sampling_config
from kis.eval import ablation as ablation_mod
from kis.eval import coverage as coverage_mod
from kis.ingest.extract import process_video
from kis.ingest.probe import run_probe
from kis.ingest.shots import detect_shots
from kis.verify import verify_timestamps


def _default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings()
    if getattr(args, "index_dir", None):
        settings.index_dir = Path(args.index_dir)
    if getattr(args, "dataset_dir", None):
        settings.dataset_dir = Path(args.dataset_dir)
    return settings


def _shots_job(job: dict) -> dict:
    try:
        shots, backend = detect_shots(
            job["video_path"],
            job["duration_ms"],
            job["fps"],
            backend=job["backend"],
            min_shot_ms=job["min_shot_ms"],
            scene_threshold=job["scene_threshold"],
            ffmpeg=job["ffmpeg"],
        )
        return {"video_id": job["video_id"], "shots": shots, "backend": backend}
    except Exception as exc:  # noqa: BLE001
        return {"video_id": job["video_id"], "error": str(exc)}


def cmd_probe(args: argparse.Namespace) -> int:
    settings = _settings(args)
    result = run_probe(settings.db_path, settings.dataset_dir, settings.ffprobe)
    print(f"probed {result['nvideos']} files, {result['total']} videos in database")
    return 0


def cmd_shots(args: argparse.Namespace) -> int:
    settings = _settings(args)
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    videos = db.get_videos(conn)
    if not videos:
        print("no videos in database; run `kis probe` first", file=sys.stderr)
        return 1

    jobs = []
    for video in videos:
        if not args.force and db.get_shots(conn, video["video_id"]):
            continue
        jobs.append(
            {
                "video_id": video["video_id"],
                "video_path": video["video_path"],
                "duration_ms": int(video["duration_ms"]),
                "fps": video["fps_source"],
                "backend": args.backend,
                "min_shot_ms": int(settings.min_shot_s * 1000),
                "scene_threshold": args.threshold,
                "ffmpeg": settings.ffmpeg,
            }
        )

    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        print("all videos already have shots (use --force to redo)")
        conn.close()
        return 0

    errors = 0
    backends: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_shots_job, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="shots"):
            result = future.result()
            if "error" in result:
                errors += 1
                print(f"\n{result['video_id']}: {result['error']}", file=sys.stderr)
                continue
            db.replace_shots(conn, result["video_id"], result["shots"])
            backends[result["backend"]] = backends.get(result["backend"], 0) + 1

    print(f"processed {len(jobs) - errors}/{len(jobs)} videos; backends: {backends}")
    conn.close()
    return 1 if errors else 0


def cmd_keyframes(args: argparse.Namespace) -> int:
    settings = _settings(args)
    settings.grid_s = args.grid_s
    if args.max_per_shot is not None:
        settings.max_per_shot = args.max_per_shot

    conn = db.connect(settings.db_path)
    db.migrate(conn)
    videos = db.get_videos(conn)
    if not videos:
        print("no videos in database; run `kis probe` first", file=sys.stderr)
        conn.close()
        return 1

    videos_with_shots = []
    missing_shots = []
    for video in videos:
        if db.get_shots(conn, video["video_id"]):
            videos_with_shots.append(video)
        else:
            missing_shots.append(video["video_id"])
    if not videos_with_shots:
        print("no shots in database; run `kis shots` first", file=sys.stderr)
        conn.close()
        return 1
    if missing_shots:
        print(
            f"warning: {len(missing_shots)} videos without shots will be skipped "
            f"(e.g. {missing_shots[0]})",
            file=sys.stderr,
        )
    videos = videos_with_shots
    if args.limit:
        videos = videos[: args.limit]

    cfg = sampling_config(settings)
    jobs = [
        {
            "video_id": video["video_id"],
            "db_path": str(settings.db_path.resolve()),
            "index_dir": str(settings.index_dir.resolve()),
            "keyframes_dir": str(settings.keyframes_dir.resolve()),
            "thumbnails_dir": str(settings.thumbnails_dir.resolve()),
            "profile": args.profile,
            "cfg": cfg,
            "force": args.force,
            "ffmpeg": settings.ffmpeg,
            "ffprobe": settings.ffprobe,
        }
        for video in videos
    ]
    conn.close()

    errors = 0
    totals = {
        "n_candidates": 0,
        "n_keyframes": 0,
        "dropped_gap": 0,
        "dropped_phash": 0,
        "n_iframes": 0,
        "skipped": 0,
        "degraded": 0,
    }
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_video, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="keyframes"):
            result = future.result()
            if "error" in result:
                errors += 1
                print(f"\n{result['video_id']}: {result['error']}", file=sys.stderr)
                continue
            if result.get("skipped"):
                totals["skipped"] += 1
                continue
            if result.get("degraded"):
                totals["degraded"] += 1
            for key in ("n_candidates", "n_keyframes", "dropped_gap", "dropped_phash", "n_iframes"):
                totals[key] += result.get(key, 0)

    print(
        "keyframes: "
        f"{totals['n_keyframes']} kept / {totals['n_candidates']} candidates "
        f"(gap drops {totals['dropped_gap']}, pHash drops {totals['dropped_phash']}), "
        f"{totals['n_iframes']} I-frames, degraded {totals['degraded']}, "
        f"skipped {totals['skipped']}, errors {errors}"
    )
    return 1 if errors else 0


def cmd_eval_coverage(args: argparse.Namespace) -> int:
    settings = _settings(args)
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    summary = coverage_mod.compute_coverage(conn, radius_ms=args.radius)
    conn.close()

    report = coverage_mod.render_report(summary)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"wrote {report_path}")
    print(
        f"videos={summary['nvideos']} keyframes={summary['nkeyframes']} "
        f"coverage={summary['coverage_1s']:.4f} "
        f"max_gap p50/p95/p99/max={summary['max_gap_p50']}/"
        f"{summary['max_gap_p95']}/{summary['max_gap_p99']}/{summary['max_gap_max']} ms"
    )
    passed = coverage_mod.gate_passes(summary)
    print("coverage gate:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


def cmd_eval_ablation(args: argparse.Namespace) -> int:
    settings = _settings(args)
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    cfg = sampling_config(settings)
    summary = ablation_mod.compute_ablation(
        conn,
        ffprobe=settings.ffprobe,
        ffmpeg=settings.ffmpeg,
        cfg=cfg,
        radius_ms=args.radius,
        limit=args.limit,
    )
    conn.close()

    report = ablation_mod.render_report(summary)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"wrote {report_path}")
    for profile in ablation_mod.PROFILES:
        row = summary[profile]
        print(
            f"{profile}: candidates={row['n_candidates']} "
            f"coverage={row['coverage_1s']:.4f} "
            f"max_gap p99={row['max_gap_p99']} ms degraded={row['degraded']} "
            f"gate={'PASS' if ablation_mod.gate_passes(row) else 'FAIL'}"
        )
    return 0


def cmd_verify_timestamps(args: argparse.Namespace) -> int:
    settings = _settings(args)
    conn = db.connect(settings.db_path)
    db.migrate(conn)
    result = verify_timestamps(conn, args.sample, settings.ffprobe)
    conn.close()

    print(
        f"timestamp check: {result['passed']}/{result['checked']} exact "
        f"({result['failed']} failed)"
    )
    if args.show_failures:
        for row in result["results"]:
            if not row.get("ok"):
                print(row)
    return 0 if result["failed"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kis", description="KIS self-keyframe pipeline")
    parser.add_argument("--index-dir", help="artifact directory (default: index)")
    parser.add_argument("--dataset-dir", help="video directory (default: Dataset/Video)")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="ffprobe videos into the database")
    probe.add_argument("--dataset", dest="dataset_dir", help="video directory")
    probe.set_defaults(func=cmd_probe)

    shots = sub.add_parser("shots", help="detect shot boundaries")
    shots.add_argument("--workers", type=int, default=_default_workers())
    shots.add_argument("--backend", choices=["auto", "scenedetect", "ffmpeg"], default="auto")
    shots.add_argument("--threshold", type=float, default=0.30)
    shots.add_argument("--limit", type=int, help="process at most N pending videos")
    shots.add_argument("--force", action="store_true", help="re-detect existing videos")
    shots.set_defaults(func=cmd_shots)

    keyframes = sub.add_parser("keyframes", help="select and extract keyframes")
    keyframes.add_argument(
        "--profile", choices=["v0", "v1", "v2", "v3", "iframe", "grid", "shot", "hybrid"], default="hybrid"
    )
    keyframes.add_argument("--grid-s", type=float, default=3.0)
    keyframes.add_argument("--max-per-shot", type=int)
    keyframes.add_argument("--workers", type=int, default=_default_workers())
    keyframes.add_argument("--limit", type=int, help="process at most N videos")
    keyframes.add_argument("--force", action="store_true", help="reprocess existing videos")
    keyframes.set_defaults(func=cmd_keyframes)

    eval_parser = sub.add_parser("eval", help="evaluation gates")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    cov = eval_sub.add_parser("coverage", help="temporal coverage and max-gap gate")
    cov.add_argument("--report", default="eval/coverage.md")
    cov.add_argument("--radius", type=int, default=1000, help="coverage radius in ms")
    cov.set_defaults(func=cmd_eval_coverage)

    abl = eval_sub.add_parser("ablation", help="compare candidate profiles V0-V3")
    abl.add_argument("--report", default="eval/ablation.md")
    abl.add_argument("--radius", type=int, default=1000, help="coverage radius in ms")
    abl.add_argument("--limit", type=int, help="evaluate at most N videos")
    abl.set_defaults(func=cmd_eval_ablation)

    debug = sub.add_parser("debug", help="diagnostics")
    debug_sub = debug.add_subparsers(dest="debug_command", required=True)
    verify = debug_sub.add_parser("verify-timestamps", help="cross-check t_ms with ffprobe")
    verify.add_argument("--sample", type=int, default=100)
    verify.add_argument("--show-failures", action="store_true")
    verify.set_defaults(func=cmd_verify_timestamps)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
