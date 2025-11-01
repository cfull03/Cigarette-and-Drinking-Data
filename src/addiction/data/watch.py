# file: src/addiction/data/watch.py
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock, Timer
from typing import Optional, Set

import pandas as pd
from watchdog.events import PatternMatchingEventHandler, FileCreatedEvent, FileModifiedEvent, FileMovedEvent
from watchdog.observers import Observer

from .ingest import load_schema, process_one, TransformConfig
from ..utilities.config import Config, load_config
from ..utilities.logging import get_logger


def _done_marker_for(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".done.json")

def _is_done_marker(path: Path) -> bool:
    return path.name.endswith(".done.json") and ".csv" in path.name

def _stable(path: Path, wait_secs: float = 0.25, retries: int = 8) -> bool:
    try:
        prev = path.stat().st_size
    except FileNotFoundError:
        return False
    for _ in range(retries):
        time.sleep(wait_secs)
        try:
            cur = path.stat().st_size
        except FileNotFoundError:
            return False
        if cur != prev or cur == 0:
            prev = cur
            continue
        time.sleep(wait_secs)
        try:
            nxt = path.stat().st_size
        except FileNotFoundError:
            return False
        return nxt == cur and nxt > 0
    return False


class _Debouncer:
    def __init__(self, delay_sec: float = 0.6):
        self.delay_sec = delay_sec
        self._timers: dict[Path, Timer] = {}
        self._lock = Lock()
    def schedule(self, key: Path, fn, *args, **kwargs) -> None:
        with self._lock:
            t = self._timers.get(key)
            if t and t.is_alive(): t.cancel()
            nt = Timer(self.delay_sec, fn, args=args, kwargs=kwargs)
            self._timers[key] = nt; nt.start()


class IngestHandler(PatternMatchingEventHandler):
    def __init__(self, cfg: Config, schema_path: Path, tcfg: TransformConfig, debounce_ms: int = 500, max_workers: int = 2):
        super().__init__(patterns=["*.csv"], ignore_directories=True, case_sensitive=False)
        self.cfg = cfg
        self.schema_path = schema_path
        self.schema = load_schema(schema_path)
        self.tcfg = tcfg
        self.log = get_logger("watch", cfg=cfg, to_file=True)
        self.debounce_ms = debounce_ms
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._scheduled: Set[Path] = set()
        self._sched_lock = Lock()
        self._debounce = _Debouncer(delay_sec=max(0.05, debounce_ms / 1000.0))

    def on_created(self, event: FileCreatedEvent) -> None:
        self._maybe_schedule(Path(event.src_path))
    def on_modified(self, event: FileModifiedEvent) -> None:
        self._maybe_schedule(Path(event.src_path))
    def on_moved(self, event: FileMovedEvent) -> None:
        self._maybe_schedule(Path(event.dest_path))

    def _maybe_schedule(self, path: Path) -> None:
        if _is_done_marker(path) or _done_marker_for(path).exists(): return
        with self._sched_lock:
            if path in self._scheduled: return
            self._scheduled.add(path)
        self._debounce.schedule(path, self._process_when_stable, path)

    def _process_when_stable(self, path: Path) -> None:
        try:
            if not _stable(path, wait_secs=max(0.05, self.debounce_ms / 1000.0), retries=10):
                self.log.warning("Not stable yet; retry on next event: %s", path.name)
                with self._sched_lock: self._scheduled.discard(path)
                return
            self.executor.submit(self._ingest_one, path)
        except Exception as e:
            self.log.exception("Scheduling error for %s: %s", path.name, e)
            with self._sched_lock: self._scheduled.discard(path)

    def _ingest_one(self, path: Path) -> None:
        try:
            pd.read_csv(path, nrows=3)
        except Exception as e:
            self.log.exception("Unreadable CSV %s: %s", path.name, e)
            with self._sched_lock: self._scheduled.discard(path)
            return
        try:
            result = process_one(path, self.cfg, self.schema, self.log, tcfg=self.tcfg)
            self.log.info("Processed: %s", json.dumps(result.get("report", {})))
        except Exception as e:
            self.log.exception("Failed to ingest %s: %s", path.name, e)
        finally:
            with self._sched_lock: self._scheduled.discard(path)


def _process_existing(interim: Path, handler: IngestHandler) -> None:
    for p in sorted(interim.glob("*.csv")):
        if _done_marker_for(p).exists(): continue
        handler._maybe_schedule(p)


def run_watch(cfg: Config, schema_path: Path, tcfg: TransformConfig, debounce_ms: int = 500, max_workers: int = 2) -> None:
    interim = cfg.paths.interim_dir; interim.mkdir(parents=True, exist_ok=True)
    handler = IngestHandler(cfg, schema_path, tcfg, debounce_ms=debounce_ms, max_workers=max_workers)
    observer = Observer(); observer.schedule(handler, path=str(interim), recursive=False)
    handler.log.info("Watching %s (debounce=%dms, workers=%d, outliers=%s, k=%.2f, csv=%s, parquet=%s)",
                    interim, debounce_ms, max_workers, tcfg.outlier_mode, tcfg.iqr_k, tcfg.save_csv, tcfg.save_parquet)
    _process_existing(interim, handler)
    observer.start()
    try:
        while True: time.sleep(1.0)
    except KeyboardInterrupt:
        handler.log.info("Stopping watcher…"); observer.stop()
    observer.join(); handler.executor.shutdown(wait=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Watch interim/ for CSVs → process → .done.json")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--schema", default="config/schema.yaml")
    ap.add_argument("--debounce-ms", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--outliers", choices=["off", "iqr"], default="iqr")
    ap.add_argument("--iqr-k", type=float, default=1.5)
    ap.add_argument("--save-csv", action="store_true")
    ap.add_argument("--no-parquet", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = TransformConfig(
        outlier_mode=args.outliers,
        iqr_k=args.iqr_k,
        save_csv=bool(args.save_csv),
        save_parquet=not bool(args.no_parquet),
    )
    run_watch(cfg, Path(args.schema), tcfg, debounce_ms=args.debounce_ms, max_workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
