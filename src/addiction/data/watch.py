# file: src/addiction/data/watch.py
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Set

from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent, FileMovedEvent
from watchdog.observers import Observer

import pandas as pd

from .ingest import load_schema, process_one  # reuse cleaner + marker writer
from ..utilities.config import Config, load_config
from ..utilities.logging import get_logger


def _is_csv(path: Path) -> bool:
    return path.suffix.lower() == ".csv"


def _is_done_marker(path: Path) -> bool:
    return path.suffixes[-2:] == [".csv", ".done.json"] or path.name.endswith(".done.json")


def _stable(path: Path, wait_secs: float = 0.25, retries: int = 8) -> bool:
    """File considered stable if size unchanged over `retries` intervals."""
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
        if cur != prev:
            prev = cur
            continue
        # unchanged once; check one more time for safety
        time.sleep(wait_secs)
        try:
            nxt = path.stat().st_size
        except FileNotFoundError:
            return False
        return nxt == cur
    return False


class IngestHandler(FileSystemEventHandler):
    def __init__(self, cfg: Config, schema_path: Path, debounce_ms: int = 500, max_workers: int = 2):
        super().__init__()
        self.cfg = cfg
        self.schema_path = schema_path
        self.schema = load_schema(schema_path)
        self.log = get_logger("watch", cfg=cfg, to_file=True)
        self.debounce_ms = debounce_ms
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._scheduled: Set[Path] = set()

    # ---- schedule on create/modify/move ----
    def on_created(self, event: FileCreatedEvent) -> None:
        self._maybe_schedule(Path(event.src_path))

    def on_modified(self, event: FileModifiedEvent) -> None:
        self._maybe_schedule(Path(event.src_path))

    def on_moved(self, event: FileMovedEvent) -> None:
        self._maybe_schedule(Path(event.dest_path))

    # ---- helpers ----
    def _maybe_schedule(self, path: Path) -> None:
        if path.is_dir():
            return
        if _is_done_marker(path):
            return
        if not _is_csv(path):
            return
        # Skip already processed
        if path.with_suffix(path.suffix + ".done.json").exists():
            return
        # Debounce multiple rapid events
        if path in self._scheduled:
            return
        self._scheduled.add(path)
        self.executor.submit(self._process_when_stable, path)

    def _process_when_stable(self, path: Path) -> None:
        try:
            if not _stable(path, wait_secs=max(0.05, self.debounce_ms / 1000.0), retries=10):
                self.log.warning("File not stable yet, skipping for now: %s", path.name)
                # allow rescheduling on next fs event
                self._scheduled.discard(path)
                return
            self._ingest_one(path)
        finally:
            # allow future reprocessing if file changes again
            self._scheduled.discard(path)

    def _ingest_one(self, path: Path) -> None:
        try:
            # sanity check: readable CSV
            _ = pd.read_csv(path, nrows=5)
        except Exception as e:
            self.log.exception("Unreadable CSV %s: %s", path.name, e)
            return
        try:
            result = process_one(path, self.cfg, self.schema, self.log)
            self.log.info("Processed: %s", json.dumps(result["report"]))
        except Exception as e:
            self.log.exception("Failed to ingest %s: %s", path.name, e)


def run_watch(cfg: Config, schema_path: Path, debounce_ms: int = 500, max_workers: int = 2) -> None:
    interim = cfg.paths.interim_dir
    interim.mkdir(parents=True, exist_ok=True)
    handler = IngestHandler(cfg, schema_path, debounce_ms=debounce_ms, max_workers=max_workers)

    observer = Observer()
    observer.schedule(handler, path=str(interim), recursive=False)
    handler.log.info("Watching %s for CSV files… (debounce=%dms, workers=%d)", interim, debounce_ms, max_workers)
    observer.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        handler.log.info("Stopping watcher…")
        observer.stop()
    observer.join()
    handler.executor.shutdown(wait=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Continuous watcher: validate/clean interim CSVs -> processed + .done.json")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--schema", default="config/schema.yaml")
    ap.add_argument("--debounce-ms", type=int, default=500, help="Stability debounce in milliseconds")
    ap.add_argument("--workers", type=int, default=2, help="Concurrent ingestion workers")
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        run_watch(cfg, Path(args.schema), debounce_ms=args.debounce_ms, max_workers=args.workers)
        return 0
    except Exception as e:
        log = get_logger("watch", cfg=cfg, to_file=True)
        log.exception("Watcher failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
