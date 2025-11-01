# file: tests/test_watch.py
# pytest integration test; no real processing: monkeypatch ingest funcs.

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from watchdog.observers import Observer

# Import the watcher components
from addiction.data.watch import IngestHandler, _process_existing, _done_marker_for


@pytest.mark.parametrize("debounce_ms", [100])
def test_watch_processes_new_csv(tmp_path: Path, monkeypatch, debounce_ms: int):
    # Arrange minimal cfg with interim dir
    interim = tmp_path / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    cfg = SimpleNamespace(paths=SimpleNamespace(interim_dir=interim))

    # Monkeypatch load_schema -> simple stub
    def fake_load_schema(_path: Path):
        return {"ok": True}

    # Monkeypatch process_one -> write .done marker and return report
    def fake_process_one(path: Path, cfg, schema, log):
        marker = _done_marker_for(path)
        marker.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {"report": {"rows_in": 1, "rows_out": 1}}

    # Monkeypatch logger to avoid file I/O
    class FakeLog:
        def info(self, *a, **k): ...
        def warning(self, *a, **k): ...
        def exception(self, *a, **k): ...

    def fake_get_logger(*a, **k):
        return FakeLog()

    monkeypatch.setattr("addiction.data.watch.load_schema", fake_load_schema)
    monkeypatch.setattr("addiction.data.watch.process_one", fake_process_one)
    monkeypatch.setattr("addiction.data.watch.get_logger", fake_get_logger)

    # Create handler & observer directly (not run_watch) so we can stop cleanly
    handler = IngestHandler(cfg=cfg, schema_path=tmp_path / "schema.yaml", debounce_ms=debounce_ms, max_workers=1)
    observer = Observer()
    observer.schedule(handler, path=str(interim), recursive=False)

    try:
        observer.start()
        # Backlog: none yet
        _process_existing(interim, handler)

        # Act: drop a CSV
        csv_path = interim / "sample.csv"
        pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)

        # Wait for marker (up to ~5s)
        marker = _done_marker_for(csv_path)
        for _ in range(50):
            if marker.exists():
                break
            time.sleep(0.1)

        # Assert
        assert marker.exists(), "Expected .done.json marker to be written"
        data = json.loads(marker.read_text(encoding="utf-8"))
        assert data.get("ok") is True
    finally:
        observer.stop()
        observer.join()
        handler.executor.shutdown(wait=True)


def test_process_existing_on_start(tmp_path: Path, monkeypatch):
    interim = tmp_path / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    cfg = SimpleNamespace(paths=SimpleNamespace(interim_dir=interim))

    def fake_load_schema(_path: Path): return {"ok": True}
    def fake_process_one(path: Path, cfg, schema, log):
        _done_marker_for(path).write_text("{}", encoding="utf-8")
        return {"report": {}}
    class FakeLog:
        def info(self, *a, **k): ...
        def warning(self, *a, **k): ...
        def exception(self, *a, **k): ...
    def fake_get_logger(*a, **k): return FakeLog()

    monkeypatch.setattr("addiction.data.watch.load_schema", fake_load_schema)
    monkeypatch.setattr("addiction.data.watch.process_one", fake_process_one)
    monkeypatch.setattr("addiction.data.watch.get_logger", fake_get_logger)

    # Pre-drop CSV
    csv1 = interim / "backlog.csv"
    csv1.write_text("a\n1\n", encoding="utf-8")

    handler = IngestHandler(cfg=cfg, schema_path=tmp_path / "schema.yaml", debounce_ms=50, max_workers=1)
    # Process backlog synchronously via helper
    _process_existing(interim, handler)

    # Wait for marker (up to ~2s)
    marker = _done_marker_for(csv1)
    for _ in range(20):
        if marker.exists():
            break
        time.sleep(0.1)

    assert marker.exists(), "Backlog CSV should be processed at startup"
