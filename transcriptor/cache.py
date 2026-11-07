"""Cache su disco: audio convertito, blocchi parziali, trascrizione e pacchetti studio, per hash del file."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def cache_root() -> Path:
    env = os.environ.get("TRANSCRIPTOR_CACHE_DIR")
    root = Path(env).expanduser() if env else Path.home() / ".cache" / "transcriptor"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def studypack_key(**params) -> str:
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


class ChunkStore:
    """Risultati parziali per blocco: permette di riprendere una trascrizione interrotta."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.plan_path = self.directory / "plan.json"

    def ensure_plan(self, plan: List[dict]) -> None:
        if self.plan_path.exists():
            try:
                if _read_json(self.plan_path) == plan:
                    return
            except Exception:
                pass
            for item in self.directory.glob("*.json"):
                item.unlink()
        _write_json(self.plan_path, plan)

    def load(self, index: int) -> Optional[List[dict]]:
        path = self.directory / f"{index:04d}.json"
        if not path.exists():
            return None
        try:
            return _read_json(path)
        except Exception:
            return None

    def save(self, index: int, segments: List[dict]) -> None:
        _write_json(self.directory / f"{index:04d}.json", segments)

    def clear(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


class LessonCache:
    def __init__(self, file_hash: str, root: Optional[Path] = None):
        self.file_hash = file_hash
        self.directory = (root or cache_root()) / file_hash
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def wav_path(self) -> Path:
        return self.directory / "audio16k.wav"

    @property
    def transcript_path(self) -> Path:
        return self.directory / "transcript.json"

    @property
    def source_info_path(self) -> Path:
        return self.directory / "source.json"

    def chunk_store(self) -> ChunkStore:
        return ChunkStore(self.directory / "chunks")

    # --- sorgente ---
    def save_source_info(self, name: str, size: int, path: Optional[str] = None) -> None:
        info = {"name": name, "size": size, "path": path, "saved_at": time.time()}
        _write_json(self.source_info_path, info)

    def load_source_info(self) -> Optional[Dict]:
        if not self.source_info_path.exists():
            return None
        try:
            return _read_json(self.source_info_path)
        except Exception:
            return None

    def source_copy_path(self, suffix: str) -> Path:
        return self.directory / f"source{suffix.lower()}"

    def find_source_copy(self) -> Optional[Path]:
        for item in self.directory.glob("source.*"):
            if item.suffix != ".json":
                return item
        return None

    # --- trascrizione ---
    def has_transcript(self) -> bool:
        return self.transcript_path.exists()

    def load_transcript(self) -> Optional[Tuple[List[dict], Dict]]:
        if not self.has_transcript():
            return None
        try:
            data = _read_json(self.transcript_path)
            return data["segments"], data["meta"]
        except Exception:
            return None

    def save_transcript(self, segments: List[dict], meta: Dict) -> None:
        _write_json(self.transcript_path, {"segments": segments, "meta": meta})

    # --- pacchetto studio ---
    def studypack_path(self, key: str) -> Path:
        return self.directory / f"studypack_{key}.json"

    def load_studypack(self, key: str) -> Optional[Dict]:
        path = self.studypack_path(key)
        if not path.exists():
            return None
        try:
            return _read_json(path)
        except Exception:
            return None

    def save_studypack(self, key: str, pack: Dict) -> None:
        _write_json(self.studypack_path(key), pack)

    def list_studypacks(self) -> List[Path]:
        return sorted(self.directory.glob("studypack_*.json"))

    def clear(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


def list_cached_lessons(root: Optional[Path] = None) -> List[Dict]:
    root = root or cache_root()
    lessons: List[Dict] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        cache = LessonCache(directory.name, root)
        info = cache.load_source_info() or {}
        lessons.append({
            "hash": directory.name,
            "name": info.get("name") or directory.name[:12],
            "size": info.get("size"),
            "saved_at": info.get("saved_at") or directory.stat().st_mtime,
            "has_transcript": cache.has_transcript(),
            "studypacks": len(cache.list_studypacks()),
        })
    lessons.sort(key=lambda item: item["saved_at"], reverse=True)
    return lessons


def cache_size_bytes(root: Optional[Path] = None) -> int:
    root = root or cache_root()
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def clear_cache(root: Optional[Path] = None) -> None:
    root = root or cache_root()
    for directory in root.iterdir():
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
