"""Job in background: la UI di Streamlit resta reattiva e un rerun non fa ripartire il lavoro."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional


class Job:
    def __init__(self, kind: str, key: str):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.key = key
        self.status = "running"  # running | done | error | cancelled
        self.stage = ""
        self.done = 0.0
        self.total = 1.0
        self.message = ""
        self.log: List[str] = []
        self.result: Any = None
        self.error: Optional[str] = None
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()

    def set_progress(self, stage: str, done: float, total: float, message: str) -> None:
        with self._lock:
            self.stage = stage
            self.done = float(done)
            self.total = float(total) if total else 1.0
            self.message = message
            if message and (not self.log or self.log[-1] != message):
                self.log.append(message)
                if len(self.log) > 200:
                    self.log = self.log[-200:]

    @property
    def fraction(self) -> float:
        if self.status == "done":
            return 1.0
        return max(0.0, min(1.0, self.done / self.total if self.total else 0.0))

    @property
    def running(self) -> bool:
        return self.status == "running"

    @property
    def elapsed(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    def cancel(self) -> None:
        self.cancel_event.set()


_REGISTRY: Dict[str, Job] = {}
_REGISTRY_LOCK = threading.Lock()


def start_job(kind: str, key: str, target: Callable[[Job], Any]) -> Job:
    existing = find_running(kind, key)
    if existing is not None:
        return existing
    job = Job(kind, key)

    def runner() -> None:
        try:
            job.result = target(job)
            job.status = "cancelled" if job.cancel_event.is_set() else "done"
        except Exception as exc:  # noqa: BLE001
            if job.cancel_event.is_set():
                job.status = "cancelled"
            else:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.log.append(traceback.format_exc()[-2000:])
        finally:
            job.finished_at = time.time()

    with _REGISTRY_LOCK:
        _REGISTRY[job.id] = job
    thread = threading.Thread(target=runner, name=f"{kind}-{job.id}", daemon=True)
    thread.start()
    return job


def get_job(job_id: Optional[str]) -> Optional[Job]:
    if not job_id:
        return None
    with _REGISTRY_LOCK:
        return _REGISTRY.get(job_id)


def find_running(kind: str, key: str) -> Optional[Job]:
    with _REGISTRY_LOCK:
        for job in _REGISTRY.values():
            if job.kind == kind and job.key == key and job.running:
                return job
    return None


def running_jobs() -> List[Job]:
    with _REGISTRY_LOCK:
        return [job for job in _REGISTRY.values() if job.running]
